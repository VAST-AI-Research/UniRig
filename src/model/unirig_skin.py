import torch
from torch import nn, FloatTensor, LongTensor, Tensor
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import pad
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoConfig
import math
import torch_scatter
from flash_attn.modules.mha import MHA

from .spec import ModelSpec, ModelInput
from .parse_encoder import MAP_MESH_ENCODER, get_mesh_encoder

from ..data.utils import linear_blend_skinning

class FrequencyPositionalEmbedding(nn.Module):
    """The sin/cosine positional embedding. Given an input tensor `x` of shape [n_batch, ..., c_dim], it converts
    each feature dimension of `x[..., i]` into:
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # only present if include_input is True.
        ], here f_i is the frequency.

    Denote the space is [0 / num_freqs, 1 / num_freqs, 2 / num_freqs, 3 / num_freqs, ..., (num_freqs - 1) / num_freqs].
    If logspace is True, then the frequency f_i is [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...];
    Otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)].

    Args:
        num_freqs (int): the number of frequencies, default is 6;
        logspace (bool): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
            otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)];
        input_dim (int): the input dimension, default is 3;
        include_input (bool): include the input tensor or not, default is True.

    Attributes:
        frequencies (torch.Tensor): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
                otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1);

        out_dim (int): the embedding size, if include_input is True, it is input_dim * (num_freqs * 2 + 1),
            otherwise, it is input_dim * num_freqs * 2.

    """

    def __init__(
        self,
        num_freqs: int = 6,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = True,
    ) -> None:
        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self._get_dims(input_dim)

    def _get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward process.

        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: an embedding of `x` of shape [..., dim * (num_freqs * 2 + temp)]
                where temp is 1 if include_input is True and 0 otherwise.
        """

        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies.to(device=x.device)).view(
                *x.shape[:-1], -1
            )
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x

class ResidualCrossAttn(nn.Module):
    def __init__(self, feat_dim: int, num_heads: int):
        super().__init__()
        assert feat_dim % num_heads == 0, "feat_dim must be divisible by num_heads"

        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        # self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
        self.attention = MHA(embed_dim=feat_dim, num_heads=num_heads, cross_attn=True)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Linear(feat_dim * 4, feat_dim),
        )
        
    def forward(self, q, kv):
        residual = q
        attn_output = self.attention(q, x_kv=kv)
        x = self.norm1(residual + attn_output)
        x = self.norm2(x + self.ffn(x))
        return x

class BoneEncoder(nn.Module):
    def __init__(
        self,
        feat_bone_dim: int,
        feat_dim: int,
        embed_dim: int,
        num_heads: int,
        num_attn: int,
    ):
        super().__init__()
        self.feat_bone_dim = feat_bone_dim
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.num_attn = num_attn
        
        self.position_embed = FrequencyPositionalEmbedding(input_dim=self.feat_bone_dim)

        self.bone_encoder = nn.Sequential(
            self.position_embed,
            nn.Linear(self.position_embed.out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )
        self.attn = nn.ModuleList()
        for _ in range(self.num_attn):
            self.attn.append(ResidualCrossAttn(feat_dim, self.num_heads))

    def forward(
        self,
        base_bone: FloatTensor,
        num_bones: LongTensor,
        parents: LongTensor,
        min_coord: FloatTensor,
        global_latents: FloatTensor,
    ):
        # base_bone: (B, J, C)
        B = base_bone.shape[0]
        J = base_bone.shape[1]
        x = self.bone_encoder((base_bone-min_coord[:, None, :]).reshape(-1, base_bone.shape[-1])).reshape(B, J, -1)

        latents = torch.cat([x, global_latents], dim=1)
        
        for (i, attn) in enumerate(self.attn):
            x = attn(x, latents)
        return x

class SkinweightPred(nn.Module):
    def __init__(self, in_dim, mlp_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

class UniRigSkin(ModelSpec):
    
    def process_fn(self, batch: List[ModelInput]) -> List[Dict]:
        max_bones = 0
        for b in batch:
            max_bones = max(max_bones, b.asset.J)
        res = []
        current_offset = 0
        for b in batch:
            vertex_groups = b.asset.sampled_vertex_groups
            current_offset += b.vertices.shape[0]
            # (N, J)
            voxel_skin = vertex_groups['voxel_skin']
            if 'skin' in vertex_groups:
                skin = vertex_groups['skin']
            else:
                skin = np.zeros_like(voxel_skin)
            
            voxel_skin = np.pad(voxel_skin, ((0, 0), (0, max_bones-b.asset.J)), 'constant', constant_values=0.0)
            skin = np.pad(skin, ((0, 0), (0, max_bones-b.asset.J)), 'constant', constant_values=0.0)
            
            # (J, 4, 4)
            res.append({
                'voxel_skin': voxel_skin,
                'skin': skin,
                'offset': current_offset,
            })
        return res
    
    def __init__(self, mesh_encoder, global_encoder, **kwargs):
        super().__init__()
        
        self.num_train_vertex       = kwargs['num_train_vertex']
        self.feat_dim               = kwargs['feat_dim']
        self.num_heads              = kwargs['num_heads']
        self.grid_size              = kwargs['grid_size']
        self.mlp_dim                = kwargs['mlp_dim']
        self.num_bone_attn          = kwargs['num_bone_attn']
        self.num_mesh_bone_attn     = kwargs['num_mesh_bone_attn']
        self.bone_embed_dim         = kwargs['bone_embed_dim']
        self.voxel_mask             = kwargs.get('voxel_mask', 2)

        self.mesh_encoder = get_mesh_encoder(**mesh_encoder)
        self.global_encoder = get_mesh_encoder(**global_encoder)
        if isinstance(self.mesh_encoder, MAP_MESH_ENCODER.ptv3obj):
            self.feat_map = nn.Sequential(
                nn.Linear(mesh_encoder['enc_channels'][-1], self.feat_dim),
                nn.LayerNorm(self.feat_dim),
                nn.GELU(),
            )
        else:
            raise NotImplementedError()
        if isinstance(self.global_encoder, MAP_MESH_ENCODER.michelangelo_encoder):
            self.out_proj = nn.Sequential(
                nn.Linear(self.global_encoder.width, self.feat_dim),
                nn.LayerNorm(self.feat_dim),
                nn.GELU(),
            )
        else:
            raise NotImplementedError()

        self.bone_encoder = BoneEncoder(
            feat_bone_dim=3,
            feat_dim=self.feat_dim,
            embed_dim=self.bone_embed_dim,
            num_heads=self.num_heads,
            num_attn=self.num_bone_attn,
        )
        
        self.downscale = nn.Sequential(
            nn.Linear(2 * self.num_heads, self.num_heads),
            nn.LayerNorm(self.num_heads),
            nn.GELU(),
        )
        self.skinweight_pred = SkinweightPred(
            self.num_heads,
            self.mlp_dim,
        )
        
        self.mesh_bone_attn = nn.ModuleList()
        self.mesh_bone_attn.extend([
            ResidualCrossAttn(self.feat_dim, self.num_heads) for _ in range(self.num_mesh_bone_attn)
        ])

        self.qmesh = nn.Linear(self.feat_dim, self.feat_dim * self.num_heads)
        self.kmesh = nn.Linear(self.feat_dim, self.feat_dim * self.num_heads)

        self.voxel_skin_embed = nn.Linear(1, self.num_heads)
        self.voxel_skin_norm = nn.LayerNorm(self.num_heads)
        self.attn_skin_norm = nn.LayerNorm(self.num_heads)

    def encode_mesh_cond(self, vertices: FloatTensor, normals: FloatTensor) -> FloatTensor:
        assert not torch.isnan(vertices).any()
        assert not torch.isnan(normals).any()
        if isinstance(self.global_encoder, MAP_MESH_ENCODER.michelangelo_encoder):
            if (len(vertices.shape) == 3):
                shape_embed, latents, token_num, pre_pc = self.global_encoder.encode_latents(pc=vertices, feats=normals)
            else:
                shape_embed, latents, token_num, pre_pc = self.global_encoder.encode_latents(pc=vertices.unsqueeze(0), feats=normals.unsqueeze(0))
            latents = self.out_proj(latents)
            return latents
        else:
            raise NotImplementedError()

    def _get_predict(self, batch: Dict) -> FloatTensor:
        '''
        Return predicted skin.
        '''
        
        num_bones: Tensor = batch['num_bones']
        vertices: FloatTensor = batch['vertices'] # (B, N, 3)
        normals: FloatTensor = batch['normals']
        joints: FloatTensor = batch['joints']
        tails: FloatTensor = batch['tails']
        voxel_skin: FloatTensor = batch['voxel_skin']
        parents: LongTensor = batch['parents']
        
        # turn inputs' dtype into model's dtype
        dtype = next(self.parameters()).dtype
        vertices = vertices.type(dtype)
        normals = normals.type(dtype)
        joints = joints.type(dtype)
        tails = tails.type(dtype)
        voxel_skin = voxel_skin.type(dtype)
        
        B = vertices.shape[0]
        N = vertices.shape[1]
        J = joints.shape[1]
        
        assert vertices.dim() == 3
        assert normals.dim() == 3
        
        part_offset = torch.tensor([(i+1)*N for i in range(B)], dtype=torch.int64, device=vertices.device)
        idx_ptr = torch.nn.functional.pad(part_offset, (1, 0), value=0)
        min_coord = torch_scatter.segment_csr(vertices.reshape(-1, 3), idx_ptr, reduce="min")

        pack = []
        if self.training:
            train_indices = torch.randperm(N)[:self.num_train_vertex]
            pack.append(train_indices)
        else:
            for i in range((N + self.num_train_vertex - 1) // self.num_train_vertex):
                pack.append(torch.arange(i*self.num_train_vertex, min((i+1)*self.num_train_vertex, N)))
        
        # (B, seq_len, feat_dim)
        global_latents = self.encode_mesh_cond(vertices, normals)
        bone_feat = self.bone_encoder(
            base_bone=joints,
            num_bones=num_bones,
            parents=parents,
            min_coord=min_coord,
            global_latents=global_latents,
        )
        
        if isinstance(self.mesh_encoder, MAP_MESH_ENCODER.ptv3obj):
            feat = torch.cat([vertices, normals, torch.zeros_like(vertices)], dim=-1)
            ptv3_input = {
                'coord': vertices.reshape(-1, 3),
                'feat': feat.reshape(-1, 9),
                'offset': torch.tensor(batch['offset']),
                'grid_size': self.grid_size,
            }
            if not self.training:
                # must cast to float32 to avoid sparse-conv precision bugs
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    mesh_feat = self.mesh_encoder(ptv3_input).feat
                    mesh_feat = self.feat_map(mesh_feat).view(B, N, self.feat_dim)
            else:
                mesh_feat = self.mesh_encoder(ptv3_input).feat
                mesh_feat = self.feat_map(mesh_feat).view(B, N, self.feat_dim)
            mesh_feat = mesh_feat.type(dtype)
        else:
            raise NotImplementedError()

        # (B, J + seq_len, feat_dim)
        latents = torch.cat([bone_feat, global_latents], dim=1)
        # (B, N, feat_dim)
        for block in self.mesh_bone_attn:
            mesh_feat = block(
                q=mesh_feat,
                kv=latents,
            )

        # trans to (B, num_heads, J, feat_dim)
        bone_feat = self.kmesh(bone_feat).view(B, J, self.num_heads, self.feat_dim).transpose(1, 2)

        skin_pred_list = []
        if not self.training:
            skin_mask = voxel_skin.clone()
            for b in range(B):
                num = num_bones[b]
                for i in range(num):
                    p = parents[b, i]
                    if p < 0:
                        continue
                    skin_mask[b, :, p] += skin_mask[b, :, i]
        for indices in pack:
            cur_N = len(indices)
            # trans to (B, num_heads, N, feat_dim)
            cur_mesh_feat = self.qmesh(mesh_feat[:, indices]).view(B, cur_N, self.num_heads, self.feat_dim).transpose(1, 2)

            # attn_weight shape : (B, num_heads, N, J)
            attn_weight = F.softmax(torch.bmm(
                cur_mesh_feat.reshape(B * self.num_heads, cur_N, -1), 
                bone_feat.transpose(-2, -1).reshape(B * self.num_heads, -1, J)
            ) / math.sqrt(self.feat_dim), dim=-1, dtype=dtype)
            # (B, num_heads, N, J) -> (B, N, J, num_heads)
            attn_weight = attn_weight.reshape(B, self.num_heads, cur_N, J).permute(0, 2, 3, 1)
            attn_weight = self.attn_skin_norm(attn_weight)
            
            embed_voxel_skin = self.voxel_skin_embed(voxel_skin[:, indices].reshape(B, cur_N, J, 1))
            embed_voxel_skin = self.voxel_skin_norm(embed_voxel_skin)
            
            attn_weight = torch.cat([attn_weight, embed_voxel_skin], dim=-1)
            attn_weight = self.downscale(attn_weight)
        
            # (B, N, J, num_heads * (1+c)) -> (B, N, J)
            skin_pred = torch.zeros(B, cur_N, J).to(attn_weight.device, dtype)
            for i in range(B):
                # (N*J, C)
                input_features = attn_weight[i, :, :num_bones[i], :].reshape(-1, attn_weight.shape[-1])
                
                if self.training:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        pred = self.skinweight_pred(input_features).reshape(cur_N, num_bones[i])
                        skin_pred[i, :, :num_bones[i]] = F.softmax(pred)
                else:
                    pred = self.skinweight_pred(input_features).reshape(cur_N, num_bones[i])
                    skin_pred[i, :, :num_bones[i]] = F.softmax(pred)
            skin_pred_list.append(skin_pred)
        skin_pred_list = torch.cat(skin_pred_list, dim=1)
        if not self.training:
            for i in range(B):
                n = num_bones[i]
                skin_pred_list[i, :, :n] = skin_pred_list[i, :, :n] * torch.pow(skin_mask[i, :, :n], self.voxel_mask)
                skin_pred_list[i, :, :n] = skin_pred_list[i, :, :n] / skin_pred_list[i, :, :n].sum(dim=-1, keepdim=True)
        return skin_pred_list, torch.cat(pack, dim=0)
    
    def training_step(self, batch: Dict) -> Dict[str, FloatTensor]:
        
        num_bones: Tensor = batch['num_bones']
        vertices: FloatTensor = batch['vertices'] # (B, N, 3)
        skin_gt: FloatTensor = batch['skin']
        
        # turn inputs' dtype into model's dtype
        dtype = next(self.parameters()).dtype
        vertices = vertices.type(dtype)
        skin_gt = skin_gt.type(dtype)
        
        B = vertices.shape[0]
        N = vertices.shape[1]
        
        matrix_local = batch.get('matrix_local')
        if matrix_local is not None:
            matrix_local: FloatTensor
            matrix_local = matrix_local.type(dtype)
            
        pose_matrix = batch.get('pose_matrix')
        if pose_matrix is not None:
            pose_matrix: FloatTensor
            pose_matrix = pose_matrix.type(dtype)
        
        skin_pred, indices = self._get_predict(batch=batch)
        vertices = vertices[:, indices]
        skin_gt = skin_gt[:, indices]
        res = {}
        
        if pose_matrix is not None:
            vertices_gt = linear_blend_skinning(
                vertex=vertices,
                matrix_local=matrix_local,
                matrix=pose_matrix,
                skin=skin_gt,
                pad=1,
                value=1.0,
            )
            vertices_pred = linear_blend_skinning(
                vertex=vertices,
                matrix_local=matrix_local,
                matrix=pose_matrix,
                skin=skin_pred,
                pad=1,
                value=1.0,
            )
            res['vertices_gt'] = vertices_gt
            res['vertices_pred'] = vertices_pred
            res['vertex_loss'] = F.mse_loss(vertices_gt, vertices_pred)
            
            eps = 1e-6
            normalization_loss = 0.
            for i in range(B):
                J = num_bones[i].item()
                for j in range(J):
                    mask = torch.nonzero(skin_gt[i, :, j] > eps).squeeze(-1)
                    if mask.size(0) == 0:
                        continue
                    _l = F.mse_loss(vertices_gt[i, mask], vertices_pred[i, mask])
                    normalization_loss += _l / J
            normalization_loss /= B
            res['normalization_loss'] = normalization_loss
        
        skin_l1_loss = 0.
        skin_l2_loss = 0.
        skin_smooth_l1_loss = 0.
        skin_zero_l1_loss = 0.
        skin_non_zero_l1_loss = 0.
        bce_loss = 0.
        iou_100 = 0.
        iou_010 = 0.
        iou_001 = 0.
        mask_accu_001 = 0.
        mask_accu_005 = 0.
        mask_accu_010 = 0.
        mask_accu_050 = 0.
        for i in range(B):
            J = num_bones[i].item()
            c_skin_pred = skin_pred[i, :, :J]
            c_skin_gt = skin_gt[i, :, :J]
            
            skin_l1_loss += torch.nn.functional.l1_loss(c_skin_pred, c_skin_gt, reduction='mean')
            skin_l2_loss += torch.nn.functional.mse_loss(c_skin_pred, c_skin_gt, reduction='mean')
            skin_smooth_l1_loss += F.smooth_l1_loss(c_skin_pred, c_skin_gt, reduction='mean', beta=1.0)
            bce_loss += (-c_skin_gt * torch.log(c_skin_pred + eps) - (1 - c_skin_gt) * torch.log(1 - c_skin_pred + eps)).mean()
            
            def get_iou(threshold):
                cap = ((c_skin_gt > threshold) & (c_skin_pred > threshold)).float()
                cup = ((c_skin_gt > threshold) | (c_skin_pred > threshold)).float()
                return ((cap.sum(dim=0) + 0.001) / (cup.sum(dim=0) + 0.001)).mean()
            iou_100 += get_iou(0.100)
            iou_010 += get_iou(0.010)
            iou_001 += get_iou(0.001)
            
            mask_accu_001 += ((c_skin_gt - c_skin_pred).abs() < 0.001).float().mean()
            mask_accu_005 += ((c_skin_gt - c_skin_pred).abs() < 0.005).float().mean()
            mask_accu_010 += ((c_skin_gt - c_skin_pred).abs() < 0.010).float().mean()
            mask_accu_050 += ((c_skin_gt - c_skin_pred).abs() < 0.050).float().mean()
            for j in range(J):
                mask = skin_gt[i, :, j] < 1e-6
                if (~mask).any():
                    skin_non_zero_l1_loss += torch.nn.functional.l1_loss(skin_pred[i, ~mask, j], skin_gt[i, ~mask, j], reduction='mean') / J
                if mask.any():
                    skin_zero_l1_loss += torch.nn.functional.l1_loss(skin_pred[i, mask, j], skin_gt[i, mask, j], reduction='mean') / J
        skin_l1_loss /= B
        skin_l2_loss /= B
        skin_smooth_l1_loss /= B
        skin_zero_l1_loss /= B
        skin_non_zero_l1_loss /= B
        bce_loss /= B
        iou_100 /= B
        iou_010 /= B
        iou_001 /= B
        mask_accu_001 /= B
        mask_accu_005 /= B
        mask_accu_010 /= B
        mask_accu_050 /= B
        
        if self.current_epoch > 20 and mask_accu_005 < 0.01:
            f = open(f"{torch.distributed.get_rank()}.txt", 'a')
            if isinstance(mask_accu_005, Tensor):
                f.write(f"{batch['path']} {mask_accu_005.item()}\n")
            else:
                f.write(f"{batch['path']} {mask_accu_005}\n")
        
        res['skin_l1_loss'] = skin_l1_loss
        res['skin_l2_loss'] = skin_l2_loss
        res['skin_smooth_l1_loss'] = skin_smooth_l1_loss
        res['skin_zero_l1_loss'] = skin_zero_l1_loss
        res['skin_non_zero_l1_loss'] = skin_non_zero_l1_loss
        res['bce_loss'] = bce_loss
        res['iou_100'] = iou_100
        res['iou_010'] = iou_010
        res['iou_001'] = iou_001
        res['mask_accu_001'] = mask_accu_001
        res['mask_accu_005'] = mask_accu_005
        res['mask_accu_010'] = mask_accu_010
        res['mask_accu_050'] = mask_accu_050
        
        return res
    
    def forward(self, data: Dict) -> Dict:
        return self.training_step(data=data)
    
    def predict_step(self, batch: Dict):
        with torch.no_grad():
            num_bones: Tensor = batch['num_bones']
            
            skin_pred, _ = self._get_predict(batch=batch)
            outputs = []
            for i in range(skin_pred.shape[0]):
                outputs.append(skin_pred[i, :, :num_bones[i]])
            return outputs
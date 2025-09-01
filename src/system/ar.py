from collections import defaultdict
import lightning as L
import os
import torch
import numpy as np
from torch import Tensor
from typing import Dict, Union, List
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

from numpy import ndarray

from .metrics import J2J, J2B, B2B

from .optimizer import get_optimizer
from .scheduler import get_scheduler

from ..data.raw_data import RawData
from ..data.order import OrderConfig, get_order
from ..model.spec import ModelSpec
from ..tokenizer.spec import DetokenizeOutput

class ARSystem(L.LightningModule):
    
    def __init__(
        self,
        steps_per_epoch: int,
        model: ModelSpec,
        optimizer_config,
        loss_config: Union[Dict[str, float], None],
        generate_kwargs: Dict={},
        output_path: Union[str, None]=None,
        record_res: Union[bool]=False,
        validate_cast: str='bfloat16',
        val_interval: Union[int, None]=None,
        val_start_from: Union[int, None]=None,
        scheduler_config=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.steps_per_epoch    = steps_per_epoch
        self.model              = model
        self.optimizer_config   = optimizer_config
        self.loss_config        = loss_config
        self.generate_kwargs    = generate_kwargs
        self.output_path        = output_path
        self.record_res         = record_res
        self.validate_cast      = validate_cast
        self.val_interval       = val_interval
        self.val_start_from     = val_start_from
        self.scheduler_config   = scheduler_config
        
        if self.record_res:
            assert self.output_path is not None, "record_res is True, but output_path in ar is None"
        
        if loss_config is not None:
            assert 'loss_sum' not in loss_config, 'loss cannot be named `loss_sum`'
            assert 'val_loss_sum' not in loss_config, 'loss cannot be named `val_loss_sum`'
    
    def forward(self, batch, validate: bool=False) -> Dict[str, Tensor]:
        loss_dict = self.model.training_step(batch)
        loss_sum = 0.
        cls = batch['cls'][0] # guaranteed to be the same cls in dataloader
        if validate:
            for name in self.loss_config:
                assert name in loss_dict, f'unspecified loss {name}'
                self._validation_loss[f"val_{cls}_{name}"].append(loss_dict[name].item())
                loss_sum += self.loss_config[name] * loss_dict[name]
            self._validation_loss[f"val_{cls}_loss_sum"].append(loss_sum.item())
            self.log('val_loss_sum', loss_sum, prog_bar=True, logger=True) # for monitor
        else:
            log_loss_dict = {}
            for name in self.loss_config:
                assert name in loss_dict, f"unspecified loss name: `{name}`"
                loss_sum += self.loss_config[name] * loss_dict[name]
                log_loss_dict[name] = loss_dict[name]
            log_loss_dict['loss_sum'] = loss_sum
            self.log_dict(log_loss_dict, prog_bar=True, logger=True)
        self.log('batch_size', len(batch['cls']))
        return loss_sum
    
    def training_step(self, batch, batch_idx, dataloader_idx=None) -> Tensor:
        assert self.loss_config is not None

        # record learning rate
        if hasattr(self.trainer, 'optimizers') and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            if 'lr' in optimizer.param_groups[0]:
                current_lr = optimizer.param_groups[0]['lr']
                self.log('lr', current_lr, prog_bar=True, logger=True)
        
        return self.forward(batch)
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None) -> Tensor:
        assert self.loss_config is not None
        return self.forward(batch, validate=True)
    
    def on_validation_epoch_start(self):
        self._validation_loss = defaultdict(list)
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.record_res:
            os.makedirs(self.output_path, exist_ok=True)
    
    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if self.val_start_from is not None and self.val_interval is not None:
            if self.current_epoch % self.val_interval != 0 or self.current_epoch < self.val_start_from:
                return
        cls = batch['cls'][0] # guaranteed to be the same cls in dataloader
        B = batch['joints'].shape[0]
        origin_vertices = batch['origin_vertices']
        origin_vertex_normals = batch['origin_vertex_normals']
        origin_faces = batch['origin_faces']
        origin_face_normals = batch['origin_face_normals']
        num_points = batch['num_points']
        num_faces = batch['num_faces']
        
        if isinstance(origin_vertices, torch.Tensor):
            origin_vertices = origin_vertices.detach().cpu().numpy()
        if isinstance(origin_vertex_normals, torch.Tensor):
            origin_vertex_normals = origin_vertex_normals.detach().cpu().numpy()
        if isinstance(origin_faces, torch.Tensor):
            origin_faces = origin_faces.detach().cpu().numpy()
        if isinstance(origin_face_normals, torch.Tensor):
            origin_face_normals = origin_face_normals.detach().cpu().numpy()
        if isinstance(num_points, torch.Tensor):
            num_points = num_points.detach().cpu().numpy()
        if isinstance(num_faces, torch.Tensor):
            num_faces = num_faces.detach().cpu().numpy()
        try:
            if self.validate_cast == 'bfloat16':
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction: List[DetokenizeOutput] = self._predict_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
            else:
                prediction: List[DetokenizeOutput] = self._predict_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        except Exception as e:
            print(str(e))
            self._validation_loss[f"val_{cls}_fail"].append(B)
            return
        for (id, res) in enumerate(prediction):
            num_bones = batch['num_bones'][id]
            joints_b: Tensor = batch['joints'][id, :num_bones]
            parents: Tensor = batch['parents'][id, :num_bones]
            bones_b = []
            for (i, pid) in enumerate(parents):
                if i == 0:
                    continue
                bones_b.append(torch.cat([joints_b[pid], joints_b[i]]))
            if len(bones_b) == 0:
                continue
            bones_b = torch.stack(bones_b)
            device = bones_b.device
            joints_a = torch.from_numpy(res.joints).to(device)
            bones_a = torch.from_numpy(res.bones[1:]).to(device) # start from 1 to remove (root, root)
            continuous_range = res.continuous_range
            if len(bones_b) == 0 or len(bones_a) ==0:
                continue
            j2j = J2J(joints_a=joints_a, joints_b=joints_b, continuous_range=continuous_range)
            j2b = J2B(joints_a=joints_a, joints_b=joints_b, bones_a=bones_a, bones_b=bones_b, continuous_range=continuous_range)
            b2b = B2B(bones_a=bones_a, bones_b=bones_b, continuous_range=continuous_range)
            self._validation_loss[f"val_{cls}_j2j"].append(j2j.item())
            self._validation_loss[f"val_{cls}_j2b"].append(j2b.item())
            self._validation_loss[f"val_{cls}_b2b"].append(b2b.item())
            self._validation_loss[f"val_{cls}_fail"].append(0.)
            if self.record_res:
                path = os.path.join(self.output_path, str(self.current_epoch), batch['path'][id])
                os.makedirs(self.output_path, exist_ok=True)
                res.export_skeleton(path+"_skeleton.obj")
                joints = batch['joints'][id]
                parents = batch['parents'][id]
                num_bones = batch['num_bones'][id]
                res._export_skeleton(joints=joints, parents=parents[:num_bones], path=path+"_skeleton_ref.obj")
                
                num_p = num_points[id]
                num_f = num_faces[id]
                
                raw_data = RawData(
                    vertices=origin_vertices[id, :num_p],
                    vertex_normals=origin_vertex_normals[id, :num_p],
                    faces=origin_faces[id, :num_f],
                    face_normals=origin_face_normals[id, :num_f],
                    joints=res.joints,
                    tails=res.tails,
                    parents=res.parents,
                    skin=None,
                    no_skin=None,
                    names=res.names,
                    matrix_local=None,
                    path=None,
                    cls=None,
                )
                raw_data.export_fbx(path=path+".fbx")
    
    def on_validation_epoch_end(self):
        # calculate per class validation loss
        val_loss_j2j = 0.
        tot_j2j = 0
        for (cls, d) in self._validation_loss.items():
            sum = 0.
            for x in d:
                sum += x
            # j2j
            if 'j2j' in cls:
                val_loss_j2j += sum
                tot_j2j += len(d)
            if len(d) != 0:
                sum /= len(d)
            else:
                sum = -1.
            self._validation_loss[cls] = sum
        if tot_j2j == 0:
            self._validation_loss['val_loss_j2j'] = 1.
        else:
            self._validation_loss['val_loss_j2j'] = val_loss_j2j / tot_j2j
        d = dict(sorted(self._validation_loss.items())) # sort to prevent wandb bugs
        self.log_dict(d, prog_bar=False, logger=True, sync_dist=True)
    
    def _predict_step(self, batch, batch_idx, dataloader_idx=None):
        batch['generate_kwargs'] = self.generate_kwargs
        res = self.model.predict_step(batch)
        assert isinstance(res, list), f"expect type of prediction from {self.model.__class__} to be a list, found: {type(res)}"
        return res
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        try:
            prediction: List[DetokenizeOutput] = self._predict_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
            return prediction
        except Exception as e:
            print(str(e))
            return []
    
    def configure_optimizers(self) -> Dict:
        _d = {}
        optimizer = get_optimizer(model=self.model, config=self.optimizer_config)
        if self.scheduler_config is not None and 'steps_per_epoch' not in self.scheduler_config:
            self.scheduler_config['steps_per_epoch'] = self.steps_per_epoch
        if self.scheduler_config is not None:
            _d['lr_scheduler'] = {
                'scheduler': get_scheduler(optimizer=optimizer, config=self.scheduler_config),
                'interval': 'step',
                'frequency': 1,
            }
        _d['optimizer'] = optimizer
        
        return _d
    
class ARWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Union[str, None],
        order_config: Union[OrderConfig, None]=None,
        **kwargs
    ):
        super().__init__('batch')
        self.output_dir         = output_dir
        self.npz_dir            = kwargs.get('npz_dir', None)
        self.user_mode          = kwargs.get('user_mode', False)
        self.output_name        = kwargs.get('output_name', None) # for a single name
        self.repeat             = kwargs.get('repeat', 1)
        self.add_num            = kwargs.get('add_num', False)
        self.export_npz         = kwargs.get('export_npz', None)
        self.export_obj         = kwargs.get('export_obj', None)
        self.export_fbx         = kwargs.get('export_fbx', None)
        self.export_pc          = kwargs.get('export_pc', None)
        if order_config is not None:
            self.order = get_order(config=order_config)
        else:
            self.order = None
        
        self._epoch = 0
        
    def on_predict_end(self, trainer, pl_module):
        if self._epoch < self.repeat - 1:
            print(f"Finished prediction run {self._epoch + 1}/{self.repeat}, starting next run...")
            self._epoch += 1
            trainer.predict_dataloader = trainer.datamodule.predict_dataloader()
            trainer.predict_loop.run()

    def write_on_batch_end(self, trainer, pl_module: ARSystem, prediction: List[Dict], batch_indices, batch, batch_idx, dataloader_idx):
        assert 'path' in batch
        paths = batch['path']
        detokenize_output_list: List[DetokenizeOutput] = prediction
        vertices = batch['vertices']
        
        origin_vertices = batch['origin_vertices']
        origin_vertex_normals = batch['origin_vertex_normals']
        origin_faces = batch['origin_faces']
        origin_face_normals = batch['origin_face_normals']
        num_points = batch['num_points']
        num_faces = batch['num_faces']
        
        if isinstance(origin_vertices, torch.Tensor):
            origin_vertices = origin_vertices.detach().cpu().numpy()
        if isinstance(origin_vertex_normals, torch.Tensor):
            origin_vertex_normals = origin_vertex_normals.detach().cpu().numpy()
        if isinstance(origin_faces, torch.Tensor):
            origin_faces = origin_faces.detach().cpu().numpy()
        if isinstance(origin_face_normals, torch.Tensor):
            origin_face_normals = origin_face_normals.detach().cpu().numpy()
        if isinstance(num_points, torch.Tensor):
            num_points = num_points.detach().cpu().numpy()
        if isinstance(num_faces, torch.Tensor):
            num_faces = num_faces.detach().cpu().numpy()

        for (id, detokenize_output) in enumerate(detokenize_output_list):
            assert isinstance(detokenize_output, DetokenizeOutput), f"expect item of the list to be DetokenizeOutput, found: {type(detokenize_output)}"
            def make_path(save_name: str, suffix: str, trim: bool=False):
                if trim:
                    path = os.path.relpath(paths[id], self.npz_dir)
                else:
                    path = paths[id]

                if self.output_dir is not None:
                    path = os.path.join(self.output_dir, path)
                
                if self.add_num:
                    path = os.path.join(path, f"{save_name}_{self._epoch}.{suffix}")
                else:
                    path = os.path.join(path, f"{save_name}.{suffix}")
                return path
            
            num_p = num_points[id]
            num_f = num_faces[id]
            
            raw_data = RawData(
                vertices=origin_vertices[id, :num_p],
                vertex_normals=origin_vertex_normals[id, :num_p],
                faces=origin_faces[id, :num_f],
                face_normals=origin_face_normals[id, :num_f],
                joints=detokenize_output.joints,
                tails=detokenize_output.tails,
                parents=detokenize_output.parents,
                skin=None,
                no_skin=detokenize_output.no_skin,
                names=detokenize_output.names,
                matrix_local=None,
                path=None,
                cls=detokenize_output.cls,
            )
            if not self.user_mode and self.export_npz is not None:
                print(make_path(self.export_npz, 'npz'))
                raw_data.save(path=make_path(self.export_npz, 'npz'))
            if not self.user_mode and self.export_obj is not None:
                raw_data.export_skeleton(path=make_path(self.export_obj, 'obj'))
            if not self.user_mode and self.export_pc is not None:
                raw_data.export_pc(path=make_path(self.export_pc, 'obj'))
            if self.export_fbx is not None:
                if not self.user_mode:
                    raw_data.export_fbx(path=make_path(self.export_fbx, 'fbx'))
                else:
                    if self.output_name is not None:
                        raw_data.export_fbx(path=self.output_name)
                    else:
                        raw_data.export_fbx(path=make_path(self.export_fbx, 'fbx', trim=True))
import torch
from torch import nn, FloatTensor, LongTensor
import numpy as np
from torch.nn.functional import pad
from typing import Dict, List, Union
from transformers import AutoModelForCausalLM, AutoConfig, LogitsProcessor, LogitsProcessorList

from .spec import ModelSpec, ModelInput
from .parse_encoder import MAP_MESH_ENCODER, get_mesh_encoder

from ..tokenizer.spec import TokenizerSpec, DetokenizeOutput
from copy import deepcopy

class VocabSwitchingLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: TokenizerSpec, start_tokens: LongTensor):
        self.tokenizer = tokenizer
        self.start_tokens = start_tokens
        assert start_tokens.ndim == 1

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        # input_ids shape: (batch_size, seq_len)
        for batch_idx, sequence in enumerate(input_ids):
            mask = torch.full_like(scores[batch_idx], float('-inf'))
            sequence = torch.cat([self.start_tokens, sequence])
            tokens = self.tokenizer.next_posible_token(ids=sequence.detach().cpu().numpy())
            mask[tokens] = 0
            scores[batch_idx] = scores[batch_idx] + mask
        return scores

class UniRigAR(ModelSpec):
    
    def process_fn(self, batch: List[ModelInput]) -> List[Dict]:
        if batch[0].joints is None: # predict
            return [{} for _ in range(len(batch))]
        max_length = 0
        for b in batch:
            max_length = max(max_length, b.tokens.shape[0])
        res = [{
            'input_ids': np.pad(b.tokens, ((0, max_length-b.tokens.shape[0])), 'constant', constant_values=b.pad),
            'attention_mask': np.pad(torch.ones(b.tokens.shape[0]), ((0, max_length - b.tokens.shape[0])), 'constant', constant_values=0.),
        } for b in batch]
        return res
    
    def __init__(self, llm, mesh_encoder, **kwargs):
        super().__init__()
        self.tokenizer: TokenizerSpec = kwargs.get('tokenizer')
        self.vocab_size = self.tokenizer.vocab_size
        
        _d = llm.copy()
        _d['vocab_size'] = self.tokenizer.vocab_size
        llm_config = AutoConfig.from_pretrained(**_d)
        # Force float32 precision for the model
        llm_config.torch_dtype = torch.float32
        # Force enable pre_norm
        llm_config.pre_norm = True
        self.transformer = AutoModelForCausalLM.from_config(config=llm_config)
        
        self.hidden_size = llm.hidden_size
        
        self.mesh_encoder = get_mesh_encoder(**mesh_encoder)
        
        if (
            isinstance(self.mesh_encoder, MAP_MESH_ENCODER.michelangelo) or
            isinstance(self.mesh_encoder, MAP_MESH_ENCODER.michelangelo_encoder)
        ):
            self.output_proj = nn.Linear(self.mesh_encoder.width, self.hidden_size)
        else:
            raise NotImplementedError()
            
    def encode_mesh_cond(self, vertices: FloatTensor, normals: FloatTensor) -> FloatTensor:
        assert not torch.isnan(vertices).any()
        assert not torch.isnan(normals).any()
        if (
            isinstance(self.mesh_encoder, MAP_MESH_ENCODER.michelangelo) or
            isinstance(self.mesh_encoder, MAP_MESH_ENCODER.michelangelo_encoder)
        ):
            if (len(vertices.shape) == 3):
                shape_embed, latents, token_num, pre_pc = self.mesh_encoder.encode_latents(pc=vertices, feats=normals)
            else:
                shape_embed, latents, token_num, pre_pc = self.mesh_encoder.encode_latents(pc=vertices.unsqueeze(0), feats=normals.unsqueeze(0))
            latents = self.output_proj(latents)
            return latents
        else:
            raise NotImplementedError()
    
    def training_step(self, batch: Dict) -> Dict[str, FloatTensor]:
        cond = self.encode_mesh_cond(vertices=batch['vertices'], normals=batch['normals']).to(dtype=self.transformer.dtype)
        B = cond.shape[0]
        input_ids: LongTensor = batch['input_ids']
        inputs_embeds = self.transformer.get_input_embeddings()(input_ids).to(dtype=self.transformer.dtype)
        
        inputs_embeds = torch.concat([cond, inputs_embeds], dim=1)
        
        attention_mask = batch['attention_mask']
        # add attention to condition
        attention_mask = pad(attention_mask, (cond.shape[1], 0, 0, 0), value=1.)
        output = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        
        # (B, L, vocab_size)
        logit = output.logits[:, cond.shape[1]:].reshape(B, -1, self.vocab_size)
        # compute loss with shift one-token right
        device = logit.device # (B, n, num_discrete)
        logit = logit[:, :-1] # (B, n)
        num_discrete = self.tokenizer.num_discrete
        s = torch.nn.functional.softmax(logit, dim=-1)
        
        label = input_ids[:, 1:].clone() # (B, n)
        mask = label < num_discrete
        dis = torch.arange(num_discrete, device=device).view(1, 1, -1) # (B, n, num_discrete)
        dis = (dis - label.unsqueeze(2).repeat(1, 1, num_discrete)).type(torch.float32) / num_discrete
        dis_loss = (s[:, :, :num_discrete] * torch.abs(dis))[mask].sum() / 50 # ignore padding loss
        
        label[attention_mask[:, cond.shape[1] + 1:]==0] = -100
        
        assert not torch.isnan(logit).any(), logit
        ce_loss = nn.functional.cross_entropy(logit.permute(0, 2, 1), label)
        return {
            'ce_loss': ce_loss,
            'dis_loss': dis_loss,
        }
    
    def forward(self, data: Dict):
        return self.training_step(data=data)
    
    @torch.no_grad()
    def generate(
        self,
        vertices: FloatTensor,
        normals: FloatTensor,
        cls: Union[str, None]=None,
        **kwargs,
    ) -> DetokenizeOutput:
        '''
        Do not support batch!
        '''
        cond = self.encode_mesh_cond(vertices=vertices, normals=normals).to(dtype=self.transformer.dtype)
        
        start_tokens = [self.tokenizer.bos]
        
        if cls is not None:
            start_tokens.append(self.tokenizer.cls_name_to_token(cls=cls))
        start_tokens = torch.tensor(start_tokens).to(cond.device)
        start_embed = self.transformer.get_input_embeddings()(
            start_tokens.unsqueeze(0)
        ).to(dtype=self.transformer.dtype)
        cond = torch.cat([cond, start_embed], dim=1)
        
        processor = VocabSwitchingLogitsProcessor(
            tokenizer=self.tokenizer,
            start_tokens=start_tokens,
        )
        results = self.transformer.generate(
            inputs_embeds=cond,
            bos_token_id=self.tokenizer.bos,
            eos_token_id=self.tokenizer.eos,
            pad_token_id=self.tokenizer.pad,
            logits_processor=LogitsProcessorList([processor]),
            **kwargs,
        )
        output_ids = results[0, :]
        for token in reversed(start_tokens):
            output_ids = pad(output_ids, (1, 0), value=token)
        output_ids = output_ids.detach().cpu().numpy()
        
        res = self.tokenizer.detokenize(ids=output_ids)
        return res
    
    def predict_step(self, batch: Dict, no_cls: bool=False):
        vertices: FloatTensor   = batch['vertices']
        normals : FloatTensor   = batch['normals']
        paths   : List[str]     = batch['path']
        cls = batch['cls']
        generate_kwargs = deepcopy(batch['generate_kwargs'])

        no_cls = generate_kwargs.get('no_cls', False)
        use_dir_cls = generate_kwargs.get('use_dir_cls', False)
        assign_cls = generate_kwargs.get('assign_cls', None)

        generate_kwargs.pop('no_cls', None)
        generate_kwargs.pop('use_dir_cls', None)
        generate_kwargs.pop('assign_cls', None)

        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)
            normals  = normals.unsqueeze(0)
        outputs = []
        for i in range(vertices.shape[0]):
            if no_cls:
                _cls = None
            elif assign_cls is not None:
                _cls = assign_cls
            elif use_dir_cls:
                _cls = paths[i].removeprefix('./').split('/')[0]
            else:
                _cls = cls[i]
            res = self.generate(vertices=vertices[i], normals=normals[i], cls=_cls, **generate_kwargs)
            outputs.append(res)
        return outputs
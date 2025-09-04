import torch
from torch.optim import Optimizer

from ..model.spec import ModelSpec

def get_optimizer(model: ModelSpec, config) -> Optimizer:
    MAP = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
    }
    __target__ = config.__target__
    del config.__target__
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](params=model.parameters(), **config)
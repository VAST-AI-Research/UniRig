import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer

from ..model.spec import ModelSpec

def get_scheduler(optimizer: Optimizer, config) -> LRScheduler:
    MAP = {
        'one_cycle_lr': torch.optim.lr_scheduler.OneCycleLR,
    }
    
    __target__ = config.__target__
    del config.__target__
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](optimizer=optimizer, **config)
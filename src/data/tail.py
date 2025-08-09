from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from numpy import ndarray

from typing import Tuple

from .asset import Asset
from .spec import ConfigSpec

@dataclass
class TailConfig(ConfigSpec):
    '''
    Config to handle tails.
    '''
    
    # copy joints to tails
    copy_joint_to_tail: bool
    
    # if the joint has only one son, then connect tail to son's joint
    connect_tail_to_unique_son: bool
    
    @classmethod
    def parse(cls, config) -> 'TailConfig':
        cls.check_keys(config)
        return TailConfig(
            copy_joint_to_tail=config.copy_joint_to_tail,
            connect_tail_to_unique_son=config.connect_tail_to_unique_son,
        )

class Tail():
    
    def __init__(self, config: TailConfig):
        self.config = config
    
    def process_tail(self, asset: Asset):
        if self.config.copy_joint_to_tail:
            asset.tails = asset.joints.copy()
        if self.config.connect_tail_to_unique_son and asset.tails is not None:
            children = defaultdict(list)
            for (id, p) in enumerate(asset.parents):
                if p is not None:
                    children[p].append(id)
            for i in range(asset.J):
                if len(children[i]) == 1:
                    asset.tails[i] = asset.joints[children[i][0]]

def get_tail(config: TailConfig) -> Tail:
    return Tail(config=config)
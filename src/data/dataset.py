from copy import deepcopy
from dataclasses import dataclass
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import torch
from torch import LongTensor
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Union, Callable
import os
import numpy as np

from .raw_data import RawData
from .asset import Asset
from .transform import TransformConfig, transform_asset
from .datapath import DatapathConfig, Datapath
from .spec import ConfigSpec

from ..tokenizer.spec import TokenizerSpec, TokenizerConfig
from ..tokenizer.parse import get_tokenizer
from ..model.spec import ModelInput

@dataclass
class DatasetConfig(ConfigSpec):
    '''
    Config to handle dataset format.
    '''
    # shuffle dataset
    shuffle: bool

    # batch size
    batch_size: int

    # number of workers
    num_workers: int
    
    # datapath
    datapath_config: DatapathConfig
    
    # use pin memory
    pin_memory: bool = True
    
    # use persistent workers
    persistent_workers: bool = True
    
    @classmethod
    def parse(cls, config) -> 'DatapathConfig':
        cls.check_keys(config)
        return DatasetConfig(
            shuffle=config.shuffle,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            datapath_config=DatapathConfig.parse(config.datapath_config),
        )
    
    def split_by_cls(self) -> Dict[str, 'DatasetConfig']:
        res: Dict[str, DatasetConfig] = {}
        datapath_config_dict = self.datapath_config.split_by_cls()
        for cls in self.datapath_config.data_path:
            res[cls] = deepcopy(self)
            res[cls].datapath_config = datapath_config_dict[cls]
        return res

class UniRigDatasetModule(pl.LightningDataModule):  
    def __init__(
        self,
        process_fn: Union[Callable[[List[ModelInput]], Dict]]=None,
        train_dataset_config: Union[DatasetConfig, None]=None,
        predict_dataset_config: Union[Dict[str, DatasetConfig], None]=None,
        validate_dataset_config: Union[Dict[str, DatasetConfig], None]=None,
        train_transform_config: Union[TransformConfig, None]=None,
        predict_transform_config: Union[TransformConfig, None]=None,
        validate_transform_config: Union[TransformConfig, None]=None,
        tokenizer_config: Union[TokenizerConfig, None]=None,
        debug: bool=False,
        data_name: str='raw_data.npz',
        datapath: Union[Datapath, None]=None,
        cls: Union[str, None]=None,
    ):
        super().__init__()
        self.process_fn                 = process_fn
        self.train_dataset_config       = train_dataset_config
        self.predict_dataset_config     = predict_dataset_config
        self.validate_dataset_config    = validate_dataset_config
        self.train_transform_config     = train_transform_config
        self.predict_transform_config   = predict_transform_config
        self.validate_transform_config  = validate_transform_config
        self.tokenizer_config           = tokenizer_config
        self.debug                      = debug
        self.data_name                  = data_name
        
        if debug:
            print("\033[31mWARNING: debug mode, dataloader will be extremely slow !!!\033[0m")
        
        # build train datapath
        if self.train_dataset_config is not None:
            self.train_datapath = Datapath(self.train_dataset_config.datapath_config)
        else:
            self.train_datapath = None
        
        # build validate datapath
        if self.validate_dataset_config is not None:
            self.validate_datapath = {
                cls: Datapath(self.validate_dataset_config[cls].datapath_config)
                for cls in self.validate_dataset_config
            }
        else:
            self.validate_datapath = None
        
        if datapath is not None:
            self.train_datapath = None
            self.validate_datapath = None
            self.predict_datapath = {
                cls: deepcopy(datapath),
            }
            self.predict_dataset_config = {
                cls: DatasetConfig(
                    shuffle=False,
                    batch_size=1,
                    num_workers=0,
                    datapath_config=deepcopy(datapath),
                    pin_memory=False,
                    persistent_workers=False,
                )
            }
        else:
            # build predict datapath
            if self.predict_dataset_config is not None:
                self.predict_datapath = {
                    cls: Datapath(self.predict_dataset_config[cls].datapath_config)
                    for cls in self.predict_dataset_config
                }
            else:
                self.predict_datapath = None
        
        # get tokenizer
        if tokenizer_config is None:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(config=tokenizer_config)

    def prepare_data(self):
        pass

    def setup(self, stage=None):   
        if self.predict_datapath is not None:
            self._predict_ds = {}
            for cls in self.predict_datapath:
                self._predict_ds[cls] = UniRigDataset(
                    process_fn=self.process_fn,
                    data=self.predict_datapath[cls].get_data(),
                    name=f"predict-{cls}",
                    tokenizer=self.tokenizer,
                    transform_config=self.predict_transform_config,
                    debug=self.debug,
                    data_name=self.data_name,
                )
        
        if self.validate_datapath is not None:
            self._validation_ds = {}
            for cls in self.validate_datapath:
                self._validation_ds[cls] = UniRigDataset(
                    process_fn=self.process_fn,
                    data=self.validate_datapath[cls].get_data(),
                    name=f"validate-{cls}",
                    tokenizer=self.tokenizer,
                    transform_config=self.validate_transform_config,
                    debug=self.debug,
                    data_name=self.data_name,
                )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # rebuild every time
        assert self.train_datapath is not None, "do not have training data"
        self._train_ds = UniRigDataset(
            process_fn=self.process_fn,
            data=self.train_datapath.get_data(),
            name="train",
            tokenizer=self.tokenizer,
            transform_config=self.train_transform_config,
            debug=self.debug,
            data_name=self.data_name,
        )
        return self._create_dataloader(
            dataset=self._train_ds,
            config=self.train_dataset_config,
            is_train=True,
            drop_last=False,
        )
    
    def predict_dataloader(self):
        if not hasattr(self, "_predict_ds"):
            self.setup()
        return self._create_dataloader(
            dataset=self._predict_ds,
            config=self.predict_dataset_config,
            is_train=False,
            drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(
            dataset=self._validation_ds,
            config=self.validate_dataset_config,
            is_train=False,
            drop_last=False,
        )

    def _create_dataloader(
        self,
        dataset: Union[Dataset, Dict[str, Dataset]],
        config: DatasetConfig,
        is_train: bool,
        **kwargs,
    ) -> Union[DataLoader, Dict[str, DataLoader]]:
        def create_single_dataloader(dataset, config: Union[DatasetConfig, Dict[str, DatasetConfig]], **kwargs):
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=config.shuffle,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=config.persistent_workers,
                collate_fn=dataset.collate_fn,
                **kwargs,
            )
        if isinstance(dataset, Dict):
            return {k: create_single_dataloader(v, config[k], **kwargs) for k, v in dataset.items()}
        else:
            return create_single_dataloader(dataset, config, **kwargs)

class UniRigDataset(Dataset):
    def __init__(
        self,
        data: List[Tuple[str, str]], # (cls, part)
        name: str,
        process_fn: Union[Callable[[List[ModelInput]], Dict]]=None,
        tokenizer: Union[TokenizerSpec, None]=None,
        transform_config: Union[TransformConfig, None]=None,
        debug: bool=False,
        data_name: str='raw_data.npz',
    ) -> None:
        super().__init__()
        
        self.data               = data
        self.name               = name
        self.process_fn         = process_fn
        self.tokenizer          = tokenizer
        self.transform_config   = transform_config
        self.debug              = debug
        self.data_name          = data_name
        
        if not debug:
            assert self.process_fn is not None, 'missing data processing function'

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> ModelInput:
        cls, dir_path = self.data[idx]
        raw_data = RawData.load(path=os.path.join(dir_path, self.data_name))
        asset = Asset.from_raw_data(raw_data=raw_data, cls=cls, path=dir_path, data_name=self.data_name)
        
        first_augments, second_augments = transform_asset(
            asset=asset,
            transform_config=self.transform_config,
        )
        if self.tokenizer is not None and asset.parents is not None:
            tokens = self.tokenizer.tokenize(input=asset.get_tokenize_input())
        else:
            tokens = None
        return ModelInput(
            tokens=tokens,
            pad=None if self.tokenizer is None else self.tokenizer.pad,
            vertices=asset.sampled_vertices.astype(np.float32),
            normals=asset.sampled_normals.astype(np.float32),
            joints=None if asset.joints is None else asset.joints.astype(np.float32),
            tails=None if asset.tails is None else asset.tails.astype(np.float32),
            asset=asset,
            augments=None,
        )

    def _collate_fn_debug(self, batch):
        return batch
    
    def _collate_fn(self, batch):
        return data.dataloader.default_collate(self.process_fn(batch))


    def collate_fn(self, batch):
        if self.debug:
            return self._collate_fn_debug(batch)
        return self._collate_fn(batch)
import argparse
import yaml
from box import Box
import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger
from typing import List
from math import ceil
import numpy as np
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy

from src.data.asset import Asset
from src.data.dataset import UniRigDatasetModule, DatasetConfig, ModelInput
from src.data.transform import TransformConfig
from src.tokenizer.spec import TokenizerConfig
from src.tokenizer.parse import get_tokenizer
from src.data.track import TrackLoaderConfig
from src.model.parse import get_model
from src.system.parse import get_system, get_writer

from tqdm import tqdm
import time

def load(task: str, path: str) -> Box:
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    print(f"\033[92mload {task} config: {path}\033[0m")
    return Box(yaml.safe_load(open(path, 'r')))

def debug_fn(data: L.LightningDataModule, **kwargs):
    tokenizer_config = kwargs.get('tokenizer_config')
    if tokenizer_config is not None:
        tokenizer = get_tokenizer(tokenizer_config)
    else:
        tokenizer = None
    for batch in data.train_dataloader():
        batch: List[ModelInput]
        # parts: List[Asset] = batch[0].asset.meta['parts']
        # for i, part in enumerate(parts):
        #     part.export_pc(f"tmp_debug/{i}_pc.obj")
        #     print("?", part.sampled_vertices.max(axis=0), part.sampled_vertices.min(axis=0))
        # print("OK")
        # exit()
        for b in batch:
            path = os.path.join("tmp_debug", b.asset.path.removeprefix('.').removeprefix('/').replace('/', '-'))
            # b.asset.export_render(os.path.join(path, "render.png"))
            # b.asset.export_fbx(os.path.join(path, "res.fbx"), 'geodesic_distance', use_tail=False, use_extrude_bone=True, use_origin=True, do_not_normalize=True)
            res = tokenizer.detokenize(b.tokens)
            res.export_skeleton(os.path.join(path, 'skeleton.obj'))
            b.asset.export_skeleton(os.path.join(path, 'skeleton_gt.obj'))
            print(path)
            exit()
            diff = np.abs(b.asset.vertex_groups['skin'].sum(axis=-1)-1.0).max()

if __name__ == "__main__":
    # Set float32 matmul precision to 'high' to properly utilize Tensor Cores
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False, default=123)
    args = parser.parse_args()
    
    L.seed_everything(args.seed, workers=True)
    
    task = load('task', args.task)
    mode = task.mode
    assert mode in ['train', 'predict', 'debug', 'validate']
    
    debug = task.get('debug', False)
    if mode == 'debug':
        debug = True
    
    data_config = load('data', os.path.join('configs/data', task.components.data))
    transform_config = load('transform', os.path.join('configs/transform', task.components.transform))
    
    # get tokenizer
    tokenizer_config = task.components.get('tokenizer', None)
    if tokenizer_config is not None:
        tokenizer_config = load('tokenizer', os.path.join('configs/tokenizer', task.components.tokenizer))
        tokenizer_config = TokenizerConfig.parse(config=tokenizer_config)
    
    # get track loader
    track_loader_config = task.components.get('track_loader', None)
    if track_loader_config is not None:
        track_loader_config = load('track_loader', os.path.join('configs/track_loader', task.components.track_loader))
        track_loader_config = TrackLoaderConfig.parse(config=track_loader_config.track_loader_config)
    
    # get train dataset
    train_dataset_config = data_config.get('train_dataset_config', None)
    if train_dataset_config is not None:
        train_dataset_config = DatasetConfig.parse(config=train_dataset_config)
    
    # get validate dataset
    validate_dataset_config = data_config.get('validate_dataset_config', None)
    if validate_dataset_config is not None:
        validate_dataset_config = DatasetConfig.parse(config=validate_dataset_config).split_by_cls()
        
    # get predict dataset
    predict_dataset_config = data_config.get('predict_dataset_config', None)
    if predict_dataset_config is not None:
        predict_dataset_config = DatasetConfig.parse(config=predict_dataset_config).split_by_cls()
    
    # get train transform
    train_transform_config = config=transform_config.get('train_transform_config', None)
    if train_transform_config is not None:
        train_transform_config = TransformConfig.parse(config=train_transform_config)
    
    # get validate transform
    validate_transform_config = transform_config.get('validate_transform_config', None)
    if validate_transform_config is not None:
        validate_transform_config = TransformConfig.parse(config=validate_transform_config)
    
    # get predict transform
    predict_transform_config = transform_config.get('predict_transform_config', None)
    if predict_transform_config is not None:
        predict_transform_config = TransformConfig.parse(config=predict_transform_config)
    
    # get model
    model_config = task.components.get('model', None)
    if model_config is not None:
        model_config = load('model', os.path.join('configs/model', model_config))
        if tokenizer_config is not None:
            tokenizer = get_tokenizer(config=tokenizer_config)
        else:
            tokenizer = None
        model = get_model(tokenizer=tokenizer, **model_config)
    else:
        model = None
    
    optimizer_config = task.get('optimizer', None)
    loss_config = task.get('loss', None)
    # set data
    data = UniRigDatasetModule(
        process_fn=None if model is None else model._process_fn,
        train_dataset_config=train_dataset_config,
        validate_dataset_config=validate_dataset_config,
        predict_dataset_config=predict_dataset_config,
        train_transform_config=train_transform_config,
        validate_transform_config=validate_transform_config,
        predict_transform_config=predict_transform_config,
        tokenizer_config=tokenizer_config,
        track_loader_config=track_loader_config,
        debug=debug,
    )
    
    # add call backs
    callbacks = []

    ## get checkpoint callback
    checkpoint_config = task.get('checkpoint', None)
    if checkpoint_config is not None:
        checkpoint_config['dirpath'] = os.path.join('experiments', task.experiment_name)
        callbacks.append(ModelCheckpoint(**checkpoint_config))
    
    ## get writer callback
    writer_config = task.get('writer', None)
    if writer_config is not None:
        assert predict_transform_config is not None, 'missing predict_transform_config in transform'
        callbacks.append(get_writer(**writer_config, order_config=predict_transform_config.order_config))
    
    # get trainer
    trainer_config = task.get('trainer', {})
    
    # get scheduler
    scheduler_config = task.get('scheduler', None)
    
    # get system
    system_config = task.components.get('system', None)
    if system_config is not None:
        system_config = load('system', os.path.join('configs/system', system_config))
        system = get_system(
            **system_config,
            model=model,
            optimizer_config=optimizer_config,
            loss_config=loss_config,
            scheduler_config=scheduler_config,
            steps_per_epoch=1 if train_dataset_config is None else 
            ceil(len(data.train_dataloader()) // trainer_config.devices // trainer_config.num_nodes),
        )
    else:
        system = None
    
    # set wandb
    wandb_config = task.get('wandb', None)
    if wandb_config is not None:
        print(f"\033[92mInitializing WandbLogger with config: {wandb_config}\033[0m")
        logger = WandbLogger(
            config={
                'task': task,
                'data': data_config,
                'tokenizer': tokenizer_config,
                'track_loader': track_loader_config,
                'train_dataset_config': train_dataset_config,
                'validate_dataset_config': validate_dataset_config,
                'predict_dataset_config': predict_dataset_config,
                'train_transform_config': train_transform_config,
                'validate_transform_config': validate_transform_config,
                'predict_transform_config': predict_transform_config,
                'model_config': model_config,
                'optimizer_config': optimizer_config,
                'system_config': system_config,
                'checkpoint_config': checkpoint_config,
                'writer_config': writer_config,
            },
            log_model=True,
            **wandb_config
        )
        if logger.experiment.id is not None:
            print(f"\033[92mWandbLogger started: {logger.experiment.id}\033[0m")
            # Get the run URL using wandb.run.get_url() which is more reliable
            run_url = logger.experiment.get_url() if hasattr(logger.experiment, 'get_url') else logger.experiment.url
            print(f"\033[92mWandbLogger url: {run_url}\033[0m")
        else:
            print("\033[91mWandbLogger failed to start\033[0m")
    else:
        logger = None

    # set ckpt path
    resume_from_checkpoint = task.get('resume_from_checkpoint', None)
    strategy = FSDPStrategy(
        # Enable activation checkpointing on these layers
        auto_wrap_policy={
            torch.nn.MultiheadAttention
        },
        activation_checkpointing_policy={
            torch.nn.Linear,
            torch.nn.MultiheadAttention,
        },
    )
    trainer_config['strategy'] = strategy
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        **trainer_config,
    )
    
    if mode == 'predict':
        assert resume_from_checkpoint is not None, 'expect resume_from_checkpoint in task'
        trainer.predict(system, datamodule=data, ckpt_path=resume_from_checkpoint, return_predictions=False)
    elif mode == 'train':
        trainer.fit(system, datamodule=data, ckpt_path=resume_from_checkpoint)
    elif mode == 'validate':
        trainer.validate(system, datamodule=data, ckpt_path=resume_from_checkpoint)
    elif mode == 'debug':
        debug_fn(data=data, tokenizer_config=tokenizer_config)
    else:
        assert 0
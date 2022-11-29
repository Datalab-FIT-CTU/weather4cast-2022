# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
# 
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime
import os
import torch 
import numpy as np

from utils.data_utils import load_config
from utils.w4c_dataloader import RainData
from models.models import *

WANDB_LOGGING = False
if WANDB_LOGGING:
    import wandb

class DataModule(pl.LightningDataModule):
    """ Class to handle training/validation splits in a single object
    """
    def __init__(self, params, training_params, mode):
        super().__init__()
        self.params = params     
        self.training_params = training_params
        if mode in ['train']:
            self.train_ds = RainData('training', **self.params)
            self.val_ds = RainData('validation', **self.params)
        if mode in ['val']:
            self.val_ds = RainData('validation', **self.params)    
        if mode in ['predict']:    
            self.test_ds = RainData('test', **self.params)

    def __load_dataloader(self, dataset, shuffle=True, pin=True):
        dl = DataLoader(dataset, 
                        batch_size=self.training_params['batch_size'],
                        num_workers=self.training_params['n_workers'],
                        shuffle=shuffle, pin_memory=pin, prefetch_factor=2,
                        persistent_workers=False)
        return dl
    
    def train_dataloader(self):
        return self.__load_dataloader(self.train_ds, shuffle=True, pin=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=5, num_workers=4, shuffle=False)

    def test_dataloader(self):
        return self.__load_dataloader(self.test_ds, shuffle=False, pin=True)


def load_model(Model, params, checkpoint_path=''):
    """ loads a model from a checkpoint or from scratch if checkpoint_path='' """
    if checkpoint_path == '':
        print('-> Modelling from scratch!  (no checkpoint loaded)')
        model = Model(params)            
    else:
        print(f'-> Loading model checkpoint: {checkpoint_path}')
        model = Model.load_from_checkpoint(checkpoint_path, config=params)
    return model

def get_trainer(gpus,params):
    """ get the trainer, modify here its options:
        - save_top_k
     """
    max_epochs=params['train']['max_epochs'];
    print("Trainig for",max_epochs,"epochs");
    
    paralel_training = None
    ddppplugin = None   

    if WANDB_LOGGING:
        wandb.login(force=True, key="7f97310b61bd5aeaaf34b363b74f851109876005", relogin=True)
        wandb_logger = pl.loggers.WandbLogger(name=params['experiment']['name'], project="w4c stage 2")

    checkpoint_callback = ModelCheckpoint(
        monitor=params["train"]["monitor_metric"],
        mode=params["train"]["monitor_mode"],
        save_top_k=1,
        save_last=True,
        filename='{epoch:02d}-{' + params["train"]["monitor_metric"] + ':.6f}'
    )

    if params['train']['early_stopping']: 
        early_stop_callback = EarlyStopping(
            monitor=params["train"]["monitor_metric"],
            patience=params['train']['patience'],
            mode=params["train"]["monitor_mode"],
            strict=False
        )
        callback_funcs = [checkpoint_callback, early_stop_callback]
    else: 
        callback_funcs = [checkpoint_callback]
   
    trainer = pl.Trainer(gpus=gpus, max_epochs=max_epochs,
                         gradient_clip_val=params['model']['gradient_clip_val'],
                         gradient_clip_algorithm=params['model']['gradient_clip_algorithm'],
                         accelerator=paralel_training,
                         callbacks=callback_funcs,
                         logger=wandb_logger if WANDB_LOGGING else None,
                         profiler='simple',
                         precision=params['experiment']['precision'],
                         default_root_dir="checkpoints",
                         val_check_interval=3000,
                         plugins=ddppplugin,
                        )
    
    return trainer, checkpoint_callback



def train(config, gpus, checkpoint_path): 
    """ main training/evaluation method
    """

    data = DataModule(config['dataset'], config['train'], "train")
    model = load_model(WeatherFusionNet, config, checkpoint_path)

    trainer, checkpoint_callback = get_trainer(gpus, config)

    trainer.fit(model, data, ckpt_path=checkpoint_path)

    # restore best checkpoint
    print(checkpoint_callback.best_model_path)
    model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    results = trainer.validate(model, data.val_dataloader())[0]
    if WANDB_LOGGING:
        wandb.log(results)  # final log to display best metrics on the table
        wandb.finish()



def update_params_based_on_args(options):
    params = load_config(options.config_path)
    
    if options.name != '':
        params['experiment']['name'] = options.name
        print(params['experiment']['name'])
    return params
    
def set_parser():
    """ set custom parser """
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--config_path", type=str, required=False, default='models/configurations/config.yaml',
                        help="path to config-yaml")
    parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=1, 
                        help="specify gpu(s): 1 or 1 5 or 0 1 2 (-1 for no gpu)")
    parser.add_argument("-c", "--checkpoint", type=str, required=False, default='', 
                        help="init a model from a checkpoint path. '' as default (random weights)")
    parser.add_argument("-n", "--name", type=str, required=False, default='WeatherFusionNet', 
                         help="Set the name of the experiment")

    return parser

def main():
    parser = set_parser()
    options = parser.parse_args()

    params = update_params_based_on_args(options)
    train(params, options.gpus, options.checkpoint)

if __name__ == "__main__":
    main()
    """ examples of usage:

    1) train from scratch on one GPU
    python train.py --gpus 2 --config_path models/configurations/config.yaml --name experiment_name

    2) train from scratch on four GPUs
    python train.py --gpus 0 1 2 3 --config_path models/configurations/config.yaml --name experiment_name
    
    3) fine tune a model from a checkpoint on one GPU
    python train.py --gpus 1 --config_path models/configurations/config.yaml --checkpoint "path/to/checkpoint.ckpt" --name experiment_name

    """

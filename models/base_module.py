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


from turtle import pos
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.evaluate import *        


VERBOSE = False
#VERBOSE = True

class BaseLitModule(pl.LightningModule):
    def __init__(self, config: dict):
        super(BaseLitModule, self).__init__()

        self.save_hyperparameters()
        self.config = config

        pos_weight = torch.tensor(config['train']['pos_weight']);

        self.loss = config['train']['loss']
        self.loss_fn = {
            'smoothL1': nn.SmoothL1Loss(), 'L1': nn.L1Loss(), 'mse': F.mse_loss,
            'BCELoss': nn.BCELoss(), 
            'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(pos_weight=pos_weight),
            'CrossEntropy': nn.CrossEntropyLoss(),
            'DiceBCE': DiceBCELoss(),
            'DiceLoss': DiceLoss(),
            'FocalLoss': FocalLoss(pos_weight=pos_weight),
        }[self.loss]

        self.relu = nn.ReLU() # None
    
    def on_fit_start(self):
        """ create a placeholder to save the results of the metric per variable """
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
        
    def forward(self, x):
        x = self.model(x)

        # if self.loss =='BCELoss':
        #     x = self.relu(x)
        return x

    def retrieve_only_valid_pixels(self, x, m):
        """ we asume 1s in mask are invalid pixels """
        ##print(f"x: {x.shape} | mask: {m.shape}")
        return x[~m]

    def get_target_mask(self, metadata):
        mask = metadata['target']['mask']
        #print("mask---->", mask.shape)
        return mask
    
    def _compute_loss(self, y_hat, y, agg=True, mask=None):
        if mask is not None:
            y_hat[mask] = 0
            y[mask] = 0
        # print("================================================================================")
        # print(y_hat.shape, y_hat.min(), y_hat.max())
        # print(y.shape, y.min(), y.max())
        if agg:
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat, y, reduction='none')
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y, metadata  = batch
        y_hat = self.forward(x)
        mask = self.get_target_mask(metadata)
        loss = self._compute_loss(y_hat, y, mask=mask)
        self.log('train_loss', loss)
        return loss
                
    def validation_step(self, batch, batch_idx, phase='val'):
        x, y, metadata  = batch
        y_hat = self.forward(x)
        mask = self.get_target_mask(metadata)
        loss = self._compute_loss(y_hat, y, mask=mask)

        
        idx_gt0 = y_hat>=self.config["train"]["val_logits_threshold"]
        y_hat[idx_gt0] = 1
        y_hat[~idx_gt0] = 0

        if mask is not None:
            y_hat[mask] = 0
            y[mask] = 0
    
        recall, precision, F1, acc, csi = recall_precision_f1_acc(y, y_hat)
        iou = iou_class(y_hat, y)

        #LOGGING
        self.log(f'{phase}_loss', loss)
        values = {'val_acc': acc, 'val_recall': recall, 'val_precision': precision, 'val_F1': F1, 'val_iou': iou, 'val_CSI': csi}
        self.log_dict(values)

        self.log("positive_ratio", float(y_hat.sum()) / np.prod(np.array(y_hat.shape)))
    
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, metadata = batch
        y_hat = self(x)
        
        # set the logits threshold equivalent to sigmoid(x)>=0.5
        idx_gt0 = y_hat>=self.config["predict"]["val_logits_threshold"]
        y_hat[idx_gt0] = 1
        y_hat[~idx_gt0] = 0
        return y_hat

    def configure_optimizers(self):
        if VERBOSE: print("Learning rate:",self.config["train"]["lr"], "| Weight decay:",self.config["train"]["weight_decay"])
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config["train"]["lr"]), weight_decay=float(self.config["train"]["weight_decay"])) 
        # optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config["train"]["lr"]))
        return optimizer

    def seq_metrics(self, y_true, y_pred):
        text = ''
        cm = confusion_matrix(y_true, y_pred).ravel()
        if len(cm)==4:
            tn, fp, fn, tp = cm
            recall, precision, F1 = 0, 0, 0

            if (tp + fn) > 0:
                recall = tp / (tp + fn)
            r = f'r: {recall:.2f}'
            
            if (tp + fp) > 0:
                precision = tp / (tp + fp)
            p = f'p: {precision:.2f}'

            if (precision + recall) > 0:
                F1 = 2 * (precision * recall) / (precision + recall)
            f = f'F1: {F1:.2f}'

            acc = (tn + tp) / (tn+fp+fn+tp)
            text = f"{r} | {p} | {f} | acc: {acc} "

        return text


#PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, targets, inputs, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                     
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class FocalLoss(nn.Module):
    def __init__(self, pos_weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inputs, targets, alpha=2, gamma=0.8):
        bce = self.bce(inputs, targets)
        bce_exp = torch.exp(-bce)
        focal_loss = alpha * (1 - bce_exp)**gamma * bce

        return focal_loss
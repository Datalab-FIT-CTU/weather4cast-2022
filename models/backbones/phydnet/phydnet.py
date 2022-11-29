import torch
import random

from .layers.phydnet_layers import ConvLSTM, PhyCell, EncoderRNN
from .base_model import PLBaseModule
from . import loss as module_loss
from . import metric_detail as module_metric_detail


class PhyDNet(PLBaseModule):
    """PhyDNet implementation, adapted from https://github.com/vincent-leguen/PhyDNet/blob/master/main.py"""
    def __init__(self, config):
        self.config = config

        loss = getattr(module_loss, config['loss']['type'])(**config['loss']['args'])

        metrics = {}
        for m,a in config['metrics_detail']:
            _metric = getattr(module_metric_detail, m)(**a)
            metrics.update({_metric.__name__: _metric})
        
        super().__init__(config, loss=loss, metrics=metrics, output_length=config['arch']['args']['len_out'],
            ndim=5, debug_val_idxs=config['arch'].get('debug_val_idxs'))

        # initialize optimization stuff
        self.teacher_forcing_ratio = 1.
        self.phycell_pred_only = config['arch'].get('phycell_pred_only', False)
        self.intensity_class = config['arch'].get('intensity_class', False)
        if self.intensity_class:
            self.icloss = module_loss.ICLoss(alpha=5., beta=0.67)
        self.convcell_norm_lambda = config['arch'].get('convcell_norm_lambda', 0)
        if self.convcell_norm_lambda:
            self.normloss = module_loss.ModuleNormLoss(alpha=config['arch'].get('convcell_norm_alpha', 0))

        # initialize model stuff
        self.patch_size = config['arch'].get('encoder_args', {}).get('patch_size', 1)
        _input_shape = (config['arch']['args']['img_height']//4//self.patch_size, config['arch']['args']['img_width']//4//self.patch_size)
        
        if config['arch'].get('phycell_args', None) is not None:
            self.phycell  =  PhyCell(input_shape=_input_shape, **config['arch']['phycell_args'])
        else:
            self.phycell = None
        if config['arch'].get('convcell_args', None) is not None:
            self.convcell =  ConvLSTM(input_shape=_input_shape, **config['arch']['convcell_args'])
        else:
            self.convcell = None
        self.encoder = EncoderRNN(self.phycell, self.convcell,
                                  intensity_class=self.intensity_class,
                                  **config['arch'].get('encoder_args', {}))

    def set_input_shape(self, height, width):
        _input_shape = (height//4//self.patch_size, width//4//self.patch_size)
        if self.phycell is not None:
            self.phycell.input_shape = _input_shape
        if self.convcell is not None:
            self.convcell.input_shape = _input_shape

    def set_output_length(self, output_length):
        super().set_output_length(output_length)

    def infer(self, x, **kwargs):
        self.set_input_shape(*x.shape[-2:])
        return super().infer(x, **kwargs)

    def compute_loss(self, output, target, loss_dict):
        o = output[:, :1] if self.intensity_class else output
        icl = ccn = 0.

        l = self.loss(o, target)
        loss_dict['img_loss'] = loss_dict.get('img_loss', 0) + l.detach()

        if self.intensity_class:
            icl = self.icloss(output[:, 1:], target) * self.intensity_class
            loss_dict['ic_loss'] = loss_dict.get('ic_loss', 0) + icl.detach()

        if self.convcell_norm_lambda:
            ccn = self.normloss(self.convcell) * self.convcell_norm_lambda
            loss_dict['convcell_norm'] = loss_dict.get('convcell_norm', 0) + ccn.detach()

        return l + icl + ccn

    def forward(self, input_tensor, target_tensor=None, compute_loss=False, teacher_forcing_ratio=0):
        assert not compute_loss or target_tensor is not None

        # input_tensor : torch.Size([batch_size, input_length, channels, rows, cols])
        input_length  = input_tensor.size(1)
        target_length = self.output_length # target_tensor.size(1)

        loss = 0
        loss_dict = {}
        for ei in range(input_length - 1):
            output_image = self.encoder(input_tensor[:,ei,:,:,:], (ei==0), activation_last=False)
            if compute_loss:
                loss += self.compute_loss(output_image, input_tensor[:,ei+1,:,:,:], loss_dict)

        decoder_input = input_tensor[:,-1,:,:,:] # first decoder input = last image of input sequence
        predictions = []
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
        for di in range(target_length):
            output_image = self.encoder(decoder_input, decoding=getattr(self, 'phycell_pred_only', False), activation_last=False)

            target = target_tensor[:,di,:,:,:] if target_tensor is not None else None
            if target is not None and compute_loss:
                loss += self.compute_loss(output_image, target, loss_dict)
            
            if target is not None and use_teacher_forcing:
                decoder_input = target # Teacher forcing    
            else:
                decoder_input = output_image # [B, CH, H, W]

            predictions.append(output_image)

        predictions = torch.stack(predictions, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        ret = predictions

        if compute_loss:
            loss = loss / target_length
            for k,v in loss_dict.items():
                loss_dict[k] = v / target_length
            reg, loss_dict_phycell = self.phycell.regularize()
            loss = loss + reg
            loss_dict.update(loss_dict_phycell)
            loss_dict.update({'loss': loss.detach()})
            ret = ret, loss, loss_dict

        return ret

    def training_step(self, train_batch, batch_idx):
        input_tensor, target_tensor = train_batch

        self.teacher_forcing_ratio = \
            max(0., self.teacher_forcing_ratio - self.config['arch']['args']['teacher_step'])
        self.log("teacher_forcing_ratio", self.teacher_forcing_ratio)

        output, loss, loss_dict = self(input_tensor, target_tensor=target_tensor,
                                       compute_loss=True,teacher_forcing_ratio=self.teacher_forcing_ratio)

        train_losses = {f'train/{k}': v.detach() for k,v in loss_dict.items()}
        self.log_dict(train_losses)

        return loss

    def validation_step(self, val_batch, batch_idx):
        input_tensor, target_tensor = val_batch

        output, loss, res = self(input_tensor, target_tensor=target_tensor, compute_loss=True, teacher_forcing_ratio=0)
        
        # res = {'loss': loss.detach().view(-1)}

        for key, loss_fn in self.metrics.items():
            res[key] = loss_fn(output, target_tensor).mean()
        
        self.log_dict(res)
        return res

    def configure_optimizers(self):
        opt_cls = getattr(torch.optim, self.config['optimizer']['type'])
        opt = opt_cls(self.parameters(), **self.config['optimizer']['args'])

        # sch = ReduceLROnPlateau(opt, mode='min', patience=2, factor=0.1, verbose=True)
        return [opt], [] # [sch]
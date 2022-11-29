from abc import abstractmethod

import numpy as np

import pytorch_lightning as pl
import torch

from . import metric_detail as module_metric_detail


class BaseModel(pl.LightningModule):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class PLBaseModule(pl.LightningModule):
    def __init__(self, config=None, loss=torch.nn.MSELoss(), metrics=None, output_length=6, ndim=5,
                 dataset_train=None, dataset_val=None, dataset_test=None, debug_val_idxs=None):
        super().__init__()

        self.config = config
        self.loss = loss
        if metrics is None and config is not None:
            self.metrics = {}
            for m,a in config['metrics_detail']:
                _metric = getattr(module_metric_detail, m)(**a)
                self.metrics.update({_metric.__name__: _metric})
        else:
            self.metrics = metrics if metrics is not None else {}
        assert isinstance(self.metrics, dict), 'metrics has to be dict'

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test

        self.output_length = output_length
        self.ndim = ndim

        self.debug_val_idxs = debug_val_idxs if debug_val_idxs is not None else []

    @abstractmethod
    def set_input_shape(self, height, width):
        """
        Change expected shape of the input.
        """
        raise NotImplementedError

    def set_output_length(self, output_length):
        self.output_length = output_length
            

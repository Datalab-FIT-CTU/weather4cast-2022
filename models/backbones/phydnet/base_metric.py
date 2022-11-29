from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn

import ray

@ray.remote
def _par_func(o, t, i, f):
    return f(o[i].squeeze(), t[i].squeeze())

class BaseMetric(nn.Module):
    """
    Base class for detailed metrics
    """
    def __init__(self, name, ndim=4, work_type=torch.Tensor, par: bool = False):
        super().__init__()
        
        self.__name__ = name
        
        self.ndim = None
        self._check_ndim(ndim)

        self.device = 'cpu'
        self.work_type = work_type
        self.correct_type = True

        self.par = par

    def _check_ndim(self, ndim):
        if self.ndim == ndim:
            return

        self.ndim = max(ndim, 4)
        self.avg_dims = list(range(2, self.ndim))

    @abstractmethod
    def forward(self, output, target):
        """
        Forward pass logic

        :return: computed metric
        """
        raise NotImplementedError

    def preprocess(self, output, target):
        with torch.no_grad():
            assert output.shape == target.shape, f'Tensors have different shapes {output.shape} and {target.shape}'
            assert type(output) == type(target), f'Tensors have different data types.'
            self._check_ndim(output.ndim)

            if isinstance(output, np.ndarray) and (self.work_type == torch.Tensor):
                self.correct_type = False

                output = torch.from_numpy(output)
                target = torch.from_numpy(target)
            elif isinstance(output, torch.Tensor) and (self.work_type == np.ndarray):
                self.correct_type = False
                self.device = output.get_device() if output.get_device() >= 0 else 'cpu'

                output = output.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
            else:
                self.correct_type = True

            while output.ndim < self.ndim:
                output = output[None, ...]
                target = target[None, ...]

        return output, target

    def postprocess(self, met):
        if self.correct_type is False:
            if self.work_type == np.ndarray:
                met = torch.from_numpy(met).to(self.device)
            elif self.work_type == torch.Tensor:
                met = met.cpu().detach().numpy()

        return met
    
    def compute_images_isolated(self, output, target, func):
        with torch.no_grad():
            res_shape = output.shape[:2]
            _output = output.reshape(-1, *output.shape[2:])
            _target = target.reshape(-1, *target.shape[2:])

            _lib = np if self.work_type == np.ndarray else torch # not nice, but it works for both work types
            if ray.is_initialized() and self.par:
                _output_ray = ray.put(_output)
                _target_ray = ray.put(_target)

                _res = ray.get([_par_func.remote(_output_ray, _target_ray, i, func) for i in range(_output.shape[0])])

                res = _lib.stack(_res)
            else:
                res = _lib.zeros(res_shape, dtype=output.dtype).flatten()

                for i in range(_output.shape[0]): # process 2D images separately
                    res[i] = func(_output[i].squeeze(), _target[i].squeeze())

        return res.reshape(res_shape)

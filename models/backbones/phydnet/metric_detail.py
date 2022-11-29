from .base_metric import BaseMetric

from abc import abstractmethod
import numpy as np
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn

class mae(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__('mae', work_type=torch.Tensor, **kwargs)

    def forward(self, output, target):
        with torch.no_grad():
            output, target = self.preprocess(output, target)
            losses = nn.L1Loss(reduction='none')(output, target).mean(dim=self.avg_dims)
        return self.postprocess(losses)

class mse(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__('mse', work_type=torch.Tensor, **kwargs)

    def forward(self, output, target):
        with torch.no_grad():
            output, target = self.preprocess(output, target)
            losses = nn.MSELoss(reduction='none')(output, target).mean(dim=self.avg_dims)
        return self.postprocess(losses)

# -----------------------------------------------------------------------------

class SkillScore(BaseMetric):
    def __init__(self, name, threshold=0.2, **kwargs):
        super().__init__(name, work_type=torch.Tensor, **kwargs)
        self.threshold = threshold

    def convert(self, x):
        one = torch.Tensor([1]).type_as(x)#to('cuda' if x.get_device() >= 0 else 'cpu')
        zero = torch.Tensor([0]).type_as(x)#to('cuda' if x.get_device() >= 0 else 'cpu')
        
        return torch.where(x >= self.threshold, one, zero).bool()
    
    # count for every image in sequence & batch alone
    def count_TP(self, output, target):
        """ count True Positives """
        return torch.sum(output & target, dim=self.avg_dims).float()

    def count_FN(self, output, target):
        """ count False Negatives """
        return torch.sum((output ^ target) & target, dim=self.avg_dims).float()
    
    def count_FP(self, output, target):
        """ count False Positives """
        return torch.sum((output ^ target) & output, dim=self.avg_dims).float()

    def count_TN(self, output, target):
        """ count True Negatives """
        return torch.sum(~(output | target), dim=self.avg_dims).float()
    
    def forward(self, output, target):
        with torch.no_grad():
            output, target = self.preprocess(output, target)

            O = self.convert(output)
            T = self.convert(target)
            
            TP = self.count_TP(O, T)
            FN = self.count_FN(O, T)
            FP = self.count_FP(O, T)
            TN = self.count_TN(O, T)

            res = self.postprocess(self.compute(O, T, TP, FN, FP, TN))

        return res

    @abstractmethod
    def compute(self, O, T, TP, FN, FP, TN):
        """
        Method for actual skill score computation

        Returns
        -------
        x : np.ndarray
            Skill score value.
        """
        raise NotImplementedError

class CSI(SkillScore):
    def __init__(self, threshold, remove_nan=False, **kwargs):
        super().__init__(f'CSI_{threshold:.2f}', threshold, **kwargs)
        self.remove_nan = remove_nan

    def compute(self, O, T, TP, FN, FP, TN):
        x = TP / (TP + FN + FP)
        if self.remove_nan:
            x[x != x] = 0

        return x

# -----------------------------------------------------------------------------

class KSDistMP(BaseMetric):
    """Kolmogorov-Smirnov Distance, implemented according to 1951 paper.
    
    https://luk.staff.ugm.ac.id/jurnal/freepdf/2280095Massey-Kolmogorov-SmirnovTestForGoodnessOfFit.pdf
    
    Args:
        bins: list of bin centers for ECDF computation
    """
    def __init__(self, bins=None, **kwargs):
        super().__init__('KSDist', work_type=np.ndarray)
        
        if bins is not None:
            self.bins = bins
        else:
            self.bins = np.array([0] + [1/30 + i*1/15 for i in range(15)] + [1])
            
    def get_ecdf(self, x, bins=None):
        """Compute empirical cumulative distribution function.
        """
        bins = self.bins if bins is None else bins
        
        counts, _ = np.histogram(x, bins=bins)
        cusum = np.cumsum(counts)
        
        return cusum / cusum[-1]

    def get_ksd(self, o, t):
        o_cdf = self.get_ecdf(o)
        t_cdf = self.get_ecdf(t)
        
        return np.abs(o_cdf - t_cdf).max()
        
    def forward(self, output, target):
        output, target = self.preprocess(output, target)
        res = self.compute_images_isolated(output, target, \
            lambda o, t: self.get_ksd(o, t))

        return self.postprocess(res)

class SSIM(BaseMetric):
    """Wrapper for scikit-image SSIM.
    """
    def __init__(self, **kwargs):
        super().__init__('SSIM', work_type=np.ndarray)
    
    def forward(self, output, target):
        output, target = self.preprocess(output, target)
        res = self.compute_images_isolated(output, target, \
            lambda o, t: ssim(o, t))

        return self.postprocess(res)

class PredictionMean(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__('prediction_mean', work_type=torch.Tensor, **kwargs)
        
    def forward(self, output, target):
        with torch.no_grad():
            output, target = self.preprocess(output, target)
            losses = torch.mean(output, dim=self.avg_dims)
        return self.postprocess(losses)
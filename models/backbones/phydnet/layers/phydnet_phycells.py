from typing import List, Tuple
import torch
import torch.nn as nn

from ..base_model import BaseModel
from .phydnet_constrain_moments import K2M, M2K


def get_phy_conv(in_channels, out_channels, kernel_size, norm='bn', bias=True,
                 init_yx: Tuple[int, int] = None):
    m2k = M2K(list(kernel_size))
    _padding = kernel_size[0] // 2, kernel_size[1] // 2

    f = nn.Sequential()
    f.add_module('conv1', nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=(1, 1), padding=_padding,
                                    bias=bias))

    if norm == 'bn':
        f.add_module('bn1', nn.BatchNorm2d(out_channels))
    elif norm == 'gn':
        raise NotImplementedError

    # initialize to correct differential operator if wanted
    if init_yx:
        m = torch.zeros(kernel_size)
        m[init_yx[0], init_yx[1]] = 1
        w = m2k(m.double()).float()
        f.conv1.weight.data[:, :] = w # copy in channel dims
    
    return f

class BasePhyCell_Cell(BaseModel):
    """Base PhyCell_Cell defining regularization of differential operators.
    """
    L2 = nn.MSELoss() # needed for moment regularization                      

    def __init__(self, input_dim: int = 64, F_hidden_dim: int = [49],
                 kernel_size: Tuple[int, int] = (7, 7), bias: int = 1,
                 lambda_moment: float = 0.2, *args, **kwargs) -> None:
        super().__init__()

        self.input_dim      = input_dim
        self.F_hidden_dim   = F_hidden_dim
        self.kernel_size    = kernel_size
        self.padding        = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias           = bias
        self.lambda_moment  = lambda_moment

        self.init_regularization()

        self.K = None

    def init_prediction(self, batch_size, input_shape):
        pass    

    def init_regularization(self):
        # constraints for forcing differential operators weights
        self.register_buffer('constraints', torch.zeros((
            self.F_hidden_dim, *self.kernel_size)))
        ind = 0
        for i in range(0, self.kernel_size[0]):
            for j in range(0, self.kernel_size[1]):
                self.constraints[ind, i, j] = 1
                ind += 1

        self.k2m = K2M(list(self.kernel_size)).to(self.device)

    def regularize(self):
        """Moment regularization for forcing weights of differential operators.
        
        Returns
        -------
            Value of the moment regularization loss.
        """
        loss = 0.

        # moment regularization of differential operators
        # size (F_hidden_dim, input_dim, K, K)
        for b in range(0, self.input_dim):
            filters = self.F.conv1.weight[:, b, :, :] # (F_hidden_dim, K, K)     
            m = self.k2m(filters.double()).float()
            loss += self.L2(m, self.constraints) # constraints is a precomputed matrix
        loss *= self.lambda_moment

        return loss, {'moment': loss.detach()}

class PhyCell_Cell_1(BasePhyCell_Cell):
    """PhyCell_Cell with brute-force non-linearity.

    Computes all 2nd degree terms from the partial derivatives
    using matrix multiplication and includes them in the creation
    of the final polynomial.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # approximation of partial derivatives
        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=self.input_dim,
                                             out_channels=self.F_hidden_dim, 
                                             kernel_size=self.kernel_size,
                                             stride=(1,1),
                                             padding=self.padding))
        self.F.add_module('bn1', nn.GroupNorm(self.kernel_size[0], self.F_hidden_dim))

        # layers for selection of higher degree terms
        self._uti = torch.triu_indices(self.F_hidden_dim, self.F_hidden_dim) # prepare upper triangular due to duplicates
        _total_terms = self.F_hidden_dim + self._uti.shape[1]
 
        self.register_buffer('constraints',
                             torch.zeros((_total_terms, 1, 1))) # TODO only for back compatibility

        # 1x1 linear combination of partial derivatives and their combinations
        self.linear_combination = nn.Conv2d(in_channels=_total_terms,
                                            out_channels=self.input_dim,
                                            kernel_size=(1,1), stride=(1,1),
                                            padding=(0,0))

        # Kalman filter gating parameter
        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                                  out_channels= self.input_dim,
                                  kernel_size=(3,3),
                                  padding=(1,1), bias=self.bias)

    def forward(self, x, hidden): # x [batch_size, hidden_dim, height, width]      
        partials = self.F(hidden) # compute partial derivatives
        # compute higher order terms
        d2_terms = torch.einsum('bihw,bjhw->bijhw', partials, partials) # compute pair-wise products
        d2_terms = d2_terms[:, self._uti[0], self._uti[1]] # select upper triangular due to duplicates
        partials = torch.cat([partials, d2_terms], dim=1) # channel axis

        hidden_tilde = hidden + self.linear_combination(partials) # prediction
        
        # correction using Kalman filter
        if x is not None:
            combined = torch.cat([x, hidden], dim=1) # channel axis
            combined_conv = self.convgate(combined)
            K = torch.sigmoid(combined_conv)
            self.K = K
            next_hidden = hidden_tilde + K * (x - hidden_tilde) # correction , Haddamard product
        else:
            next_hidden = hidden_tilde

        return next_hidden


# =============================================================================
# =============================================================================
class PhyCell_Cell_AdvDiff(BasePhyCell_Cell):
    """PhyCell_Cell with hand-engineered Advection-Diffusion equation.
    """
    def __init__(self, U_kernel: Tuple[int, int] = (5, 5),
                 continuity_loss: bool = False,
                 qij_bias: bool = True, qij_norm: str = 'bn',
                 qij_init: bool = False, terms_gn_groups: int = None,
                 **kwargs):
        assert kwargs.get('F_hidden_dim') == 4
        super().__init__(**kwargs)

        def _qij_init_if(y, x, ctr=qij_init):
            return (y, x) if ctr else None
        qij_kwargs = {
            'kernel_size': self.kernel_size,
            'norm': qij_norm,
            'bias': qij_bias,
        }
        
        # approximation of partial derivatives
        self.F = [
            get_phy_conv(in_channels=self.input_dim, out_channels=1,
                         init_yx=_qij_init_if(0, 1), **qij_kwargs),
            get_phy_conv(in_channels=self.input_dim, out_channels=1,
                         init_yx=_qij_init_if(1, 0), **qij_kwargs),
            get_phy_conv(in_channels=self.input_dim, out_channels=1,
                         init_yx=_qij_init_if(0, 2), **qij_kwargs),
            get_phy_conv(in_channels=self.input_dim, out_channels=1,
                         init_yx=_qij_init_if(2, 0), **qij_kwargs),
        ]
        self.F = nn.ModuleList(self.F)

        # simple partial differentiations w.r. to x or y
        if continuity_loss:
            self.Fx = get_phy_conv(in_channels=1, out_channels=1,
                                   init_yx=_qij_init_if(0, 1), **qij_kwargs)
            self.Fy = get_phy_conv(in_channels=1, out_channels=1,
                                   init_yx=_qij_init_if(1, 0), **qij_kwargs)

        self.U_kernel = U_kernel
        self.U = get_phy_conv(in_channels=self.input_dim, out_channels=2,
                              kernel_size=self.U_kernel)

        self.terms_gn = nn.GroupNorm(terms_gn_groups, 4) if terms_gn_groups else None

        # 1x1 linear combination of partial derivatives and their combinations
        self.linear_combination = nn.Conv2d(in_channels=self.F_hidden_dim,
                                            out_channels=self.input_dim,
                                            kernel_size=(1,1), stride=(1,1),
                                            padding=(0,0))

        # Kalman filter gating parameter
        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                                  out_channels= self.input_dim,
                                  kernel_size=(3,3),
                                  padding=(1,1), bias=self.bias)

        self.continuity_loss_ctr = continuity_loss
        self.u_continuity_loss = 0 # loss guarding continuity of speed vectors
        self.speed = None
        self._hidden = None
        self._phi = None
        self._terms = None
        self._terms_gn = None

    def forward(self, x, hidden): # x [batch_size, hidden_dim, height, width]
        self.speed = self.U(hidden) # compute speed vectors u
        self._hidden = hidden

        # save result of continuity equation -> to force it to 0 in .regularize()
        if self.continuity_loss_ctr:
            self.u_continuity_loss += \
                torch.pow(self.Fx(self.speed[:, 0:1]) + self.Fy(self.speed[:, 1:2]), 2).mean()

        # compute advection terms
        vh_x = self.F[0](self.speed[:, 0:1] * hidden)
        wh_y = self.F[1](self.speed[:, 1:2] * hidden)

        # compute diffusion terms
        h_xx = self.F[2](hidden)
        h_yy = self.F[3](hidden)

        terms = torch.cat([vh_x, wh_y, h_xx, h_yy], dim=1) # channel axis
        self._terms = terms
        if self.terms_gn is not None:
            terms = self.terms_gn(terms)
        self._terms_gn = terms

        self._phi = self.linear_combination(terms)
        hidden_tilde = hidden + self._phi # prediction
        
        # correction using Kalman filter
        if x is not None:
            combined = torch.cat([x, hidden], dim=1) # channel axis
            combined_conv = self.convgate(combined)
            K = torch.sigmoid(combined_conv)
            self.K = K # save Kalman gate for exploration purposes
            next_hidden = hidden_tilde + K * (x - hidden_tilde) # correction , Haddamard product
        else:
            next_hidden = hidden_tilde

        return next_hidden

    def init_regularization(self):
        # constraints for forcing differential operators weights
        self.register_buffer('constraints', torch.zeros((
            self.F_hidden_dim, *self.kernel_size)))

        self.constraints[0, 0, 1] = 1 # dx
        self.constraints[1, 1, 0] = 1 # dy
        self.constraints[2, 0, 2] = 1 # dx^2
        self.constraints[3, 2, 0] = 1 # dy^2

        self.k2m = K2M(list(self.kernel_size)).to(self.device)

    def regularize(self):
        loss = 0.
        loss_moment = 0.

        filters = [self.F[i].conv1.weight for i in range(self.F_hidden_dim)]
        filters = torch.cat(filters, dim=0)

        # moment regularization of differential operators
        # size (F_hidden_dim, input_dim, K, K)
        for b in range(0, self.input_dim):
            m = self.k2m(filters[:, b].double()).float() # (F_hidden_dim, K, K)
            loss_moment += self.L2(m, self.constraints) # constraints is a precomputed matrix

        # regularize simple differentiations
        if self.continuity_loss_ctr:
            f = [self.Fx.conv1.weight, self.Fy.conv1.weight]
            f = torch.cat(f, dim=0)
            m = self.k2m(f[:, 0].double()).float()
            loss_moment += self.L2(m, self.constraints[:2])
        
        loss_moment *= self.lambda_moment
        loss_dict = {'moment': loss_moment.detach()}
        loss += loss_moment

        # regularize speed vectors with continuity equation
        if self.continuity_loss_ctr:
            loss_dict.update({'continuity': self.u_continuity_loss.detach()})
            loss += self.u_continuity_loss
            self.u_continuity_loss = 0

        return loss, loss_dict
# =============================================================================
# =============================================================================
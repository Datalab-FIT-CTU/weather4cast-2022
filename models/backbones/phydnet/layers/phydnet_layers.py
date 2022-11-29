from re import A
import numpy as np
import torch
import torch.nn as nn

from ..base_model import BaseModel
from .phydnet_constrain_moments import K2M
from .phydnet_phycells import BasePhyCell_Cell, PhyCell_Cell_1, PhyCell_Cell_AdvDiff

from icecream import ic

def reshape_patch(img_tensor, patch_size):
    # [B, C, H, W] -> [B, C*P*P, H/P, W/P]

    # change to Torch if NumPy
    if isinstance(img_tensor, np.ndarray):
        img_tensor = torch.tensor(img_tensor)
        as_numpy = True
    else:
        as_numpy = False

    assert 4 == img_tensor.ndim, f'img_tensor shape is {img_tensor.shape}, maybe wanted to use reshape_patch_seq'

    # temp [..., C, H, W] -> [..., H, W, C]
    img_tensor = img_tensor.permute(0, 2, 3, 1)
    
    batch_size = img_tensor.shape[0]
    img_height = img_tensor.shape[1]
    img_width = img_tensor.shape[2]
    num_channels = img_tensor.shape[3]
    a = img_tensor.reshape([batch_size,
                            img_height//patch_size, patch_size,
                            img_width//patch_size, patch_size,
                            num_channels])
    b = a.transpose(2, 3)
    patch_tensor = b.reshape([batch_size,
                                img_height//patch_size, img_width//patch_size,
                                patch_size*patch_size*num_channels])

    # reverse temp [..., H/P, W/P, C*P*P] -> [..., C*P*P, H/P, W/P]
    patch_tensor = patch_tensor.permute(0, 3, 1, 2).contiguous()
    
    # reverse change to NumPy
    if as_numpy:
        patch_tensor = patch_tensor.cpu().detach().numpy()
    
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    # [B, C*P*P, H/P, W/P] -> [B, C, H, W]

    # change to Torch if NumPy
    if isinstance(patch_tensor, np.ndarray):
        patch_tensor = torch.tensor(patch_tensor)
        as_numpy = True
    else:
        as_numpy = False

    assert 4 == patch_tensor.ndim

    # temp [..., C*P*P, H/P, W/P] -> [..., H/P, W/P, C*P*P]
    patch_tensor = patch_tensor.permute(0, 2, 3, 1)

    batch_size = patch_tensor.shape[0]
    patch_height = patch_tensor.shape[1]
    patch_width = patch_tensor.shape[2]
    channels = patch_tensor.shape[3]
    img_channels = channels // (patch_size*patch_size)
    a = patch_tensor.reshape([batch_size,
                                patch_height, patch_width,
                                patch_size, patch_size,
                                img_channels])
    b = a.transpose(2, 3)
    img_tensor = b.reshape( [batch_size,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])

    # reverse temp [..., H, W, C] -> [..., C, H, W]
    img_tensor = img_tensor.permute(0, 3, 1, 2).contiguous()
    
    # reverse change to NumPy
    if as_numpy:
        img_tensor = img_tensor.cpu().detach().numpy()
    
    return img_tensor

# -----------------------------------------------------------------------------
# adapted from https://github.com/vincent-leguen/PhyDNet/blob/master/models/models.py
# -----------------------------------------------------------------------------

class PhyCell_Cell(BasePhyCell_Cell):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__(input_dim, F_hidden_dim,
                                           kernel_size, bias)
        
        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=(1,1), padding=self.padding))
        self.F.add_module('bn1',nn.GroupNorm(kernel_size[0], F_hidden_dim))        
        self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1,1), stride=(1,1), padding=(0,0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                              out_channels= self.input_dim,
                              kernel_size=(3,3),
                              padding=(1,1), bias=self.bias)
        self.K = None

    def forward(self, x, hidden): # x [batch_size, hidden_dim, height, width]      
        hidden_tilde = hidden + self.F(hidden)        # prediction
        
        if x is not None:
            combined = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
            combined_conv = self.convgate(combined)
            K = torch.sigmoid(combined_conv)
            self.K = K
            next_hidden = hidden_tilde + K * (x-hidden_tilde)   # correction , Haddamard product
        else:
            next_hidden = hidden_tilde
        return next_hidden

class PhyCell(BaseModel):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size,
                 cell_version=0, cell_kwargs=None):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []

        self.cell_list = self._init_cells(cell_version, cell_kwargs)

        # constraints for forcing differential operators weights
        # TODO only for back compatibility
        self.register_buffer('constraints', torch.zeros((
            *self.F_hidden_dims, *self.kernel_size)))

    def _init_cells(self, cell_version, cell_kwargs):
        """Initializes a list of PhyCell cells.
        """
        if cell_version == 0:
            _cell_class = PhyCell_Cell 
        elif cell_version == 1:
            _cell_class = PhyCell_Cell_1
        elif cell_version == 2:
            _cell_class = PhyCell_Cell_AdvDiff
        _cell_kwargs = cell_kwargs if cell_kwargs is not None else {}

        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(_cell_class(input_dim=self.input_dim,
                                         F_hidden_dim=self.F_hidden_dims[i],
                                         kernel_size=self.kernel_size,
                                         **_cell_kwargs))

        return nn.ModuleList(cell_list)
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        if (first_timestep):   
            batch_size = input_.data.size()[0]
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j-1],self.H[j])
        
        return self.H , self.H 
    
    def initHidden(self, batch_size):
        self.H = [] 
        for i in range(self.n_layers):
            self.H.append( torch.zeros((batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]), device=torch.device(self.device), dtype=self.dtype) )
        
        for cell in self.cell_list:
            cell.init_prediction(batch_size, self.input_shape)

    def setHidden(self, H):
        self.H = H

    def regularize(self):
        """Computes PhyCell regularization loss.

        Performs all regularizations of PhyCell cells.

        Returns
        -------
            Value of the regularization loss.
        """
        loss = 0.
        loss_dict = {}
        
        for i, _cell in enumerate(self.cell_list):
            # regularizations
            if hasattr(_cell, 'regularize') is True:
                l, d = _cell.regularize()
                loss += l
                loss_dict.update({f'{i}_{k}': v for k, v in d.items()})

        return loss, loss_dict
        
class ConvLSTM_Cell(BaseModel):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):              
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()
        
        self.height, self.width = input_shape
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)
                 
    # we implement LSTM that process only one timestep 
    def forward(self,x, hidden): # x [batch, hidden_dim, width, height]          
        h_cur, c_cur = hidden
        
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(BaseModel):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [],[]   
        
        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j],self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1],(self.H[j],self.C[j]))
        
        return (self.H,self.C) , self.H   # (hidden, output)
    
    def initHidden(self,batch_size):
        self.H, self.C = [],[]  
        for i in range(self.n_layers):
            self.H.append( torch.zeros((batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]), device=torch.device(self.device), dtype=self.dtype) )
            self.C.append( torch.zeros((batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]), device=torch.device(self.device), dtype=self.dtype) )
    
    def setHidden(self, hidden):
        H,C = hidden
        self.H, self.C = H,C
 

class dcgan_conv(BaseModel):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3,3), stride=stride, padding=1),
                nn.GroupNorm(16,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(BaseModel):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride ==2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels=nin,out_channels=nout,kernel_size=(3,3), stride=stride,padding=1,output_padding=output_padding),
                nn.GroupNorm(16,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)
        
class encoder_E(BaseModel):
    def __init__(self, nc=1, nf=32):
        super(encoder_E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nc, nf, stride=2) # (32) x 32 x 32
        self.c2 = dcgan_conv(nf, nf, stride=1) # (32) x 32 x 32
        self.c3 = dcgan_conv(nf, 2*nf, stride=2) # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)  
        h2 = self.c2(h1)    
        h3 = self.c3(h2)
        return h3

class decoder_D(BaseModel):
    def __init__(self, nc=1, nf=32, intensity_class: bool = False):
        super(decoder_D, self).__init__()
        self.upc1 = dcgan_upconv(2*nf, nf, stride=2) #(32) x 32 x 32
        self.upc2 = dcgan_upconv(nf, nf, stride=1) #(32) x 32 x 32
        self.upc3 = nn.ConvTranspose2d(in_channels=nf,out_channels=nc,kernel_size=(3,3),stride=2,padding=1,output_padding=1)  #(nc) x 64 x 64

        # create optional layer for classification of intensities
        self.intensity_class = intensity_class
        if intensity_class:
            self.upc3_ic = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, input):      
        d1 = self.upc1(input) 
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)

        if self.intensity_class:
            d3_ic = self.upc3_ic(d3)
            return torch.concat([d3, d3_ic], dim=-3)

        return d3  


class encoder_specific(BaseModel):
    def __init__(self, nc=64, nf=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1) # (64) x 16 x 16
        self.c2 = dcgan_conv(nf, nf, stride=1) # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)  
        h2 = self.c2(h1)     
        return h2

class decoder_specific(BaseModel):
    def __init__(self, nc=64, nf=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1) #(64) x 16 x 16
        self.upc2 = dcgan_upconv(nf, nc, stride=1) #(32) x 32 x 32
        
    def forward(self, input):
        d1 = self.upc1(input) 
        d2 = self.upc2(d1)  
        return d2       

        
class EncoderRNN(BaseModel):
    def __init__(self, phycell, convcell, patch_size=1, n_features=64,
                 n_channels: int = 1, intensity_class: bool = False, elev_path: str = None):
        super(EncoderRNN, self).__init__()
        self.patch_size = patch_size

        assert phycell is not None or convcell is not None, 'At least one of PhyCell or ConvCell needs to be used.'
        self.phycell = phycell.to(self.device) if phycell is not None else None
        self.convcell = convcell.to(self.device) if convcell is not None else None

        # elevation files
        if elev_path:
            self.register_buffer('elev', torch.from_numpy(np.load(elev_path))[None, None]) # [1, 1, H, W]

        # general encoder 64x64x1 -> 32x32x32
        self.encoder_E = encoder_E(n_channels*(patch_size**2), n_features//2).to(self.device)

        if phycell is not None:
            # specific image encoder 32x32x32 -> 16x16x64
            self.encoder_Ep = encoder_specific(n_features, n_features).to(self.device)
            # specific image decoder 16x16x64 -> 32x32x32
            self.decoder_Dp = decoder_specific(n_features, n_features).to(self.device)
        if convcell is not None:
            self.encoder_Er = encoder_specific(n_features, n_features).to(self.device)
            self.decoder_Dr = decoder_specific(n_features, n_features).to(self.device)
        
        # general decoder 32x32x32 -> 64x64x1
        self.intensity_class = intensity_class
        self.decoder_D = decoder_D(n_channels*(patch_size**2), n_features//2,
                                   intensity_class).to(self.device)

    def forward(self, input, first_timestep=False, decoding=False, activation_last=True):
        # use elevation if available
        if getattr(self, 'elev', None) is not None:
            input = torch.cat([input, self.elev.expand_as(input)], dim=1) # expand B, cat in C

        input = reshape_patch(input, self.patch_size) # [B, S, C, H, W] -> [B, S, C*P*P, H/P, W/P]
        input = self.encoder_E(input) # general encoder 64x64x1 -> 32x32x32     

        if decoding:  # input=None in decoding phase
            input_phys = None
            # input_phys = torch.zeros(input_conv.shape, dtype=input_conv.dtype, device=self.device) # my experiment
        elif self.phycell is not None:
            input_phys = self.encoder_Ep(input)

        # compute PhyCell physics branch
        if self.phycell is not None:
            hidden1, output1 = self.phycell(input_phys, first_timestep)
            decoded_Dp = self.decoder_Dp(output1[-1])
            # out_phys = self.decoder_D(decoded_Dp) # partial reconstructions for vizualization
            # out_phys = torch.sigmoid(out_phys) if activation_last else out_phys
        else:
            # out_phys = torch.tensor(0.)
            decoded_Dp = torch.tensor(0.)

        # compute ConvCell residual branch
        if self.convcell is not None:
            input_conv = self.encoder_Er(input)
            hidden2, output2 = self.convcell(input_conv, first_timestep)
            decoded_Dr = self.decoder_Dr(output2[-1])
            # out_conv = self.decoder_D(decoded_Dr) # partial reconstructions for vizualization
            # out_conv = torch.sigmoid(out_conv) if activation_last else out_conv
        else:
            # out_conv = torch.tensor(0.)
            decoded_Dr = torch.tensor(0.)

        # sum branches and decode
        output_image = self.decoder_D(decoded_Dp + decoded_Dr)
        output_image = torch.sigmoid(output_image) if activation_last else output_image

        # res = [out_phys, hidden1, output_image, out_phys, out_conv]
        # for i,r in enumerate(res):
        #     if not isinstance(r, torch.Tensor) or not r.dim():
        #         continue
        #     res[i] = reshape_patch_back(r, self.patch_size) # [B, S, C*P*P, H/P, W/P] -> [B, S, C, H, W]

        return reshape_patch_back(output_image, self.patch_size)
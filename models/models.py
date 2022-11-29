from utils.data_utils import load_config
import torch
from models.base_module import BaseLitModule



class UNetWrapper(torch.nn.Module):
    def __init__(self, input_channels, output_channels, nb_filter=None):
        super().__init__()
        self.input_channels = input_channels
        from models.backbones.unet import UNet
        self.model = UNet(input_channels=input_channels, num_classes=output_channels, nb_filter=nb_filter)

    def forward(self, x):
        img_w = x.shape[-2]
        img_h = x.shape[-1]

        pw = (32 - img_w % 32) // 2
        ph = (32 - img_h % 32) // 2

        x = x.reshape(-1, self.input_channels, img_w, img_h)
        x = torch.nn.functional.pad(x, (pw, pw, ph, ph), mode="replicate")  # 252x252 -> 256x256
        x = self.model(x)
        x = x.unsqueeze(1)  # add back channel dim
        x = x[..., pw:-pw, ph:-ph]  # back to 252x252
        return x


def crop_slice(img_size=252, scale_ratio=2/12):
    padding = int(img_size * (1 - scale_ratio) // 2)
    return ..., slice(padding, img_size-padding), slice(padding, img_size-padding)


class PhyDNetWrapper(torch.nn.Module):
    def __init__(self, config_path, ckpt_path=None):
        super().__init__()
        phydnet_config = load_config(config_path)
        from models.backbones.phydnet import PhyDNet
        if ckpt_path:
            self.phydnet = PhyDNet.load_from_checkpoint(ckpt_path, config=phydnet_config)
        else:
            self.phydnet = PhyDNet(phydnet_config)

    def forward(self, x):
        return self.phydnet(x)



# ---------------------------------
# LIGHTNING MODULES
# ---------------------------------



class UNetCropUpscale(BaseLitModule):
    def __init__(self, config):
        super().__init__(config)

        self.unet = UNetWrapper(
            input_channels=11 * config["dataset"]["len_seq_in"],
            output_channels=config["dataset"]["len_seq_predict"],
        )
        self.upscale = torch.nn.Upsample(scale_factor=6, mode='bilinear', align_corners=True)

    def forward(self, x, crop=True, upscale=True):
        x = self.unet(x)

        if crop:
            x = x[crop_slice()]

        if upscale:
            x = self.upscale(x[:, 0]).unsqueeze(1)

        return x


class WeatherFusionNet(BaseLitModule):
    def __init__(self, config):
        super().__init__(config)
        self.phydnet = PhyDNetWrapper("models/configurations/phydnet.yaml", ckpt_path="weights/sat-phydnet.ckpt")

        self.sat2rad = UNetWrapper(input_channels=11, output_channels=1)
        self.sat2rad.load_state_dict(torch.load("weights/sat2rad-unet.pt"))

        self.unet = UNetWrapper(input_channels=11 * (4 + 10) + 4, output_channels=32)
        self.upscale = torch.nn.Upsample(scale_factor=6, mode='bilinear', align_corners=True)

    def forward(self, x, return_inter=False):
        bs = x.shape[0]

        self.sat2rad.eval()
        with torch.no_grad():
            sat2rad_out = self.sat2rad(x.swapaxes(1, 2)).reshape(bs, 4, 252, 252)

        self.phydnet.eval()
        with torch.no_grad():
            phydnet_out = self.phydnet(x.swapaxes(1, 2)).reshape(bs, -1, 252, 252)

        x = torch.concat([x.reshape(bs, -1, 252, 252), phydnet_out, sat2rad_out], dim=1)
        unet_out = x = self.unet(x)
        x = x[crop_slice()]
        x = self.upscale(x[:, 0]).unsqueeze(1)

        if return_inter:
            return sat2rad_out, phydnet_out, unet_out, x
        return x

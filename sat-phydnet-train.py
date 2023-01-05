import argparse
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl


WANDB_LOGGING = False

if WANDB_LOGGING:
    import wandb
    wandb_logger = pl.loggers.WandbLogger(name="PhyDNet", project="w4c satellite")


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config_path", type=str, required=False, default='models/configurations/config.yaml',
                    help="path to config-yaml")
parser.add_argument("-p", "--phydnet_config_path", type=str, required=False, default='models/configurations/phydnet.yaml',
                    help="path to PhyDNet config-yaml")
parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=None, 
                    help="specify gpu(s): 1 or 1 5 or 0 1 2 etc. (leave default for no GPU)")
options = parser.parse_args()


from utils.data_utils import load_config
config = load_config(options.config_path)
phydnet_config = load_config(options.phydnet_config_path)

config["dataset"]["len_seq_in"] = 4 + phydnet_config["arch"]["args"]["len_out"]  # input + output
config["dataset"]["len_seq_predict"] = 1  # we don't need the radar data, but this can't be 0


from utils.w4c_dataloader import RainData

class SatOnly(RainData):
    def __getitem__(self, idx):
        sat, rad, metadata = super().__getitem__(idx)
        sat = sat.swapaxes(0, 1)
        return sat[:4], sat[4:]

dataset = SatOnly("training", **config["dataset"])
print(len(dataset))


# we can't easily use the actual validation set with the changed sequence lengths
# so we just split the train set instead
# TODO: shuffle before splitting
train_set = torch.utils.data.Subset(dataset, range(0, 150_000))
valid_set = torch.utils.data.Subset(dataset, range(150_000, len(dataset), 32))
print("Train size:", len(train_set))
print("Valid size:", len(valid_set))


from models.backbones.phydnet import PhyDNet


model = PhyDNet(phydnet_config)


BATCH_SIZE = 16
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16,
                          pin_memory=True, prefetch_factor=2, persistent_workers=False)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, num_workers=4,
                          pin_memory=True, prefetch_factor=2, persistent_workers=False)


checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="mse", mode="min")
trainer = pl.Trainer(
    gpus=options.gpus,
    callbacks=[
        checkpoint_callback,
        pl.callbacks.EarlyStopping(monitor="mse", mode="min", patience=10, strict=False),
    ],
    max_epochs=50,
    logger=wandb_logger if WANDB_LOGGING else None,
    default_root_dir="checkpoints",
    val_check_interval=2000,
)
trainer.fit(model, train_loader, valid_loader)


print(checkpoint_callback.best_model_path)
model = model.load_from_checkpoint(checkpoint_callback.best_model_path, config=phydnet_config)
results = trainer.validate(model, valid_loader)[0]
if WANDB_LOGGING:
    wandb.log(results)
    wandb.finish()

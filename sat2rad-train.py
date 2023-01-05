import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utils.w4c_dataloader import RainData
from utils.data_utils import load_config


WANDB_LOGGING = False

if WANDB_LOGGING:
    import wandb
    wandb_logger = pl.loggers.WandbLogger(name=f"UNet -> crop -> upscale", project="w4c sat2rad")


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config_path", type=str, required=False, default='models/configurations/config.yaml',
                    help="path to config-yaml")
parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=None, 
                    help="specify gpu(s): 1 or 1 5 or 0 1 2 etc. (leave default for no GPU)")
options = parser.parse_args()


config = load_config(options.config_path)
config["dataset"]["len_seq_in"] = 1
config["dataset"]["len_seq_predict"] = 1

train_set = RainData("training", **config["dataset"])
valid_set = RainData("validation", **config["dataset"])

# this shifts the output timestamps so they match the input
for sample in train_set.idxs + valid_set.idxs:
    sample[1][0] -= 1

print("Train size:", len(train_set))
print("Valid size:", len(valid_set))



BATCH_SIZE = 32
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16,
                          pin_memory=True, prefetch_factor=2, persistent_workers=False)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, num_workers=4,
                          pin_memory=True, prefetch_factor=2, persistent_workers=False)



from models.models import UNetCropUpscale

model = UNetCropUpscale(config)



checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    filename='{epoch:02d}-{val_loss:.6f}'
)
trainer = pl.Trainer(
    gpus=options.gpus,
    callbacks=[
        checkpoint_callback,
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            strict=False
        ),
    ],
    max_epochs=50,
    logger=wandb_logger if WANDB_LOGGING else None,
    default_root_dir="checkpoints",
    val_check_interval=2000,
)
trainer.fit(model, train_loader, valid_loader)


print(checkpoint_callback.best_model_path)
model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
results = trainer.validate(model, valid_loader)[0]
wandb.log(results)
wandb.finish()


torch.save(model.unet.state_dict(), "weights/sat2rad-unet.pt")

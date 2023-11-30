import argparse
import itertools
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

from utils.data_utils import tensor_to_submission_file
from utils.w4c_dataloader import RainData
from utils.data_utils import load_config


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config_path", type=str, required=False, default='models/configurations/config.yaml',
                    help="path to config-yaml")
parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=None, 
                    help="specify gpu(s): 1 or 1 5 or 0 1 2 etc. (leave default for no GPU)")
parser.add_argument("-c", "--checkpoint", type=str, required=False, default='weights/weather-fusion-net.ckpt', 
                    help="model checkpoint path")
parser.add_argument("-s", "--split", type=str, choices=["test", "heldout"], required=False, default="test")
parser.add_argument("-ch", "--challenge", type=str, choices=["core", "transfer"], required=False, default="core")
parser.add_argument("-o", "--out_dir", type=str, required=False, default="submission", help="Destination folder")

options = parser.parse_args()

config = load_config(options.config_path)

from models.models import ThresholdWFN
model = ThresholdWFN.load_from_checkpoint(options.checkpoint, config=config)


trainer = pl.Trainer(accelerator="gpu" if options.gpus else None, devices=options.gpus)


years = config["dataset"]["years"]
regions = config["dataset"]["regions"]
transfer_years_regions = [
    (2019, "roxi_0008"),
    (2019, "roxi_0009"),
    (2019, "roxi_0010"),
    (2020, "roxi_0008"),
    (2020, "roxi_0009"),
    (2020, "roxi_0010"),
    (2021, "boxi_0015"),
    (2021, "boxi_0034"),
    (2021, "boxi_0076"),
    (2021, "roxi_0004"),
    (2021, "roxi_0005"),
    (2021, "roxi_0006"),
    (2021, "roxi_0007"),
    (2021, "roxi_0008"),
    (2021, "roxi_0009"),
    (2021, "roxi_0010"),
]

years_regions = itertools.product(years, regions) if options.challenge == "core" else transfer_years_regions

config["predict"]["submission_out_dir"] = options.out_dir
os.system(f"rm -r {options.out_dir}/*")

for year, region in years_regions:

    config["dataset"]["regions"] = [region]
    config["dataset"]["years"] = [year]
    config["predict"]["region_to_predict"] = region
    config["predict"]["year_to_predict"] = year

    test_ds = RainData(options.split, **config["dataset"])
    loader = DataLoader(test_ds, batch_size=5, num_workers=0, shuffle=False)

    preds = trainer.predict(model, loader)
    preds = torch.concat(preds)

    tensor_to_submission_file(preds, config["predict"])

print("Zipping submission")
os.system(f"cd {options.out_dir}; zip -r submission.zip *")

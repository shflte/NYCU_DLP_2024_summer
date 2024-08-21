import yaml
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from matplotlib import pyplot as plt
from utils import set_random_seed
from dataset import CLEVRDataset
from models.conditional_unet import ClassConditionedUnet


def train(config):
    # Dataset
    dataset = CLEVRDataset(
        dataset_dir='dataset/',
        train_json='train.json',
        objects_json='objects.json',
    )
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Model
    model = ClassConditionedUnet(num_classes=24)
    model = model.cuda()


if __name__ == "__main__":
    # Load arguments from config file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    set_random_seed(config["seed"])

    train(config)

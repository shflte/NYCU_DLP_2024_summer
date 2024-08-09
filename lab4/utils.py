import os
import random
import yaml
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torchvision import transforms
from math import log10
import matplotlib.pyplot as plt


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    else:
        config = {}
    return config


def Generate_PSNR(imgs1, imgs2, data_range=1.0):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2, reduction="none")
    mse = mse.view(mse.size(0), -1).mean(dim=1)
    psnr = 20 * torch.log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    return KLD


class kl_annealing:
    def __init__(self, args, current_epoch=0):
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio
        self.current_epoch = current_epoch
        self.beta = self.calculate_beta()

    def calculate_beta(self):
        if self.kl_anneal_type == "Cyclical":
            return self.frange_cycle_linear(
                self.current_epoch,
                n_iter=self.kl_anneal_cycle,
                ratio=self.kl_anneal_ratio,
            )
        elif self.kl_anneal_type == "Monotonic":
            return min(1.0, self.current_epoch / self.kl_anneal_cycle)
        else:
            return 1.0

    def update(self):
        self.current_epoch += 1
        self.beta = self.calculate_beta()

    def get_beta(self):
        return self.beta

    def frange_cycle_linear(
        self, current_epoch, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1
    ):
        L = np.linspace(start, stop, num=int(n_iter * ratio))
        beta = np.tile(np.concatenate([L, L[::-1]]), n_cycle)
        return beta[min(current_epoch, len(beta) - 1)]


class teacher_forcing:
    def __init__(self, args, current_epoch=0):
        self.tfr_init = args.tfr_init
        self.tfr_min = args.tfr_min
        self.tfr_sde = args.tfr_sde
        self.tfr_d_step = args.tfr_d_step
        self.current_epoch = current_epoch
        self.tfr = self.calculate_tfr()

    def update(self):
        self.current_epoch += 1
        self.tfr = self.calculate_tfr()

    def calculate_tfr(self):
        if self.current_epoch <= self.tfr_sde:
            tfr = self.tfr_init
        else:
            tfr = max(
                self.tfr_min,
                self.tfr_init - self.tfr_d_step * (self.current_epoch - self.tfr_sde),
            )
        return tfr

    def get_tfr(self):
        return self.tfr

    def adapt_teacher_forcing(self):
        return random.random() < self.tfr


def save_submission(pred_seq_list, save_root):
    pred_to_int = (np.rint(torch.cat(pred_seq_list).numpy() * 255)).astype(int)
    df = pd.DataFrame(pred_to_int)
    df.insert(0, "id", range(0, len(df)))
    df.to_csv(os.path.join(save_root, f"submission.csv"), header=True, index=False)


def make_gif(images_list, img_name):
    new_list = []
    for img in images_list:
        new_list.append(transforms.ToPILImage()(img))
    new_list[0].save(
        img_name,
        format="GIF",
        append_images=new_list,
        save_all=True,
        duration=20,
        loop=0,
    )

    import matplotlib.pyplot as plt


def show_loss(losses):
    """Plot training loss over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.savefig("training_loss.png")


def show_loss_kl_anneal(losses, kl_betas):
    """Plot training loss and KL annealing beta over epochs."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(losses, color=color, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("KL Beta", color=color)
    ax2.plot(kl_betas, color=color, linestyle="--", label="KL Beta")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title("Training Loss and KL Annealing Over Time")
    plt.savefig("training_loss_kl_anneal.png")


def show_loss_tfr(losses, tfrs):
    """Plot training loss and Teacher Forcing Ratio over epochs."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(losses, color=color, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Teacher Forcing Ratio", color=color)
    ax2.plot(tfrs, color=color, linestyle="--", label="TFR")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title("Training Loss and Teacher Forcing Ratio Over Time")
    plt.savefig("training_loss_tfr.png")


def show_psnr(psnr_list):
    """Plot PSNR for all frames across training."""
    plt.figure(figsize=(10, 5))
    plt.plot(psnr_list, label="PSNR per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Across All Frames During Training")
    plt.legend()
    plt.savefig("psnr_across_all_frames.png")

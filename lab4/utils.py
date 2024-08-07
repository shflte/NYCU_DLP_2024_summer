import os
import yaml
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torchvision import transforms
from math import log10


def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    else:
        config = {}
    return config


def Generate_PSNR(imgs1, imgs2, data_range=1.0):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2)  # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
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
        self.beta = 0.0

    def update(self):
        if self.kl_anneal_type == "Cyclical":
            self.beta = self.frange_cycle_linear(
                self.current_epoch,
                n_iter=self.kl_anneal_cycle,
                ratio=self.kl_anneal_ratio,
            )
        elif self.kl_anneal_type == "Monotonic":
            self.beta = min(1.0, self.current_epoch / self.kl_anneal_cycle)
        else:
            self.beta = 1.0  # Without KL annealing

    def get_beta(self):
        return self.beta

    def frange_cycle_linear(
        self, current_epoch, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1
    ):
        L = np.linspace(start, stop, num=int(n_iter * ratio))
        beta = []
        for _ in range(n_cycle):
            beta.extend(L)
            beta.extend(L[::-1])
        beta = np.array(beta)
        return beta[min(current_epoch, len(beta) - 1)]


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
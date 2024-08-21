import random
import os
import numpy as np
import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def show_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"loss.png")


def show_denoising_grid(images, result_dir, normalize=True):
    """
    Display a grid of 11 images in a single row using matplotlib.

    Args:
        images (Tensor): A batch of images, shape (image_nums, C, H, W)
        normalize (bool): If True, normalize the images to range [0, 1]
    """
    grid = vutils.make_grid(images, nrow=11, normalize=normalize, padding=2)

    plt.figure(figsize=(20, 5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Denoising Results")
    plt.axis("off")
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(os.path.join(result_dir, "denoising_results.png"))


def show_test_results_grid(images, result_dir, normalize=True):
    """
    Display a grid of images in 4 rows with 8 images per row using matplotlib.

    Args:
        images (Tensor): A batch of images, shape (image_nums, C, H, W)
        normalize (bool): If True, normalize the images to range [0, 1]
    """
    grid = vutils.make_grid(images, nrow=8, normalize=normalize, padding=2)

    plt.figure(figsize=(20, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Test Results")
    plt.axis("off")
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(os.path.join(result_dir, "test_results.png"))

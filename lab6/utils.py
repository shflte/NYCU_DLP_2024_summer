import random
import numpy as np
import torch
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
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"loss.png")


def show_accuracies(accuracies):
    epochs = [5 * (i + 1) for i in range(len(accuracies))]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy.png")

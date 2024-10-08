import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset as torchData
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader


class LoadTrainData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(
                    mean=[0.4816, 0.4324, 0.3845], std=[0.2602, 0.2518, 0.2537]
                ),  # Normalize the pixel values
            ]
        )
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        # self.folder = glob(os.path.join(root + '/*.png'))
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)

    @property
    def info(self):
        return f"\nNumber of Training Data: {int(len(self.folder) * self.partial)}"

    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))


class LoadTestData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4868, 0.4341, 0.3844], std=[0.2620, 0.2527, 0.2543]
                ),
            ]
        )
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)

    @property
    def info(self):
        return f"\nNumber of Testing Data: {int(len(self.folder) * self.partial)}"

    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))


class LoadMaskData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)

    @property
    def info(self):
        return f"\nNumber of Mask Data For Inpainting Task: {int(len(self.folder) * self.partial)}"

    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))


def mask_latent(z_indices, mask_token_id, mask_rate):
    masked_z_indices = z_indices.clone()

    num_tokens = z_indices.shape[1]
    num_masked = int(mask_rate * num_tokens)

    batch_size = z_indices.shape[0]
    for i in range(batch_size):
        masked_indices = torch.randperm(num_tokens)[:num_masked]
        masked_z_indices[i, masked_indices] = mask_token_id

    return masked_z_indices


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

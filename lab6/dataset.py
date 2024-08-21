import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def objects_to_one_hot(objects, objects_dict):
    one_hot_vector = np.zeros(len(objects_dict))
    for obj in objects:
        obj_idx = objects_dict[obj]
        one_hot_vector[obj_idx] = 1
    return one_hot_vector


class CLEVRDataset(Dataset):
    def __init__(self, dataset_dir, train_json, objects_json, transform=None):
        """
        Args:
            dataset_dir (str): Directory with all the images.
            train_json (str): Path to the train.json file.
            objects_json (str): Path to the objects.json file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.objects_dict = json.load(open(objects_json, "r"))
        self.data_info = json.load(open(train_json, "r"))
        self.vector_length = len(self.objects_dict)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # Image
        image_name = list(self.data_info.keys())[idx]
        image_path = os.path.join(self.dataset_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if not self.transform:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(72),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        image = self.transform(image)

        # Label
        objects = self.data_info[image_name]
        one_hot_vector = objects_to_one_hot(objects, self.objects_dict)
        label = torch.tensor(one_hot_vector, dtype=torch.float32)

        return image, label


class CLEVRDatasetEval(Dataset):
    def __init__(self, eval_json, objects_json):
        """
        Args:
            eval_json (str): Path to the eval.json file.
            objects_json (str): Path to the objects.json file.
        """
        self.objects_dict = json.load(open(objects_json, "r"))
        self.data_info = json.load(open(eval_json, "r"))
        self.vector_length = len(self.objects_dict)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        objects = self.data_info[idx]
        one_hot_vector = objects_to_one_hot(objects, self.objects_dict)
        label = torch.tensor(one_hot_vector, dtype=torch.float32)

        return label

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


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

        with json.load(open(objects_json, 'r')) as f:
            self.objects_dict = f

        with json.load(open(train_json, 'r')) as f:
            self.data_info = f

        self.vector_length = len(self.objects_dict)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # Image
        image_name = list(self.data_info.keys())[idx]
        image_path = os.path.join(self.dataset_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Label
        one_hot_vector = np.zeros(self.vector_length)
        objects = self.data_info[image_name]
        for obj in objects:
            obj_idx = self.objects_dict[obj]
            one_hot_vector[obj_idx] = 1
        label = torch.tensor(one_hot_vector, dtype=torch.float32)

        return image, label

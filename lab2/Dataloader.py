import torch
import numpy as np
import os


class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        # implement the getFeatures method
        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        features = []
        for file in os.listdir(filePath):
            if file.endswith(".npy"):
                data = np.load(os.path.join(filePath, file))
                features.append(data)
        if features:
            return np.concatenate(features, axis=0).astype(np.float32)
        else:
            return np.array([], dtype=np.float32)

    def _getLabels(self, filePath):
        # implement the getLabels method
        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        labels = []
        for file in os.listdir(filePath):
            if file.endswith(".npy"):
                data = np.load(os.path.join(filePath, file))
                labels.append(data)
        if labels:
            return np.concatenate(labels, axis=0).astype(np.int64)
        else:
            return np.array([]).astype(np.int64)

    def __init__(self, mode):
        # remember to change the file path according to different experiments
        assert mode in ["train", "test", "finetune"]
        if mode == "train":
            # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
            # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
            self.features = self._getFeatures(filePath="./dataset/LOSO_train/features/")
            self.labels = self._getLabels(filePath="./dataset/LOSO_train/labels/")
        if mode == "finetune":
            # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
            self.features = self._getFeatures(filePath="./dataset/FT/features/")
            self.labels = self._getLabels(filePath="./dataset/FT/labels/")
        if mode == "test":
            # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
            # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
            self.features = self._getFeatures(filePath="./dataset/LOSO_test/features/")
            self.labels = self._getLabels(filePath="./dataset/LOSO_test/labels/")

    def __len__(self):
        # implement the len method
        return len(self.features)

    def __getitem__(self, idx):
        # implement the getitem method
        feature = self.features[idx]
        label = self.labels[idx]
        feature = feature[np.newaxis, ...]
        return feature, label

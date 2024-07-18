import torch

class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        # implement the getFeatures method
        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        pass

    def _getLabels(self, filePath):
        # implement the getLabels method
        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        pass

    def __init__(self, mode):
        # remember to change the file path according to different experiments
        assert mode in ['train', 'test', 'finetune']
        if mode == 'train':
            # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
            # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
            self.features = self._getFeatures(filePath='./dataset/LOSO_train/features/')
            self.labels = self._getLabels(filePath='./dataset/LOSO_train/labels/')
        if mode == 'finetune':
            # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
            self.features = self._getFeatures(filePath='./dataset/FT/features/')
            self.labels = self._getLabels(filePath='./dataset/FT/labels/')
        if mode == 'test':
            # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
            # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
            self.features = self._getFeatures(filePath='./dataset/SD_test/features/')
            self.labels = self._getLabels(filePath='./dataset/SD_test/labels/')

    def __len__(self):
        # implement the len method
        pass

    def __getitem__(self, idx):
        # implement the getitem method
        pass
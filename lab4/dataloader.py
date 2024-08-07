import os
from glob import glob
from torch import stack
from torch.utils.data import Dataset as torchData
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader as imgloader
from torchvision import transforms


def get_key(fp):
    filename = fp.split("/")[-1]
    filename = filename.split(".")[0].replace("frame", "")
    return int(filename)


class Dataset_Dance(torchData):
    """
    Args:
        root (str)      : The path of your Dataset
        transform       : Transformation to your dataset
        mode (str)      : train, val, test
        partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """

    def __init__(self, root, transform, mode="train", video_len=7, partial=1.0):
        super().__init__()
        assert mode in ["train", "val", "test"], "There is no such mode !!!"
        self.mode = mode

        if mode == "train":
            self.img_folder = sorted(
                glob(os.path.join(root, "train/train_img/*.png")), key=get_key
            )
            self.prefix = "train"
        elif mode == "val":
            self.img_folder = sorted(
                glob(os.path.join(root, "val/val_img/*.png")), key=get_key
            )
            self.prefix = "val"
        elif mode == "test":
            num_folders = len(glob(os.path.join(root, "test/test_img/*")))
            self.img_folders = [
                sorted(
                    glob(os.path.join(root, f"test/test_img/{i}/*.png")), key=get_key
                )
                for i in range(num_folders)
            ]
            self.label_folders = [
                sorted(
                    glob(os.path.join(root, f"test/test_label/{i}/*.png")), key=get_key
                )
                for i in range(num_folders)
            ]
            self.prefix = "test"
        else:
            raise NotImplementedError

        self.transform = transform
        self.partial = partial
        self.video_len = video_len

    def __len__(self):
        if self.mode == "test":
            return len(self.img_folders)
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):
        if self.mode == "test":
            img_paths = self.img_folders[index]
            label_paths = self.label_folders[index]
        else:
            path = self.img_folder[index]
            img_paths = []
            label_paths = []
            for i in range(self.video_len):
                label_list = self.img_folder[(index * self.video_len) + i].split("/")
                label_list[-2] = self.prefix + "_label"

                img_name = self.img_folder[(index * self.video_len) + i]
                label_name = "/".join(label_list)

                img_paths.append(img_name)
                label_paths.append(label_name)

        imgs = [self.transform(imgloader(img_path)) for img_path in img_paths]
        labels = [self.transform(imgloader(label_path)) for label_path in label_paths]
        return stack(imgs), stack(labels)


def get_dataloader(
    root,
    frame_H,
    frame_W,
    mode,
    video_len,
    batch_size,
    num_workers,
    partial=1.0,
    shuffle=False,
    drop_last=True,
):
    transform = transforms.Compose(
        [
            transforms.Resize((frame_H, frame_W)),
            transforms.ToTensor(),
        ]
    )
    dataset = Dataset_Dance(
        root=root, transform=transform, mode=mode, video_len=video_len, partial=partial
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader

import torch
import os
import cv2
import pandas as pd
from torchvision.datasets.folder import default_loader


class CUBDataSet(torch.utils.data.Dataset):
    def __init__(self,root, train=True,transform = None):
        img_folder = os.path.join(root, "images")
        img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,
                                 names=['idx', 'label'])
        train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,
                                       names=['idx', 'train_flag'])
        data = pd.concat([img_paths, img_labels, train_test_split], axis=1)
        data = data[data['train_flag'] == train]
        data['label'] = data['label'] - 1
        imgs = data.reset_index(drop=True)
        if len(imgs) == 0:
            raise (RuntimeError("no csv file"))
        self.transform = transform
        self.root = img_folder
        self.imgs = imgs
        self.train = train
        self.targets = imgs['label']

           

    def __getitem__(self, index):
        item = self.imgs.iloc[index]
        file_path = item['path']
        target = item['label']
        img = default_loader(os.path.join(self.root, file_path))
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)
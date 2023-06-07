import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

def create_dataset(use_dataset, quicktest, phase):
    return 1

class CustomImageDataset(Dataset):
    """Class that creates a dataset from image directory ready to be sent to a Dataloader"""
    def __init__(self, img_dir, transform=None):
        self.img_labels = create_labels_frame(img_dir)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform((image, img_path))
        return image, label, img_path

def create_labels_frame(img_dir):
    df = pd.DataFrame()
    for filename in os.listdir(img_dir):

        # CHANGE TO RELEVANT USE CASE
        if filename[0] == "m":
            df = pd.concat([df, pd.DataFrame([[filename, 1]])])
        else:
            df = pd.concat([df, pd.DataFrame([[filename, 0]])])
    return df

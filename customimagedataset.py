import os

import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
from csv import writer, reader
from io import StringIO
import torchvision
from pathlib import Path


def create_dataset(use_datasets, quicktest, phase, transform):
    root = os.path.join(str(Path.home()), 'Documents', 'datasets')
    dataset_list = []
    input_transform = transform['input']
    target_transform = transform['target']
    if 'cifar10' in use_datasets:
        dataset_list.append(torchvision.datasets.CIFAR10(root=root, train=(
            'train' in phase), transform=input_transform, download=True))

    if 'vocseg' in use_datasets:
        dataset_list.append(torchvision.datasets.VOCSegmentation(
            root=root, image_set=phase, transform=input_transform,
            target_transform=target_transform, download=True))

    dataset = ConcatDataset(dataset_list)
    return dataset


class CustomImageDataset(Dataset):
    # Template for creating custom dataset
    def __init__(self, img_dir, phase, transform=None, quicktest=False):
        self.img_dir = os.path.join(img_dir, phase)
        self.img_labels = self.create_labels_frame()
        self.transform = transform
        self.quicktest = quicktest

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform((image, img_path))
        return image, label, img_path


def create_labels_frame(self):
    df = pd.DataFrame()
    output = StringIO()
    csv_writer = writer(output)
    for filename in os.listdir(self.img_dir):
        csv_writer.writerow([filename])

    output.seek(0)
    df = pd.read_csv(output, dtype=str)
    df = df.set_axis(["filename"], axis="columns",
                     copy=False)

    if self.quicktest:
        df = df[:int(0.1*len(df))]
    return df

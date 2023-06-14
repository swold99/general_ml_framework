import os

import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
from csv import writer, reader
from io import StringIO
import torchvision
from pathlib import Path


def create_dataset(use_datasets, quicktest, phase, transform):
    # Function to create a dataset based on the specified options
    root = os.path.join(os.path.dirname(os.getcwd()), 'datasets')
    if not os.path.exists(root):
        os.makedirs(root)
    dataset_list = []
    input_transform = transform['input']
    target_transform = transform['target']
    
    if 'cifar10' in use_datasets:
        # Add CIFAR10 dataset to the list
        dataset_list.append(torchvision.datasets.CIFAR10(root=root, train=(
            'train' in phase), transform=input_transform, download=True))

    if 'vocseg' in use_datasets:
        # Add VOC segmentation dataset to the list
        dataset_list.append(torchvision.datasets.VOCSegmentation(
            root=root, image_set=phase, transform=input_transform,
            target_transform=target_transform, download=True))
        
    if 'vocdet' in use_datasets:
        # Add VOC detection dataset to the list
        dataset_list.append(torchvision.datasets.VOCDetection(
            root=root, image_set=phase, transform=input_transform,
            target_transform=target_transform, download=True, year='2007'))
        
    # Add more datasets by uncommenting the code and implementing the respective dataset class
    # if 'your_dataset' in use_datasets:
    #     dataset_list.append(YourDataset('path/to/something', phase=phase, transform=transform))

    # Concatenate all the datasets into a single dataset
    dataset = ConcatDataset(dataset_list)
    return dataset


class CustomImageDataset(Dataset):
    # Template for creating a custom image dataset
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
        # Create a DataFrame with image filenames as labels
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
    
class YourDataset(CustomImageDataset):
    # Custom dataset class inherited from CustomImageDataset
    def something():
        pass

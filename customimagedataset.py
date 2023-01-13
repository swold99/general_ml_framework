import os
from bs4 import BeautifulSoup
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
from torchvision import transforms


class CustomImageDataset(Dataset):
    """Class that creates a dataset from image directory ready to be sent to a Dataloader"""

    def __init__(self, img_dir, splits, curr_split, quicktest=False, random_seed=42, transform=None):
        self.img_labels, self.img_names = create_labels_frame(img_dir,  splits, curr_split, quicktest, random_seed)
        self.img_dir = img_dir
        self.transform = transforms.Compose([transforms.ToPILImage(), transform])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name + '.jpeg')
        image = read_image(img_path)
        boxes = torch.tensor(self.img_labels[img_name])
        labels = torch.zeros(boxes.shape[0])
        label_dict = {'boxes':boxes, 'labels': labels}
        if self.transform:
            image = self.transform(image)
        return image, label_dict, img_path


def create_labels_frame(img_dir, splits, curr_split, quicktest, random_seed):
    label_dict = {}
    img_names = []
    folder = os.listdir(img_dir)
    N = int(len(folder)*(1-quicktest*0.9)/2)
    n_train = int(N * splits['train'])
    n_test = int(N*splits['test'])
    if curr_split == 'train':
        split_idx = torch.randperm(N, generator=torch.Generator().manual_seed(random_seed))[:n_train]
    elif curr_split == 'val':
        split_idx = torch.randperm(N, generator=torch.Generator().manual_seed(random_seed))[n_train:-n_test]
    else:
        split_idx = torch.randperm(N, generator=torch.Generator().manual_seed(random_seed))[-n_test:]

    split_idx = 2*split_idx + 1
    for i, filename in enumerate(folder):
        if i in split_idx:
            with open(os.path.join(img_dir, filename), "r") as f:
                data = f.read()
                Bs_data = BeautifulSoup(data, 'xml')
                coords = []
                for object in Bs_data.find_all('bndbox'):
                    coords.append([int(coord) for coord in object.text.strip().split('\n')]) #xmin, ymin, xmax, ymax
                img_name = filename.split('.')[0]
                img_names.append(img_name)
                label_dict[img_name] = coords

    return label_dict, img_names

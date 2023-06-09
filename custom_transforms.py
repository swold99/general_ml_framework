import math

import cv2
import torch
from torchvision import transforms
import numpy as np


class Standardize(object):
    """Standardize data
    """

    def __init__(self):
        pass

    def __call__(self, img):
        # Get mean and std for non-air pixels
        mean, std = torch.std_mean(img)

        # Standardize image
        img = (img - mean)/std

        return img


class MinMaxScaler(object):
    # Straightforward MinMaxScaler
    def __init__(self):
        pass

    def __call__(self, img):
        # Get mean and std for non-air pixels
        max_val = torch.max(img)
        min_val = torch.min(img)

        # Standardize image
        img = (img - min_val)/(max_val - min_val)

        return img

class Gray2RGB(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img.expand(3, -1, -1)

class ResizeGray(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = img.expand(1, 1, -1, -1)
        real_resize = transforms.Resize(self.size)
        return real_resize(img)

class HistogramEqualization(object):
    def __init__(self, transform_params):
        self.contrast_factor = transform_params['contrast_factor']
        self.clipLimit = transform_params['clip_limit']
        self.tileGridSize = transform_params['tile_grid_size']

    def __call__(self, img):
        # Performs histogram equalization. Do not know how to compose it into a single transform
        img = img.numpy()
        img = (img*(2**16-1)).astype('uint16')
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        cl1 = clahe.apply(img)
        img = cl1.astype('float32')
        img = torch.from_numpy(img)
        tsfrm2 = MinMaxScaler()
        img = tsfrm2(img)
        return img
    
class ToTensorWithoutScaling(object):
    def __call__(self, img):
        mask = torch.from_numpy(np.array(img)).unsqueeze(0)
        mask[mask==255] = 21
        return mask


def compose_transforms(transform_params):
    #Composes all wanted transforms into a single transform.
    trivial_augment = transform_params['trivial_augment']
    resize = transform_params['resize']
    input_tsfrm = transforms.Compose([transforms.ToTensor()])
    target_tsfrm = transforms.Compose([ToTensorWithoutScaling()])

    if resize is not None:
        input_tsfrm = transforms.Compose([input_tsfrm, transforms.Resize(resize)])
        target_tsfrm = transforms.Compose([target_tsfrm, transforms.Resize(resize)])
    if trivial_augment:
        input_tsfrm = transforms.Compose([input_tsfrm, transforms.TrivialAugmentWide()])

    tsfrm = {'input': input_tsfrm, 'target': target_tsfrm}
    return tsfrm
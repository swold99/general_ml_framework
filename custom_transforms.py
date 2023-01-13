import math

import cv2
import torch
from torchvision import transforms


class Standardize(object):
    """Standardize data
    """

    def __init__(self):
        pass

    def __call__(self, img):
        fname = img[1]
        img = img[0]

        # Get mean and std for non-air pixels
        mean, std = torch.std_mean(img)

        # Standardize image
        img = (img - mean)/std

        return (img, fname)


class MinMaxScaler(object):
    # Straightforward MinMaxScaler
    def __init__(self):
        pass

    def __call__(self, img):
        fname = img[1]
        img = img[0]
        # Get mean and std for non-air pixels
        max_val = torch.max(img)
        min_val = torch.min(img)

        # Standardize image
        img = (img - min_val)/(max_val - min_val)

        return (img, fname)

class IntensitySpread(object):
    def __init__(self, contrast_factor = 0.5):
        self.contrast_factor = contrast_factor

    def __call__(self, img):
        fname = img[1]
        img = img[0]
        # Increases contrast where intensity is lower. Contrast factor 0.5 gives pretty good contrast imo
        return (torch.log(1 + pow(img, self.contrast_factor)*(math.e-1)), fname)


class Gray2RGB(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = img[0]
        return img.expand(3, -1, -1)

class ResizeGray(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = img[0]
        img = img.expand(1, 1, -1, -1)
        real_resize = transforms.Resize(self.size)
        return real_resize(img)

class HistogramEqualization(object):
    def __init__(self, transform_params):
        self.contrast_factor = transform_params['contrast_factor']
        self.clipLimit = transform_params['clip_limit']
        self.tileGridSize = transform_params['tile_grid_size']

    def __call__(self, img):
        fname = img[1]
        img = img[0]
        # Performs histogram equalization. Do not know how to compose it into a single transform
        img = img.numpy()
        img = (img*(2**16-1)).astype('uint16')
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        cl1 = clahe.apply(img)
        img = cl1.astype('float32')
        img = torch.from_numpy(img)
        tsfrm2 = transforms.Compose([MinMaxScaler(), IntensitySpread(contrast_factor=self.contrast_factor)])
        img = tsfrm2((img, fname))
        return img


def compose_transforms(transform_params):
    #Composes all wanted transforms into a single transform.

    RGB = transform_params['RGB']
    standardization = transform_params['standardization']
    resize = transform_params['resize']
    contrast_factor = transform_params['contrast_factor']

    tsfrm = transforms.Compose([])

    return tsfrm








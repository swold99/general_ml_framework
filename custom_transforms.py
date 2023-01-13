import math
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import torch
from torch import Tensor
from torchvision import transforms

from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F


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
    def __init__(self, contrast_factor=0.5):
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
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit,
                                tileGridSize=self.tileGridSize)
        cl1 = clahe.apply(img)
        img = cl1.astype('float32')
        img = torch.from_numpy(img)
        tsfrm2 = transforms.Compose(
            [MinMaxScaler(), IntensitySpread(contrast_factor=self.contrast_factor)])
        img = tsfrm2((img, fname))
        return img

def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

class TrivialAugmentSwold(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(magnitudes[torch.randint(
                len(magnitudes), (1,), dtype=torch.long)].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


def compose_transforms(transform_params, train=False):
    # Composes all wanted transforms into a single transform.

    im_size = transform_params['im_size']
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # tsfrm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if train:
        tsfrm = transforms.Compose([TrivialAugmentSwold(), transforms.ToTensor(
        ), transforms.Normalize(mean, std), transforms.RandomErasing()])
        # tsfrm = transforms.Compose([
        #                             transforms.RandomHorizontalFlip(),
        #                             transforms.AutoAugment(),
        #                             transforms.RandomErasing()]) #transforms.TrivialAugmentWide()
    else:
        tsfrm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])
    tsfrm = transforms.Compose([tsfrm, transforms.Resize(im_size)])
    return tsfrm

import os
import sys
from typing import Any
import torch
from networks.create_model import create_segmentation_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_classes import Trainer
from metrics import SegmentationMeter
from utils import show_segmentation_imgs
from matplotlib import pyplot as plt

class SegmentationTrainer(Trainer):
    def __init__(self, savename, params, transform_params) -> None:
        super().__init__(savename, params, transform_params)
        self.colormap = plt.cm.get_cmap('jet', self.num_classes)

    def model_factory(self):
        # Create a segmentation model for training
        self.model = create_segmentation_model(self.network, self.device, self.num_classes)

    def task_metrics(self):
        # Return an instance of SegmentationMeter for tracking training metrics
        return SegmentationMeter(self.classes, self.savename)

    def process_model_out(self, outputs, device):
        # Process the model's output to obtain predicted segmentation labels
        return torch.argmax(outputs, axis=1).to(device)
    
    def preprocess_data(self, inputs, targets):
        # Preprocess inputs and targets for training
        return inputs.to(self.device), targets.to(self.device).squeeze(1).long()
    
    def forward_pass(self, input, targets):
        # Perform a forward pass of the input through the model
        output = self.model(input)
        if self.network == 'deeplab':
            return output['out']
        return output
    
    def show_images(self, inputs, targets, preds):
        # Show images with ground truth and predicted segmentation masks
        show_segmentation_imgs(inputs, targets, preds, self.colormap)

    

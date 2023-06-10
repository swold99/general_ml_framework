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
    def __init__(self, savename, data_folder, params, transform_params) -> None:
        super().__init__(savename, data_folder, params, transform_params)
        self.colormap = plt.cm.get_cmap('jet', self.num_classes)

    def model_factory(self):
        self.model = create_segmentation_model(self.network, self.device, self.num_classes)
    
    def init_metrics(self):
        super().init_metrics()
        self.train_f1_list = []
        self.val_f1_list = []

    def task_metrics(self):
        return SegmentationMeter(self.classes, self.savename)
    
    def print_metrics(self):
        accuracy = self.metrics['accuracy']
        precision = self.metrics['mAP']
        recall = self.metrics['mAR']
        f1 = self.metrics['mAF1']
        mIoU = self.metrics['mIoU']
        print(f'accuracy: {accuracy:.4f}, precision: {precision:.4f}, ' + 
              f'recall: {recall:.4f}, f1-score: {f1:.4f}, mIoU: {mIoU:.4f},')

    def process_model_out(self, outputs):
        return torch.argmax(outputs, axis=1)
    
    def preprocess_data(self, inputs, targets):
        return inputs.to(self.device), targets.to(self.device).squeeze(1).long()
    
    def show_images(self, inputs, targets, preds):
        show_segmentation_imgs(inputs, targets, preds, self.colormap)

    

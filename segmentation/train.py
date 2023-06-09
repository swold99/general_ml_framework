import os
import sys
import torch
from networks.create_model import create_segmentation_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_classes import Trainer
from metrics import SegmentationMeter

class SegmentationTrainer(Trainer):
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

    

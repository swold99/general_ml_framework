import os
import sys
import torch
from networks.create_model import create_detection_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_classes import Trainer
from metrics import DetectionMeter
from utils import show_detection_imgs

class DetectionTrainer(Trainer):
    def model_factory(self):
        self.model = create_detection_model(self.network, self.device, self.num_classes)
    
    def init_metrics(self):
        super().init_metrics()
        self.train_f1_list = []
        self.val_f1_list = []

    def task_metrics(self):
        return DetectionMeter(self.classes, self.savename)
    
    def print_metrics(self):
        accuracy = self.metrics['accuracy']
        print(f'accuracy: {accuracy:.4f}')

    def process_model_out(self, outputs):
        return torch.argmax(outputs, axis=1)
    
    def show_images(self, inputs, targets, preds):
        show_detection_imgs(inputs, targets, preds)


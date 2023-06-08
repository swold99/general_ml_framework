import os
import sys
import torch
from networks.create_model import create_classification_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_classses import Trainer
from metrics import ClassificationMeter

class ClassificationTrainer(Trainer):
    def model_factory(self):
        self.model = create_classification_model(self.network, self.device, self.num_classes)
    
    def init_metrics(self):
        super().init_metrics()
        self.train_f1_list = []
        self.val_f1_list = []

    def task_metrics(self):
        return ClassificationMeter()
    
    def print_metrics(self):
        accuracy = self.metrics['accuracy']
        print(f'accuracy: {accuracy:.4f}')

    def process_model_out(self, outputs):
        return torch.argmax(outputs, axis=1)

    
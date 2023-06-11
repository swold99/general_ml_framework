import os
import sys
import torch
from networks.create_model import create_classification_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_classes import Trainer
from metrics import ClassificationMeter
from utils import show_classification_imgs

class ClassificationTrainer(Trainer):
    def model_factory(self):
        self.model = create_classification_model(self.network, self.device, self.num_classes)
    
    def init_metrics(self):
        super().init_metrics()
        self.train_f1_list = []
        self.val_f1_list = []

    def task_metrics(self):
        return ClassificationMeter(self.classes, self.savename)
    

    def process_model_out(self, outputs, device):
        return torch.argmax(outputs, axis=1).to(device)
    
    def show_images(self, inputs, targets, preds):
        show_classification_imgs(inputs, targets, preds)


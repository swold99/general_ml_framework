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
        # Create a classification model using the network, device, and number of classes
        self.model = create_classification_model(self.network, self.device, self.num_classes)

    def task_metrics(self):
        # Create and return a ClassificationMeter for tracking classification metrics
        return ClassificationMeter(self.classes, self.savename)

    def process_model_out(self, outputs, device):
        # Process the model outputs by taking the argmax along the axis=1 (class dimension)
        # and move the results to the specified device
        return torch.argmax(outputs, axis=1).to(device)
    
    def show_images(self, inputs, targets, preds):
        # Show the images along with their corresponding targets and predicted labels
        show_classification_imgs(inputs, targets, preds, self.classes, save=True)


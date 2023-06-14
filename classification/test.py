from base_classes import Trainer, Evaluator
from metrics import ClassificationMeter
from utils import load_model_state, show_classification_imgs
from networks.create_model import create_classification_model
import torch


class ClassificationEvaluator(Evaluator):
    def task_metrics(self):
        # Create and return a ClassificationMeter for evaluating classification metrics
        return ClassificationMeter(self.classes, self.savename, eval=True)

    def model_factory(self):
        # Create a classification model using the network, device, and number of classes
        model = create_classification_model(self.network, self.device, self.num_classes)
        
        # Set the name of the saved model file
        name = self.savename + ".pth"
        
        # Load the model state from the saved file
        self.model = load_model_state(model, name)

    def process_model_out(self, outputs, device):
        # Process the model outputs by taking the argmax along the axis=1 (class dimension)
        # and move the results to the specified device
        return torch.argmax(outputs, axis=1).to(device)

    def show_images(self, inputs, targets, preds):
        # Show the images along with their corresponding targets and predicted labels
        show_classification_imgs(inputs, targets, preds)
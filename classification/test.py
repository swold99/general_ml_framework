from base_classes import Trainer, Evaluator
from metrics import ClassificationMeter
from utils import load_model_state, show_classification_imgs
from networks.create_model import create_classification_model
import torch
from train import ClassificationTrainer


class ClassificationEvaluator(Evaluator, ClassificationTrainer):

    def model_factory(self):
        # Create a classification model using the network, device, and number of classes
        model = create_classification_model(
            self.network, self.device, self.num_classes)

        # Set the name of the saved model file
        name = self.savename + ".pth"

        # Load the model state from the saved file
        self.model = load_model_state(model, name)

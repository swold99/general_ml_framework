from base_classes import Trainer, Evaluator
from metrics import ClassificationMeter
from utils import load_model_state, show_classification_imgs
from networks.create_model import create_classification_model
import torch


class ClassificationEvaluator(Evaluator):
    def task_metrics(self):
        return ClassificationMeter(self.classes, self.savename, eval=True)

    def model_factory(self):
        model = create_classification_model(self.network, self.device, self.num_classes)
        name = self.savename + ".pth"
        self.model = load_model_state(model, name)

    def process_model_out(self, outputs, device):
        return torch.argmax(outputs, axis=1).to(device)

    def show_images(self, inputs, targets, preds):
        show_classification_imgs(inputs, targets, preds)
from base_classes import Trainer, Evaluator
from metrics import ClassificationMeter
from utils import load_model_state, show_classification_imgs
from networks.create_model import create_classification_model


class ClassificationEvaluator(Evaluator):
    def task_metrics(self):
        return ClassificationMeter(self.classes, self.savename, eval=True)

    def model_factory(self):
        model = create_classification_model(self.network, self.device, self.num_classes)
        name = self.savename + ".pth"
        self.model = load_model_state(model, name)

    def show_images(self, inputs, targets, preds):
        show_classification_imgs(inputs, targets, preds)
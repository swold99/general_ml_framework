from base_classes import Trainer, Evaluator
from metrics import DetectionMeter
from utils import load_model_state, show_detection_imgs
from networks.create_model import create_detection_model


class DetectionEvaluator(Evaluator):
    def task_metrics(self):
        return DetectionMeter(self.classes, self.savename, eval=True)

    def model_factory(self):
        model = create_detection_model(self.network, self.device, self.num_classes)
        name = self.savename + ".pth"
        self.model = load_model_state(model, name)

    def show_images(self, inputs, targets, preds):
        show_detection_imgs(inputs, targets, preds)
from base_classes import Trainer, Evaluator
from metrics import SegmentationMeter
from utils import load_model_state
from networks.create_model import create_segmentation_model


class SegmentationEvaluator(Evaluator):
    def task_metrics(self):
        return SegmentationMeter(self.classes, self.savename, eval=True)

    def model_factory(self):
        model = create_segmentation_model(self.network, self.device, self.num_classes)
        name = self.savename + ".pth"
        self.model = load_model_state(model, name)

    def preprocess_data(self, inputs, targets):
        return inputs.to(self.device), targets.to(self.device).squeeze(1)
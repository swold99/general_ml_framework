from base_classes import Trainer, Evaluator
from metrics import SegmentationMeter
from utils import load_model_state, show_segmentation_imgs
from networks.create_model import create_segmentation_model
from matplotlib import pyplot as plt


class SegmentationEvaluator(Evaluator):
    def __init__(self, savename, data_folder, params, transform_params) -> None:
        super().__init__(savename, data_folder, params, transform_params)
        self.colormap = plt.cm.get_cmap('jet', self.num_classes)

    def task_metrics(self):
        return SegmentationMeter(self.classes, self.savename, eval=True)

    def model_factory(self):
        model = create_segmentation_model(self.network, self.device, self.num_classes)
        name = self.savename + ".pth"
        self.model = load_model_state(model, name)

    def preprocess_data(self, inputs, targets):
        return inputs.to(self.device), targets.to(self.device).squeeze(1)
    
    def show_images(self, inputs, targets, preds):
        show_segmentation_imgs(inputs, targets, preds, self.colormap)

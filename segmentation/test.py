from base_classes import Trainer, Evaluator
from metrics import SegmentationMeter
from utils import load_model_state, show_segmentation_imgs
from networks.create_model import create_segmentation_model
from matplotlib import pyplot as plt
from create_dataset import create_dataset
import torch


class SegmentationEvaluator(Evaluator):
    def __init__(self, savename, data_folder, params, transform_params) -> None:
        super().__init__(savename, data_folder, params, transform_params)
        self.colormap = plt.cm.get_cmap('jet', self.num_classes)

    def task_metrics(self):
        # Return an instance of SegmentationMeter for tracking evaluation metrics
        return SegmentationMeter(self.classes, self.savename, eval=True)

    def model_factory(self):
        # Create a segmentation model for evaluation
        model = create_segmentation_model(self.network, self.device, self.num_classes)
        
        # Load the model's state from a saved checkpoint
        name = self.savename + ".pth"
        self.model = load_model_state(model, name)

    def dataloader_factory(self, params):
        # Create a dataset for image evaluation
        image_dataset = create_dataset(params['use_datasets'], params['quicktest'],
                                       'val', self.tsfrm)
        
        # Create a data loader for batch processing
        self.dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=params['batch_size'], shuffle=False,
            num_workers=params['num_workers'])

    def preprocess_data(self, inputs, targets):
        # Preprocess inputs and targets for evaluation
        return inputs.to(self.device), targets.squeeze(1)
    
    def process_model_out(self, outputs, device):
        # Process the model's output to obtain predicted segmentation labels
        return torch.argmax(outputs, axis=1).to(device)

    def show_images(self, inputs, targets, preds):
        # Show images with ground truth and predicted segmentation masks
        show_segmentation_imgs(inputs, targets, preds, self.colormap)


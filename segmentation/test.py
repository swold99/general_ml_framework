from base_classes import Trainer, Evaluator
from metrics import SegmentationMeter
from utils import load_model_state, show_segmentation_imgs
from networks.create_model import create_segmentation_model
from matplotlib import pyplot as plt
from create_dataset import create_dataset
import torch
from train import SegmentationTrainer


class SegmentationEvaluator(Evaluator, SegmentationTrainer):
    def __init__(self, savename, params, transform_params) -> None:
        super().__init__(savename, params, transform_params)
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
    
    def forward_pass(self, input, targets):
        # Perform a forward pass of the input through the model
        output = self.model(input)
        if self.network == 'deeplab':
            return output['out']
        return output


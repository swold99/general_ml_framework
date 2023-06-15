import copy
from time import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
from custom_transforms import ToTensorWithoutScaling, TrivialAugmentSWOLD
import matplotlib.pyplot as plt

from create_dataset import create_dataset
from metrics import Meter
from utils import save_model, save_fig
import torch.backends.cudnn as cudnn
from losses.dice_loss import DiceLoss

class Trainer():
    def __init__(self, savename, params, transform_params) -> None:
        # Initialize the Trainer object with the given parameters
        self.savename = savename
        self.im_size = params['im_size']
        self.learning_rate = params['learning_rate']
        self.device = params['device']
        self.classes = params['classes']
        self.num_workers = params['num_workers']
        self.batch_size = params['batch_size']
        self.num_epochs = params['num_epochs']
        self.patience = params['patience']
        self.network = params['network']
        self.show_val_imgs = params['show_val_imgs']
        self.show_test_imgs = params['show_test_imgs']
        self.num_classes = params['num_classes']
        self.quicktest = params['quicktest']
        self.task = params['task']
        self.model_factory()
        
        # Move the model to the GPU if available
        if "cuda" in self.device:
            self.model.cuda()
            cudnn.benchmark = True
        
        self.optim_factory(params)
        self.loss_factory(params)
        
        # Determine the appropriate transformations based on the task
        space_augment = True if self.task == 'classification' else False
        self.transform = self.compose_transforms(transform_params=transform_params, label_is_space_invariant=space_augment)
        self.dataloader_factory(params)
        self.init_metrics()
        self.metrics = self.task_metrics()
        self.losses = Meter()

    def model_factory(self):
        # Placeholder method to be overridden by child classes
        pass

    def optim_factory(self, params):
        # Create the optimizer based on the specified parameters
        optim_type = params['optim_type']
        learning_rate = params['learning_rate']
        momentum = params['momentum']
        nesterov = params['nesterov']
        lr_gamma = params['lr_gamma']
        step_size = params['scheduler_step_size']
        schedule_type = params['schedule_type']
        weight_decay = params['weight_decay']
        
        if optim_type == "sgd":
            # Use Stochastic Gradient Descent optimizer
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=learning_rate,
                                             momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        elif optim_type == 'adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optim_type == 'adamw':
            self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise Exception('Optimizer not implemented')

        if schedule_type is not None:
            if "step" in schedule_type:
                # Use StepLR scheduler
                self.scheduler = lr_scheduler.StepLR(
                    self.optimizer, step_size=step_size, gamma=lr_gamma)
            else:
                raise Exception('Scheduler not implemented')
        return

    def loss_factory(self, params):
        # Create the loss function based on the specified parameters
        loss_fn = params['loss_fn']
        if "cross_entropy" in loss_fn:
            self.objective = nn.CrossEntropyLoss()
        
        if "dice" in loss_fn:
            self.objective = DiceLoss(mode='multiclass')

    def dataloader_factory(self, params):
        # Create the data loaders for the training and validation datasets
        self.phases = ['train', 'val']
        image_datasets = {phase: create_dataset(params['use_datasets'], params['quicktest'],
                                                phase, self.transform) for phase in self.phases}
        self.dataloaders = {phase: torch.utils.data.DataLoader(
            image_datasets[phase], batch_size=params['batch_size'], shuffle=True,
            num_workers=params['num_workers']) for phase in self.phases}

    def init_metrics(self):
        # Initialize various metrics and tracking variables
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_f1_list = []
        self.val_f1_list = []
        self.waiting_for_improvement = 0
        self.stop_training = False
        self.best_loss = torch.inf
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def compose_transforms(self, transform_params, label_is_space_invariant=True, phase='train'):
        # Composes all wanted transforms into a single transform.
        trivial_augment = transform_params['trivial_augment']
        resize = transform_params['resize']
        input_tsfrm = transforms.Compose([transforms.ToTensor()])
        target_tsfrm = transforms.Compose([ToTensorWithoutScaling()])

        if resize is not None:
            input_tsfrm = transforms.Compose(
                [input_tsfrm, transforms.Resize(resize, antialias=True)])
            target_tsfrm = transforms.Compose(
                [target_tsfrm, transforms.Resize(resize, antialias=True)])
        
        if trivial_augment and phase != 'test':
            input_tsfrm = transforms.Compose(
                [input_tsfrm, TrivialAugmentSWOLD(label_is_space_invariant=label_is_space_invariant)])

        tsfrm = {'input': input_tsfrm, 'target': target_tsfrm}
        return tsfrm
    def train_loop(self):
        # Training loop
        since = time()
        for epoch in range(self.num_epochs):
            if self.stop_training:
                return

            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print("Learning rate: ", self.optimizer.param_groups[0]['lr'])
            print('-' * 10)

            for phase in self.phases:
                if phase == "train":
                    self.train_one_epoch(epoch)
                else:
                    self.val_one_epoch(epoch)

        time_elapsed = time() - since
        print(f'Training took {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # Load best model weights
        self.model.load_state_dict(self.best_model_wts)

        # Save model to file
        save_model(self.model, self.savename)

        # Save loss and f1-curves
        self.save_train_val_plot(self.train_loss_list, self.val_loss_list, self.train_f1_list, self.val_f1_list,
                                 self.savename)

        return self.model, self.best_epoch, self.best_loss  # , best_acc
    
    def save_train_val_plot(self, train_loss_list, val_loss_list, train_f1_list, val_f1_list, savename):
        # Save training and validation loss and F1 curves
        save_fig(self.train_loss_list, self.val_loss_list, self.train_f1_list, self.val_f1_list, self.savename)


    def train_one_epoch(self, epoch):
        # Reset metrics and losses
        self.metrics.reset()
        self.losses.reset()

        # Create a progress bar for visualization
        prog_bar = tqdm(self.dataloaders['train'])
        self.model.train()

        for batch_idx, item in enumerate(prog_bar):
            inputs, targets = item

            # Preprocess data
            inputs, targets = self.preprocess_data(inputs, targets)

            # Zero the gradients
            self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(True):
                # Forward pass
                outputs = self.forward_pass(inputs, targets)
                preds = self.process_model_out(outputs, device=self.device)

                # Compute the loss and perform backpropagation
                loss = self.objective(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # Move tensors to CPU for evaluation
            inputs, targets, preds = inputs.cpu(), targets.cpu(), preds.cpu()

            # Show images if required
            if 0:
                self.show_images(inputs, targets, preds)

            # Update losses and metrics
            self.losses.update(loss.item())
            self.metrics.update(preds, targets)

        # Adjust the learning rate
        self.scheduler.step()

        # Update progress bar description with running loss
        desc = f'Running loss: {round(self.losses.avg, 5)}'
        prog_bar.set_description(desc)

        # Calculate epoch loss and append to the train loss list
        epoch_loss = self.losses.avg
        self.train_loss_list.append(epoch_loss)
        print(f'train Loss: {epoch_loss:.4f}')

        # Compute final metrics and append F1 score to the train F1 list
        self.metric_dict = self.metrics.get_final_metrics()
        self.train_f1_list.append(self.metric_dict['mAF1'])


    def val_one_epoch(self, epoch):
        # Reset metrics and losses
        self.metrics.reset()
        self.losses.reset()

        # Create a progress bar for visualization
        prog_bar = tqdm(self.dataloaders['val'])
        self.model.eval()

        for batch_idx, item in enumerate(prog_bar):
            inputs, targets = item

            # Preprocess data
            inputs, targets = self.preprocess_data(inputs, targets)

            with torch.set_grad_enabled(False):
                # Forward pass
                outputs = self.forward_pass(inputs, targets)
                preds = self.process_model_out(outputs, device=self.device)

                # Compute the loss
                loss = self.objective(outputs, targets)

            # Move tensors to CPU for evaluation
            inputs, targets, preds = inputs.cpu(), targets.cpu(), preds.cpu()

            # Show images if required
            if self.show_val_imgs:
                self.show_images(inputs, targets, preds)

            # Update losses and metrics
            self.losses.update(loss.item())
            self.metrics.update(preds, targets)

        # Update progress bar description with running loss
        desc = f'Running loss: {round(self.losses.avg, 5)}'
        prog_bar.set_description(desc)

        # Calculate epoch loss and append to the validation loss list
        epoch_loss = self.losses.avg
        self.val_loss_list.append(epoch_loss)
        print(f'validation Loss: {epoch_loss:.4f}')

        # Compute final metrics and append F1 score to the validation F1 list
        metric_dict = self.metrics.get_final_metrics()
        self.val_f1_list.append(self.metric_dict['mAF1'])

        # Check if the current loss is the best so far
        if epoch_loss < self.best_loss:
            self.waiting_for_improvement = 0
            self.best_epoch = epoch
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            print("Best model yet")
        else:
            self.waiting_for_improvement += 1

        # Check if training should be stopped based on lack of improvement
        if self.waiting_for_improvement >= self.patience:
            print(f'Validation loss has not improved in {self.patience} epochs. Stopping training.')
            self.stop_training = True

    def forward_pass(self, input, targets):
        # Perform a forward pass of the input through the model
        return self.model(input)

    def preprocess_data(self, inputs, targets):
        # Move the inputs and targets tensors to the device (GPU)
        return inputs.to(self.device), targets.to(self.device)

    def process_model_out(self, outputs, device):
        # Move the model outputs to the specified device (GPU)
        return outputs.to(device)

    def show_images(self, inputs, targets, preds):
        # This method is responsible for showing the images (not implemented)
        pass

    def task_metrics(self):
        # This method is responsible for defining and returning the task-specific metrics (not implemented)
        pass

    def print_metrics(self):
        # This method is responsible for printing the computed metrics (not implemented)
        pass



class Evaluator(Trainer):
    def __init__(self, savename, params, transform_params) -> None:
        self.savename = savename
        self.im_size = params['im_size']
        self.device = params['device']
        self.num_workers = params['num_workers']
        self.batch_size = params['batch_size']
        self.network = params['network']
        self.num_classes = params['num_classes']
        self.task = params['task']
        self.classes = params['classes']
        self.show_val_imgs = params['show_val_imgs']
        self.show_test_imgs = params['show_test_imgs']
        self.model_factory()
        self.tsfrm = self.compose_transforms(transform_params=transform_params, phase='test')
        self.dataloader_factory(params)
        self.metrics = self.task_metrics()
        self.losses = Meter()

    def dataloader_factory(self, params):
        # Create a test dataset and dataloader
        image_dataset = create_dataset(params['use_datasets'], params['quicktest'],
                                       'test', self.tsfrm)
        self.dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=params['batch_size'], shuffle=False,
            num_workers=params['num_workers'])

    def preprocess_data(self, inputs, targets):
        # Move the inputs tensor to the device (GPU)
        return inputs.to(self.device), targets
    
    def forward_pass(self, input, targets):
        # Perform a forward pass of the input through the model
        return self.model(input)

    def test_loop(self):
        # Set the model to evaluation mode
        self.model.eval()
        # Reset the metrics and losses
        self.metrics.reset()
        self.losses.reset()
        prog_bar = tqdm(self.dataloader)
        times = []

        for batch_idx, item in enumerate(prog_bar):
            inputs, targets = item

            inputs, targets = self.preprocess_data(inputs, targets)

            t1 = time()
            # Perform a forward pass and obtain model outputs
            outputs = self.forward_pass(inputs, targets)
            t2 = time()
            times.append(t2-t1)
            preds = self.process_model_out(outputs, device='cpu')
            if self.show_test_imgs:
                self.show_images(inputs.cpu(), targets, preds)

            # Update metrics with the predicted outputs and ground truth targets
            self.metrics.update(preds, targets)

        metric_dict = self.metrics.get_final_metrics()
        print("Average inference time: ", torch.mean(
            torch.tensor(times)/self.batch_size).item(), "s")

        return metric_dict


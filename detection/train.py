from create_dataset import create_dataset
import copy
from tqdm import tqdm
from utils import show_detection_imgs, move_to, collate_fn, save_fig
from metrics import DetectionMeter
from base_classes import Trainer
from custom_transforms import TrivialAugmentSWOLD, Identity
from torchvision import transforms
import os
import sys
import torch
from networks.create_model import create_detection_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DetectionTrainer(Trainer):

    def compose_transforms(self, transform_params, label_is_space_invariant=True):
        # Composes all wanted transforms into a single transform.
        trivial_augment = transform_params['trivial_augment']
        resize = transform_params['resize']
        input_tsfrm = transforms.Compose([transforms.ToTensor()])
        target_tsfrm = Identity()

        if resize is not None:
            input_tsfrm = transforms.Compose(
                [input_tsfrm, transforms.Resize(resize, antialias=True)])
        if trivial_augment:
            input_tsfrm = transforms.Compose(
                [input_tsfrm, TrivialAugmentSWOLD(label_is_space_invariant=label_is_space_invariant)])

        tsfrm = {'input': input_tsfrm, 'target': target_tsfrm}
        return tsfrm

    def dataloader_factory(self, params):
        self.phases = ['train', 'val']
        image_datasets = {phase: create_dataset(params['use_datasets'], params['quicktest'],
                                                phase, self.transform) for phase in self.phases}
        self.dataloaders = {phase: torch.utils.data.DataLoader(
            image_datasets[phase], batch_size=params['batch_size'], shuffle=False,
            num_workers=params['num_workers'], collate_fn=collate_fn) for phase in self.phases}

    def preprocess_data(self, inputs, targets):
        tmp = []
        for element in targets:
            annotation = element['annotation']
            objects = annotation['object']
            scaling = [float(annotation['size']['height'])/self.im_size[1],
                       float(annotation['size']['width'])/self.im_size[0]]
            N = len(objects)
            boxes = torch.empty((N, 4), dtype=torch.float)
            labels = torch.empty((N), dtype=torch.int64)
            for i, obj in enumerate(objects):
                bndbox = obj['bndbox']
                box = [bndbox['xmin'], bndbox['ymin'],
                       bndbox['xmax'], bndbox['ymax']]
                # Calculate scaled box coordinates
                boxes[i, :] = torch.FloatTensor(
                    [float(val)/scaling[1-(j % 2)] for j, val in enumerate(box)])
                # Set the label index
                labels[i] = self.classes.index(obj['name'])
            tmp.append({'boxes': boxes, 'labels': labels})

        # Return preprocessed inputs and targets
        return move_to(list(inputs), self.device), move_to(tmp, self.device)

    def train_one_epoch(self, epoch):
        # Reset metrics and losses for the epoch
        self.metrics.reset()
        self.losses.reset()
        prog_bar = tqdm(self.dataloaders['train'])
        self.model.train()  # Set the model in training mode

        for batch_idx, item in enumerate(prog_bar):
            inputs, targets = item

            # Preprocess inputs and targets
            inputs, targets = self.preprocess_data(inputs, targets)

            # Clear gradients of the optimizer
            self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(True):
                # Perform forward pass through the model
                outputs = self.forward_pass(inputs, targets)
                # Calculate the loss
                loss = self.process_model_out(outputs, device=self.device)

                # Backpropagation and optimizer step
                loss.backward()
                self.optimizer.step()

            # Update losses with the current batch loss
            self.losses.update(loss.item())

            # Update progress bar description with current loss
            desc = (f'Running loss: {round(self.losses.avg,5)}')
            prog_bar.set_description(desc)

        # Step the scheduler to update the learning rate
        self.scheduler.step()

        epoch_loss = self.losses.avg
        self.train_loss_list.append(epoch_loss)

        # Print the epoch loss
        print(f'train Loss: {epoch_loss:.4f}')

    def val_one_epoch(self, epoch):
        # Reset metrics and losses for the epoch
        self.metrics.reset()
        self.losses.reset()
        prog_bar = tqdm(self.dataloaders['val'])

        for batch_idx, item in enumerate(prog_bar):
            inputs, targets = item

            # Preprocess inputs and targets
            inputs, targets = self.preprocess_data(inputs, targets)

            with torch.no_grad():
                # Perform forward pass through the model without gradient computation
                outputs = self.forward_pass(inputs, targets)
                # Calculate the loss
                loss = self.process_model_out(outputs, device=self.device)

            # Update losses with the current batch loss
            self.losses.update(loss.item())

        # Update progress bar description with current loss
        desc = (f'Running loss: {round(self.losses.avg,5)}')
        prog_bar.set_description(desc)

        epoch_loss = self.losses.avg
        self.val_loss_list.append(epoch_loss)

        # Print the validation loss
        print(f'validation Loss: {epoch_loss:.4f}')

        if epoch_loss < self.best_loss:
            # If the current loss is the best so far, update the best model and its information
            self.waiting_for_improvement = 0
            self.best_epoch = epoch
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            print("Best model yet")
        else:
            # Increment the waiting counter for improvement
            self.waiting_for_improvement += 1

        if self.waiting_for_improvement >= self.patience:
            # If the waiting counter exceeds the patience, stop training
            print(f'Validation loss has not improved in {self.patience} epochs. Stopping training.')
            self.stop_training = True

    def model_factory(self):
        # Create the detection model based on the network, device, and number of classes
        self.model = create_detection_model(self.network, self.device, self.num_classes)

    def task_metrics(self):
        # Return None since the DetectionMeter is not used
        return None

    def process_model_out(self, outputs, device):
        # Calculate the sum of losses from the outputs and move them to the specified device
        losses = sum(loss for loss in outputs.values()).to(device)
        return losses

    def forward_pass(self, input, targets):
        # Perform the forward pass through the model with the inputs and targets
        return self.model(input, targets)

    def show_images(self, inputs, targets, preds):
        # Show the images with detection results
        show_detection_imgs(inputs, targets, preds)

    def save_train_val_plot(self, train_loss_list, val_loss_list, train_f1_list, val_f1_list, savename):
        # Save the train/validation loss plot using the provided lists and savename
        save_fig(train_loss_list, val_loss_list, exp_name=savename)


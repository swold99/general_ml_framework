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
            image_datasets[phase], batch_size=params['batch_size'], shuffle=True,
            num_workers=params['num_workers'], collate_fn=collate_fn) for phase in self.phases}

    def preprocess_data(self, inputs, targets):
        tmp = []
        for element in targets:
            annotation = element['annotation']
            objects = annotation['object']
            scaling = [float(annotation['size']['height'])/self.im_size[0],
                       float(annotation['size']['width'])/self.im_size[1]]
            N = len(objects)
            boxes = torch.empty((N, 4), dtype=torch.float)
            labels = torch.empty((N), dtype=torch.int64)
            for i, object in enumerate(objects):
                bndbox = object['bndbox']
                box = [bndbox['xmin'], bndbox['ymin'],
                       bndbox['xmax'], bndbox['ymax']]
                boxes[i, :] = torch.FloatTensor(
                    [float(val)/scaling[1-(j % 2)] for j, val in enumerate(box)])
                labels[i] = self.classes.index(object['name'])
            tmp.append({'boxes': boxes, 'labels': labels})

        return move_to(list(inputs), self.device), move_to(tmp, self.device)

    def train_one_epoch(self, epoch):
        #self.metrics.reset()
        self.losses.reset()
        prog_bar = tqdm(self.dataloaders['train'])
        self.model.train()

        for batch_idx, item in enumerate(prog_bar):
            inputs, targets = item

            inputs, targets = self.preprocess_data(inputs, targets)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(True):
                outputs = self.forward_pass(inputs, targets)
                loss = self.process_model_out(outputs, device=self.device)

                loss.backward()
                self.optimizer.step()

            self.losses.update(loss.item())

        self.scheduler.step()
        desc = (f'Running loss: {round(self.losses.avg,5)}')
        prog_bar.set_description(desc)
        epoch_loss = self.losses.avg
        self.train_loss_list.append(epoch_loss)
        print(f'train Loss: {epoch_loss:.4f}')

    def val_one_epoch(self, epoch):
        #self.metrics.reset()
        self.losses.reset()
        prog_bar = tqdm(self.dataloaders['val'])

        for batch_idx, item in enumerate(prog_bar):
            inputs, targets = item

            inputs, targets = self.preprocess_data(inputs, targets)

            with torch.set_grad_enabled(False):
                outputs = self.forward_pass(inputs, targets)
                loss = self.process_model_out(outputs, device=self.device)

            self.losses.update(loss.item())

        desc = (f'Running loss: {round(self.losses.avg,5)}')
        prog_bar.set_description(desc)
        epoch_loss = self.losses.avg
        self.val_loss_list.append(epoch_loss)
        print(f'validation Loss: {epoch_loss:.4f}')

        if epoch_loss < self.best_loss:
            self.waiting_for_improvement = 0
            self.best_epoch = epoch
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            print("Best model yet")
        else:
            self.waiting_for_improvement += 1

        if self.waiting_for_improvement >= self.patience:
            print(f'Validation loss has not improved in {self.patience}'
                  ' epochs. Stopping training.')
            self.stop_training = True

    def model_factory(self):
        self.model = create_detection_model(
            self.network, self.device, self.num_classes)

    def task_metrics(self):
        return None#DetectionMeter(self.classes, self.savename)

    def process_model_out(self, outputs, device):
        losses = sum(loss for loss in outputs.values()).to(device)
        return losses

    def forward_pass(self, input, targets):
        return self.model(input, targets)

    def show_images(self, inputs, targets, preds):
        show_detection_imgs(inputs, targets, preds)

    def save_train_val_plot(self, train_loss_list, val_loss_list, train_f1_list, val_f1_list,
                            savename):
        save_fig(train_loss_list, val_loss_list, exp_name=savename)

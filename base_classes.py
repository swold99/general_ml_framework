import copy
from time import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt

from custom_transforms import compose_transforms
from create_dataset import create_dataset
from networks import create_model
from metrics import Meter
from utils import save_model, save_fig


class Trainer():
    def __init__(self, savename, data_folder, params, transform_params) -> None:
        self.savename = savename
        self.data_folder = data_folder
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
        self.optim_factory(params)
        self.loss_factory(params)
        space_augment = True if self.task == 'classification' else False
        self.transform = compose_transforms(transform_params=transform_params, label_is_space_invariant=space_augment)
        self.dataloader_factory(params)
        self.init_metrics()
        self.metrics = self.task_metrics()
        self.losses = Meter()

    def model_factory(self):
        pass

    def optim_factory(self, params):
        optim_type = params['optim_type']
        learning_rate = params['learning_rate']
        momentum = params['momentum']
        nesterov = params['nesterov']
        lr_gamma = params['lr_gamma']
        step_size = params['scheduler_step_size']
        schedule_type = params['schedule_type']
        if optim_type == "sgd":
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=learning_rate,
                                             momentum=momentum, nesterov=nesterov)
        else:
            raise Exception('Optimizer not implemented')

        if schedule_type is not None:
            if "step" in schedule_type:
                self.scheduler = lr_scheduler.StepLR(
                    self.optimizer, step_size=step_size, gamma=lr_gamma)
            else:
                raise Exception('Scheduler not implemented')
        return

    def loss_factory(self, params):
        loss_fn = params['loss_fn']
        if "cross_entropy" in loss_fn:
            self.objective = nn.CrossEntropyLoss()

    def dataloader_factory(self, params):
        self.phases = ['train', 'val']
        image_datasets = {phase: create_dataset(params['use_datasets'], params['quicktest'],
                                                phase, self.transform) for phase in self.phases}
        self.dataloaders = {phase: torch.utils.data.DataLoader(
            image_datasets[phase], batch_size=params['batch_size'], shuffle=True,
            num_workers=params['num_workers']) for phase in self.phases}

    def init_metrics(self):
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_f1_list = []
        self.val_f1_list = []
        self.waiting_for_improvement = 0
        self.stop_training = False
        self.best_loss = torch.inf
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def train_loop(self):
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
        print(
            f'Training took {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # load best model weights
        self.model.load_state_dict(self.best_model_wts)

        # Save model to file
        save_model(self.model, self.savename)

        # Save loss and f1-curves
        save_fig(self.train_loss_list, self.val_loss_list, f1_train=self.train_f1_list, f1_val=self.val_f1_list,
                 exp_name=self.savename)

        return self.model, self.best_epoch, self.best_loss  # , best_acc

    def train_one_epoch(self, epoch):
        self.metrics.reset()
        self.losses.reset()
        prog_bar = tqdm(self.dataloaders['train'])
        self.model.train()

        for batch_idx, item in enumerate(prog_bar):
            inputs, targets = item

            inputs, targets = self.preprocess_data(inputs, targets)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                preds = self.process_model_out(outputs, device=self.device)

                loss = self.objective(outputs, targets)
                loss.backward()
                self.optimizer.step()

            inputs, targets, preds = inputs.cpu(), targets.cpu(), preds.cpu()

            if 0: self.show_images(inputs, targets, preds)

            self.losses.update(loss.item())
            self.metrics.update(preds, targets)

        self.scheduler.step()
        desc = (f'Running loss: {round(self.losses.avg,5)}')
        prog_bar.set_description(desc)
        epoch_loss = self.losses.avg
        self.train_loss_list.append(epoch_loss)
        print(f'train Loss: {epoch_loss:.4f}')
        self.metric_dict = self.metrics.get_final_metrics()
        self.train_f1_list.append(self.metric_dict['mAF1'])

    def val_one_epoch(self, epoch):
        self.metrics = self.task_metrics()
        self.losses = Meter()
        prog_bar = tqdm(self.dataloaders['val'])
        self.model.eval()

        for batch_idx, item in enumerate(prog_bar):
            inputs, targets = item

            inputs, targets = self.preprocess_data(inputs, targets)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                preds = self.process_model_out(outputs, device=self.device)

                loss = self.objective(outputs, targets)

            inputs, targets, preds = inputs.cpu(), targets.cpu(), preds.cpu()

            if self.show_val_imgs:
                self.show_images(inputs, targets, preds)

            self.losses.update(loss.item())
            self.metrics.update(preds, targets)

        desc = (f'Running loss: {round(self.losses.avg,5)}')
        prog_bar.set_description(desc)
        epoch_loss = self.losses.avg
        self.val_loss_list.append(epoch_loss)
        print(f'validation Loss: {epoch_loss:.4f}')
        metric_dict = self.metrics.get_final_metrics()
        self.val_f1_list.append(self.metric_dict['mAF1'])

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

    def preprocess_data(self, inputs, targets):
        return inputs.to(self.device), targets.to(self.device)

    def process_model_out(self, outputs, device):
        return outputs.to(device)

    def show_images(self, inputs, targets):
        pass

    def task_metrics(self):
        pass

    def print_metrics(self):
        pass


class Evaluator(Trainer):
    def __init__(self, savename, data_folder, params, transform_params) -> None:
        self.savename = savename
        self.data_folder = data_folder
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
        self.tsfrm = compose_transforms(transform_params=transform_params)
        self.dataloader_factory(params, transform_params)
        self.metrics = self.task_metrics()
        self.losses = Meter()

    def dataloader_factory(self, params, transform_params):
        image_dataset = create_dataset(params['use_datasets'], params['quicktest'],
                                       'test', self.tsfrm)
        self.dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=params['batch_size'], shuffle=True,
            num_workers=params['num_workers'])

    def preprocess_data(self, inputs, targets):
        return inputs.to(self.device), targets

    def test_loop(self):
        self.model.eval()
        self.metrics.reset()
        self.losses.reset()
        prog_bar = tqdm(self.dataloader)
        times = []

        for batch_idx, item in enumerate(prog_bar):
            inputs, targets = item

            inputs, targets = self.preprocess_data(inputs, targets)

            t1 = time()
            outputs = self.model(inputs)
            t2 = time()
            times.append(t2-t1)
            preds = self.process_model_out(outputs, device='cpu')
            if self.show_test_imgs:
                self.show_images(preds, targets)

            self.metrics.update(preds, targets)

        metric_dict = self.metrics.get_final_metrics()
        print("Average inference time: ", torch.mean(
            torch.tensor(times)/self.batch_size).item(), "s")

        return metric_dict

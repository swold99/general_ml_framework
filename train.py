import os
import sys
from time import time

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from model.create_model import create_model
from utils import save_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import time

import numpy as np
import torch
from tqdm import tqdm

from custom_transforms import compose_transforms
from customimagedataset import CustomImageDataset
from utils import move_to, collate_fn


def train(file_name, data_folders, params, transform_params, nr_ensemble_models=1, load_model=False):

    # Extract params
    learning_rate = params['learning_rate']
    classes = params['classes']
    use_cuda = params['use_cuda']
    device = params['device']
    momentum = params['momentum']
    num_workers = params['num_workers']

    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    splits = params['splits']
    quicktest = params['quicktest']

    data_folder = data_folders['data']

    num_classes = len(classes)

    ####################################
    ## Init the network and optimizer ##
    ####################################

    model = create_model(model="faster-rcnn", use_cuda=use_cuda, num_classes=num_classes)

    # The objective (loss) function
    objective = nn.CrossEntropyLoss()

    best_acc = 0
    best_epoch = 0
    best_loss = 0

    #######################
    ## Train the network ##
    #######################
    train_start=time.time()

    # The optimizer used for training the model
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0.001)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_tsfrm = compose_transforms(transform_params=transform_params, train=True)
    val_tsfrm = compose_transforms(transform_params=transform_params, train=False)

    phases = ['train', 'val']

    image_datasets = {'train' : CustomImageDataset(data_folder, splits, "train", quicktest=quicktest, transform=train_tsfrm),
                    'val' : CustomImageDataset(data_folder, splits, "val", quicktest=quicktest, transform=val_tsfrm)
                    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
                    for x in ['train', 'val']}

    dataset_sizes = {phase: len(image_datasets[phase]) for phase in phases}
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.inf
    best_epoch = 0

    loss_hist = {"train": Averager(), "val": Averager()}
    loss_list = {"train": [], "val": []}

    save_best_model = SaveBestModel()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        start = time.time()
        # Each epoch has a training and validation phase
        for phase in phases:
            print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            prog_bar = tqdm(dataloaders[phase], total=len(dataloaders[phase]))

            # Iterate over data.
            for batch_idx, (inputs, targets, fnames) in enumerate(prog_bar):
                inputs = torch.stack(inputs).to(device)
                targets = list(targets)
                targets = move_to(targets, device)

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    loss_dict = model(inputs, targets)

                    losses = sum(loss for loss in loss_dict.values())
                    loss_value = losses.item()

                    loss_list[phase].append(loss_value)
                    loss_hist[phase].send(loss_value)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        losses.backward()
                        optimizer.step()

                prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

            scheduler.step()

        print(f"Epoch #{epoch+1} train loss: {loss_hist['train'].value:.3f}")
        print(f"Epoch #{epoch+1} validation loss: {loss_hist['val'].value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        save_best_model(loss_hist["val"].value, epoch, model, optimizer)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    save_model(model, file_name + ".pth")

    return model, best_acc, best_epoch, best_loss


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

    def __call__(
        self, current_valid_loss,
        epoch, model, optimizer
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'outputs/best_model.pth')

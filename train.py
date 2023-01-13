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
from utils import move_to


def train(file_name, data_folders, params, transform_params, nr_ensemble_models=1, load_model=False):

    # Extract params
    learning_rate = params['learning_rate']
    classes = params['classes']
    use_cuda = params['use_cuda']
    device = params['device']
    momentum = params['momentum']
    num_workers = params['num_workers']
    step_size = params['scheduler_step_size']
    lr_gamma = params['lr_gamma']

    batch_size = params['batch_size']
    num_epochs = params['num_epochs']

    data_path_train = data_folders['train']
    data_path_val = data_folders['val']

    num_classes = len(classes)

    ####################################
    ## Init the network and optimizer ##
    ####################################

    model = create_model(use_cuda, classes)

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

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_gamma)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    tsfrm = compose_transforms(transform_params=transform_params)

    phases = ['train', 'val']

    image_datasets = {'train' : CustomImageDataset(data_path_train, transform=tsfrm),
                    'val' : CustomImageDataset(data_path_val, transform=tsfrm)
                    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)
                    for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[phase]) for phase in phases}
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.inf
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (inputs, labels, fnames) in enumerate(tqdm(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = objective(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss: # epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    save_model(model, file_name + ".pth")

    return model, best_acc, best_epoch, best_loss
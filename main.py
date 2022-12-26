import os
from pprint import pprint
from test import test

import torch
from tqdm import tqdm

from inference import inference
from model.create_model import create_model
from train import train
from utils import load_model_state


def main():

    # Add argparser?

    # Specify folders for data
    data_folders = {}
    #data_folders['train'] =
    #data_folders['val'] =
    #data_folders['test'] =

    experiment_name= "exp1"

    default_im_size = (512, 1024)
    downsample_factor = 1
    im_size = tuple([int(x/downsample_factor) for x in default_im_size])

    params = get_default_params()
    params['im_size'] = im_size

    transform_params = get_default_transform_params(im_size)

    if 0:
        train_and_test(experiment_name, data_folders, params, transform_params)

    if 0:
        test(experiment_name, data_folders, params, transform_params)

    if 0:
        inference(experiment_name, data_folders, params, transform_params)


def train_and_test(experiment_name, data_folders, params, transform_params):

    trained_model, metrics = train(experiment_name, data_folders, params, transform_params)
    pprint(metrics)
    metrics = test(trained_model, data_folders, params, transform_params)
    pprint(metrics)
    return

def only_test(model_name, data_folders, params, transform_params):

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", experiment_name + ".pth")
    model = create_model(params['use_cuda'], params['classes'])
    name = model_name + ".pth"
    saved_model = load_model_state(model, model_name)
    metrics = test(saved_model, data_folders, params, transform_params)
    pprint(metrics)
    return

def inference(model_name, image_path, params, transform_params):

    model = create_model(params['use_cuda'], params['classes'])
    model_name = model_name + ".pth"
    saved_model = load_model_state(model, model_name)
    for image_name in os.listdir(image_path):
        max_score = inference(model_name, os.path.join(image_path, image_name), params, transform_params)
    return


def get_default_params():
    params = {}
    params['im_size'] = (512, 1024)
    params['classes'] = ["class 1", "class 2", "class 3", "class 4", "class 5"]
    params['num_epochs'] = 5
    params['batch_size'] = 4
    params['use_cuda'] = torch.cuda.is_available() # use gpu
    params['device'] = "cuda" if params['use_cuda'] else 'cpu'
    params['learning_rate'] = 0.01
    params['momentum'] = 0.1 # momentum term
    params['nesterov'] = True # use nesterov trick in optimizer
    params['scheduler_step_size'] = 7
    params['lr_gamma'] = 0.1 # learning rate decay
    return params


def get_default_transform_params(im_size):
    transform_params = {}
    transform_params['RGB'] = True
    transform_params['standardization'] = True
    transform_params['resize'] = im_size

    return transform_params


# args?


if __name__ == "__main__":
    main()

import os
from pprint import pprint
from test import test

import torch
from tqdm import tqdm

from inference import inference
from inference import inference as infer
from model.create_model import create_model
from train import train
from utils import load_model_state


def main():

    # Add argparser?

    # Specify folders for data
    data_folders = {}
    data_folders['data'] = os.path.join('data', 'images')
    data_folders['inferece'] = "inference_demo"

    experiment_name = "exp1"

    params = get_default_params(downsample_factor=1)
    params['quicktest'] = True

    transform_params = get_default_transform_params(params['im_size'])

    if 1:
        train_and_test(experiment_name, data_folders, params, transform_params)

    if 0:
        only_test(experiment_name, data_folders, params, transform_params)

    if 0:
        inference(experiment_name, data_folders, params, transform_params)


def train_and_test(experiment_name, data_folders, params, transform_params):

    trained_model, _, _, _ = train(
        experiment_name, data_folders, params, transform_params)
    metrics = test(trained_model, data_folders, params, transform_params)
    pprint(metrics)
    return


def fine_tune(model_name, data_folders, params, transform_params):  # not working rn
    # Continue training an existing model

    params['fine_tune'] = True
    model = create_model(params['use_cuda'], params['num_classes'], model_name)
    fine_tuned_model, best_acc, best_epoch, best_loss = train(
        model_name, data_folders, params, transform_params)
    return


def only_test(model_name, data_folders, params, transform_params):

    model = create_model(params['use_cuda'], params['classes'])
    name = model_name + ".pth"
    saved_model = load_model_state(model, model_name)
    metrics = test(saved_model, data_folders, params, transform_params)
    pprint(metrics)
    return


def inference(model_name, data_folders, params, transform_params):
    image_path = data_folders['inference']
    model = create_model(params['use_cuda'], params['classes'])
    predictions = []
    saved_model = load_model_state(model, model_name)
    for image_name in os.listdir(image_path):
        predictions.extend(infer(saved_model, os.path.join(
            image_path, image_name), params, transform_params))

    pprint(predictions)
    return


def get_default_params(downsample_factor):
    params = {}
    default_im_size = (512, 1024)
    downsample_factor = 1
    im_size = tuple([int(x/downsample_factor) for x in default_im_size])
    params['im_size'] = im_size
    params['classes'] = ["plate"]
    params['num_epochs'] = 5
    params['weight_decay'] = 2e-05
    params['batch_size'] = 8
    params['use_cuda'] = torch.cuda.is_available()  # use gpu
    params['device'] = "cuda" if params['use_cuda'] else 'cpu'
    params['learning_rate'] = 0.01
    params['momentum'] = 0.1  # momentum term
    params['nesterov'] = True  # use nesterov trick in optimizer
    params['min_eta'] = 0.001
    params['num_workers'] = 8
    params['quicktest'] = False
    params['fine_tune'] = False
    params['splits'] = {'train': 0.6, 'val': 0.2, 'test': 0.2}
    return params


def get_default_transform_params(im_size):
    transform_params = {}
    transform_params['im_size'] = im_size

    return transform_params


# args?


if __name__ == "__main__":
    main()

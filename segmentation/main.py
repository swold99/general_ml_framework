import os
import sys
from pprint import pprint

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.test import SegmentationEvaluator
from segmentation.train import SegmentationTrainer


def main():

    # Add argparser?

    # Specify folders for data
    data_folders = {}
    data_folders['train'] = None
    data_folders['val'] = None
    data_folders['test'] = None
    data_folders['inferece'] = "inference_demo"

    experiment_name = "exp1"

    default_im_size = (256, 256)
    downsample_factor = 1
    im_size = tuple([int(x/downsample_factor) for x in default_im_size])

    params = get_default_params()
    params['im_size'] = im_size

    transform_params = get_default_transform_params(im_size)

    if 1:
        train_and_test(experiment_name, data_folders, params, transform_params)

    if 0:
        only_test(experiment_name, data_folders, params, transform_params)


def train_and_test(experiment_name, data_folders, params, transform_params):

    model_trainer = SegmentationTrainer(
        experiment_name, data_folders, params, transform_params)
    model_trainer.train_loop()
    model_evaluator = SegmentationEvaluator(
        experiment_name, data_folders, params, transform_params)
    metrics = model_evaluator.test_loop()
    pprint(metrics)
    return


def only_test(model_name, data_folders, params, transform_params):

    model_evaluator = SegmentationEvaluator(
        model_name, data_folders, params, transform_params)
    metrics = model_evaluator.test_loop()
    pprint(metrics)
    return


def get_default_params():
    params = {}

    # General params
    params['im_size'] = (256, 256)
    params['classes'] = ["background", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair",
                         "dining table", "potted plant", "sofa", "TV/monitor", "bird", "cat", "cow", "dog", "horse",
                         "sheep", "person", "boundary"]
    params['use_cuda'] = torch.cuda.is_available()  # use gpu
    params['device'] = "cuda" if params['use_cuda'] else 'cpu'
    params['num_workers'] = 4
    params['num_classes'] = len(params['classes'])
    params['quicktest'] = True
    params['use_datasets'] = ['vocseg']

    # Train params
    params['network'] = "unet"
    params['show_val_imgs'] = True
    params['show_test_imgs'] = True
    params['num_epochs'] = 5
    params['batch_size'] = 4
    params['patience'] = 0.1 * params['num_epochs']

    # Optim params
    params['optim_type'] = 'sgd'
    params['loss_fn'] = 'cross_entropy'
    params['learning_rate'] = 0.01
    params['momentum'] = 0.1  # momentum term
    params['nesterov'] = True  # use nesterov trick in optimizer
    params['schedule_type'] = 'step'
    params['scheduler_step_size'] = torch.max(torch.tensor([1, int(0.1*params['num_epochs'])]))
    params['lr_gamma'] = 0.1  # learning rate decay
    return params


def get_default_transform_params(im_size):
    transform_params = {}
    transform_params['trivial_augment'] = False
    transform_params['resize'] = im_size

    return transform_params


# args?


if __name__ == "__main__":
    main()

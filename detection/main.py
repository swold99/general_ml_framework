import argparse
import os
import sys
from pprint import pprint

import torch

from detection.test import DetectionEvaluator
from detection.train import DetectionTrainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():

    # Add argparser?

    # Specify folders for data
    data_folders = {}
    data_folders['train'] = None
    data_folders['val'] = None
    data_folders['test'] = None
    data_folders['inferece'] = "inference_demo"

    experiment_name = "betterdet"

    default_im_size = (256, 256)
    downsample_factor = 1
    im_size = tuple([int(x/downsample_factor) for x in default_im_size])

    params = get_default_params()
    params['im_size'] = im_size

    transform_params = get_default_transform_params(im_size)

    if 0:
        train_and_test(experiment_name, data_folders, params, transform_params)

    if 1:
        only_test(experiment_name, data_folders, params, transform_params)


def train_and_test(experiment_name, data_folders, params, transform_params):

    model_trainer = DetectionTrainer(
        experiment_name, data_folders, params, transform_params)
    model_trainer.train_loop()
    model_evaluator = DetectionEvaluator(
        experiment_name, data_folders, params, transform_params)
    metrics = model_evaluator.test_loop()
    pprint(metrics)
    return


def only_test(model_name, data_folders, params, transform_params):

    model_evaluator = DetectionEvaluator(
        model_name, data_folders, params, transform_params)
    metrics = model_evaluator.test_loop()
    pprint(metrics)
    return


def get_default_params():

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--task', type=str,
                        default='detection', help='Task')  # DO NOT CHANGE

    # General params
    parser.add_argument('--classes', type=list, default=["aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair",
                                                         "diningtable", "pottedplant", "sofa", "tvmonitor", "bird", "cat", "cow", "dog", "horse",
                                                         "sheep", "person"], help='Classes')
    parser.add_argument('--use_cuda', type=bool,
                        default=torch.cuda.is_available(), help='Use GPU')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available()
                        else 'cpu', help='Device (cuda or cpu)')
    parser.add_argument('--num_workers', type=int,
                        default=4, help='Number of workers')
    parser.add_argument('--quicktest', type=bool,
                        default=True, help='Quick test')
    parser.add_argument('--use_datasets', type=list,
                        default=['vocdet'], help='List of datasets')

    # Train params
    parser.add_argument('--network', type=str,
                        default="fasterrcnn_resnet50", help='Network type')
    parser.add_argument('--show_val_imgs', type=bool,
                        default=False, help='Show validation images')
    parser.add_argument('--show_test_imgs', type=bool,
                        default=True, help='Show test images')
    parser.add_argument('--num_epochs', type=int,
                        default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--patience', type=float, default=0.1, help='Patience')

    # Optim params
    parser.add_argument('--optim_type', type=str,
                        default='sgd', help='Optimizer type')
    parser.add_argument('--loss_fn', type=str,
                        default='cross_entropy', help='Loss function')
    parser.add_argument('--learning_rate', type=float,
                        default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float,
                        default=0.1, help='Momentum term')
    parser.add_argument('--nesterov', type=bool, default=True,
                        help='Use Nesterov trick in optimizer')
    parser.add_argument('--schedule_type', type=str,
                        default='step', help='Scheduler type')
    parser.add_argument('--scheduler_step_size', type=int,
                        default=0.2, help='Scheduler step size')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.5, help='Learning rate decay')

    # Bounding box params
    parser.add_argument('--score_threshold', type=float,
                        default=0.5, help='Score threshold')
    parser.add_argument('--iou_threshold', type=float,
                        default=0.5, help='IoU threshold')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create the params dictionary
    params = vars(args)

    # Set number of classes and patience (in epochs)    params['num_classes'] = len(params['classes'])
    params['patience'] = params['patience']*params['num_epochs']
    params['scheduler_step_size'] = torch.max(torch.tensor(
        [1, int(params['scheduler_step_size']*params['num_epochs'])]))

    return params


def get_default_transform_params(im_size):
    transform_params = {}
    transform_params['trivial_augment'] = False
    transform_params['resize'] = im_size
    return transform_params


# args?


if __name__ == "__main__":
    main()

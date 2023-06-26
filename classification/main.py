import argparse
import os
import sys
from pprint import pprint

import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.test import ClassificationEvaluator
from classification.train import ClassificationTrainer



def main():

    # Set the experiment name
    experiment_name = "test"

    # Set the default image size and downsample factor
    default_im_size = (64, 64)
    downsample_factor = 1
    im_size = tuple([int(x/downsample_factor) for x in default_im_size])

    # Get the default parameters and update the image size
    params = get_default_params()
    params['im_size'] = im_size

    # Get the default transform parameters
    transform_params = get_default_transform_params(im_size)

    if 1:
        # Train and test the model
        train_and_test(experiment_name, params, transform_params)

    if 0:
        # Only test the model
        only_test(experiment_name, params, transform_params)


def train_and_test(experiment_name, params, transform_params):

    # Create a model trainer for classification
    model_trainer = ClassificationTrainer(
        experiment_name, params, transform_params)

    # Train the model
    model_trainer.train_loop()

    # Create a model evaluator for classification
    model_evaluator = ClassificationEvaluator(
        experiment_name, params, transform_params)

    # Test the model
    metrics = model_evaluator.test_loop()

    # Print the metrics
    pprint(metrics)
    return


def only_test(model_name, params, transform_params):

    # Create a model evaluator for classification
    model_evaluator = ClassificationEvaluator(
        model_name, params, transform_params)

    # Test the model
    metrics = model_evaluator.test_loop()

    # Print the metrics
    pprint(metrics)
    return

def get_default_params():

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--task', type=str,
                        default='classification', help='Task')  # DO NOT CHANGE

    # General params
    parser.add_argument('--classes', type=list, default=["airplane", "automobile",
                        "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"], help='Classes')
    parser.add_argument('--use_cuda', type=bool,
                        default=torch.cuda.is_available(), help='Use GPU')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available()
                        else 'cpu', help='Device (cuda or cpu)')
    parser.add_argument('--num_workers', type=int,
                        default=8, help='Number of workers')
    parser.add_argument('--quicktest', type=bool,
                        default=True, help='Quick test')
    parser.add_argument('--use_datasets', type=list,
                        default=['cifar10'], help='List of datasets')

    # Train params
    parser.add_argument('--network', type=str,
                        default="resnet", help='Network type')
    parser.add_argument('--show_val_imgs', type=bool,
                        default=False, help='Show validation images')
    parser.add_argument('--show_test_imgs', type=bool,
                        default=False, help='Show test images')
    parser.add_argument('--num_epochs', type=int,
                        default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patience', type=float, default=0.1, help='Patience')

    # Optim params
    parser.add_argument('--optim_type', type=str,
                        default='sgd', help='Optimizer type') # 'sgd', 'adam', 'adamw'
    parser.add_argument('--loss_fn', type=str,
                        default='cross_entropy', help='Loss function') # 'cross_entropy', 'dice'
    parser.add_argument('--learning_rate', type=float,
                        default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.01, help='Weight decay')
    
    # SGD params
    parser.add_argument('--momentum', type=float,
                        default=0.1, help='Momentum term')
    parser.add_argument('--nesterov', type=bool, default=True,
                        help='Use Nesterov trick in optimizer')
    
    # Learning rate scheduling params
    parser.add_argument('--schedule_type', type=str,
                        default='step', help='Scheduler type')
    parser.add_argument('--scheduler_step_size', type=int,
                        default=0.3, help='Scheduler step size')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.8, help='Learning rate decay')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create the params dictionary
    params = vars(args)

    # Set number of classes and patience (in epochs)    
    params['num_classes'] = len(params['classes'])
    params['patience'] = params['patience']*params['num_epochs']
    params['scheduler_step_size'] = torch.max(torch.tensor(
        [1, int(params['scheduler_step_size']*params['num_epochs'])]))

    return params


def get_default_transform_params(im_size):
    transform_params = {}
    transform_params['trivial_augment'] = True
    transform_params['resize'] = im_size
    return transform_params


# args?


if __name__ == "__main__":
    main()

import os
from pprint import pprint
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from detection.train import DetectionTrainer
from detection.test import DetectionEvaluator


def main():

    # Add argparser?

    # Specify folders for data
    data_folders = {}
    data_folders['train'] = None
    data_folders['val'] = None
    data_folders['test'] = None
    data_folders['inferece'] = "inference_demo"

    experiment_name= "exp1"

    default_im_size = (64, 64)
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

    model_trainer = DetectionTrainer(experiment_name, data_folders, params, transform_params)
    model_trainer.train_loop()
    model_evaluator = DetectionEvaluator(experiment_name, data_folders, params, transform_params)
    metrics = model_evaluator.test_loop()
    pprint(metrics)
    return

def only_test(model_name, data_folders, params, transform_params):

    model_evaluator = DetectionEvaluator(model_name, data_folders, params, transform_params)
    metrics = model_evaluator.test_loop()
    pprint(metrics)
    return


def get_default_params():
    params = {}
    params['task'] = 'detection' # DO NOT CHANGE

    # General params
    params['classes'] = ["aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair",
                         "dining table", "potted plant", "sofa", "TV/monitor", "bird", "cat", "cow", "dog", "horse",
                         "sheep", "person"]
    params['use_cuda'] = torch.cuda.is_available() # use gpu
    params['device'] = "cuda" if params['use_cuda'] else 'cpu'
    params['num_workers'] = 4
    params['num_classes'] = len(params['classes'])
    params['quicktest'] = True
    params['use_datasets'] = ['vocdet']

    # Train params
    params['network'] = "fasterrcnn_resnet50"
    params['show_val_imgs'] = True
    params['show_test_imgs'] = True
    params['num_epochs'] = 5
    params['batch_size'] = 4
    params['patience'] = 0.1 * params['num_epochs']

    # Optim params
    params['optim_type'] = 'sgd'
    params['loss_fn'] = 'cross_entropy'
    params['learning_rate'] = 0.01
    params['momentum'] = 0.1 # momentum term
    params['nesterov'] = True # use nesterov trick in optimizer
    params['schedule_type'] = 'step'
    params['scheduler_step_size'] = torch.max(torch.tensor([1, int(0.1*params['num_epochs'])]))
    params['lr_gamma'] = 0.1 # learning rate decay
    return params


def get_default_transform_params(im_size):
    transform_params = {}
    transform_params['trivial_augment'] = True
    transform_params['resize'] = im_size
    return transform_params


# args?


if __name__ == "__main__":
    main()

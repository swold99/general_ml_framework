import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from custom_transforms import compose_transforms
from customimagedataset import CustomImageDataset
from utils import get_metrics, read_image, show_tensor_image


def test(model, data_folders, params, transform_params):
    test_path = data_folders['test']
    use_cuda = params['use_cuda']
    classes = params['classes']
    num_classes = params['num_classes']
    num_workers = params['num_workers']


    # Download images and create training set, trainloader, testing set, testloader
    im_size = params['im_size']
    tsfrm = compose_transforms(transform_params=transform_params) # INREASE IMAGE SIZE

    testset = CustomImageDataset(test_path, transform=tsfrm) # add testing images here

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=
    num_workers)
    correct = 0
    total = 0
    predictions = []
    true_targets = []
    incorrect_examples = []
    top5_preds = []

    with torch.no_grad():
        # Get a grasp of what is really necessary?
        for idx, (inputs, targets, sample_fnames) in enumerate(tqdm(testloader)):

            # Move data to GPU if CUDA is available
            if use_cuda:
                targets = list(targets)
                targets = torch.tensor([int(label) for label in targets])
                targets = targets.cuda()

            inputs = inputs.clone().detach().requires_grad_(True)

            # Feed-forward the network
            outputs = model(inputs.to('cuda'))

            _, predicted = torch.max(outputs.data, 1)


            top5_preds.extend(torch.topk(outputs.data, 5, 1)[1].tolist())
            # Convert to softmax probabilities?

            # Threshold stuff?

            total += targets.size(0)

            predictions.extend(predicted.cpu())
            true_targets.extend(targets.cpu())

            incorrect_idx = ((predicted == targets) == False).nonzero()

            correct += (predicted == targets).sum().item()


    print('Accuracy of the network on the %d test images: %d %%' % (total,
        100 * correct / total))

    top5_acc = calc_top5_acc(top5_preds, true_targets)

    # Print top5 accuracy
    print('Top 5 accuracy: ', float(top5_acc))

    # create confusion matrix
    metric_dict = get_metrics(torch.tensor(predictions), torch.tensor(true_targets), n_classes)

    return metric_dict

def calc_top5_acc(top5_preds, true_targets):
    n_samples = len(top5_preds)
    top5_preds = torch.tensor(top5_preds)
    targets = torch.tensor([int(t) for t in true_targets])
    top5_correct = torch.sum(targets==top5_preds.T)
    return top5_correct/n_samples
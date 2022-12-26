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
    n_classes = len(classes)

    # Download images and create training set, trainloader, testing set, testloader
    im_size = params['im_size']
    tsfrm = compose_transforms(transform_params=transform_params) # INREASE IMAGE SIZE

    testset = CustomImageDataset(test_path, transform=tsfrm) # add testing images here

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=8)
    correct = 0
    total = 0
    predictions = []
    true_targets = []
    incorrect_examples = []

    with torch.no_grad():
        # Get a grasp of what is really necessary?
        for idx, (inputs, targets, sample_fnames) in enumerate(tqdm(testloader)):

            # Move data to GPU if CUDA is available
            if use_cuda:
                targets = list(targets)
                targets = torch.tensor([int(label)-1 for label in targets])
                targets = targets.cuda()

            inputs = inputs.clone().detach().requires_grad_(True)

            # Feed-forward the network
            outputs = model(inputs.to('cuda'))

            _, predicted = torch.max(outputs.data, 1)

            # Convert to softmax probabilities?

            # Threshold stuff?

            total += targets.size(0)

            predictions.extend(predicted.cpu())
            true_targets.extend(targets.cpu())

            incorrect_idx = ((predicted == targets) == False).nonzero()
            for idx in incorrect_idx:
                incorrect_examples.append(sample_fnames[idx])

            correct += (predicted == targets).sum().item()


    print('Accuracy of the network on the %d test images: %d %%' % (total,
        100 * correct / total))

    # create confusion matrix
    metric_dict = get_metrics(torch.tensor(predictions), torch.tensor(true_targets), n_classes)

    return metric_dict


import torch
from tqdm import tqdm

from custom_transforms import compose_transforms
from customimagedataset import CustomImageDataset


def inference(model, data_folders, params, transform_params):

    test_path = data_folders['test']
    use_cuda = params['use_cuda']
    classes = params['classes']

    predictions = []

    tsfrm = compose_transforms(transform_params=transform_params)

    testset = CustomImageDataset(test_path, transform=tsfrm)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=4)

    # PROBABLY NEEDS OPTIMIZATION
    with torch.no_grad():
        for idx, (inputs, targets, sample_fnames) in enumerate(tqdm(testloader)):

            # Move data to GPU if CUDA is available
            if use_cuda:
                inputs = inputs.cuda()

            inputs = inputs.clone().detach().requires_grad_(True)

            # Feed-forward the network
            output = model(inputs.to('cuda'))

            #print("outputs: ", outputs)
            prob, predicted = torch.max(output.data, 1)

            predictions.extend(classes[predicted.cpu()])

    return predictions








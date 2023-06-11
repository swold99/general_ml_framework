import ntpath

import torch
from PIL import Image
from tqdm import tqdm

from custom_transforms import compose_transforms
from create_dataset import CustomImageDataset


def inference(model, image_name, params, transform_params):

    test_path = data_folders['test']
    use_cuda = params['use_cuda']
    classes = params['classes']

    predictions = []

    head, tail = ntpath.split(image_name)

    tsfrm = compose_transforms(transform_params=transform_params)

    img = Image.open(image_name)

    input = tsfrm(img)

    # PROBABLY NEEDS OPTIMIZATION
    with torch.no_grad():

        # Move data to GPU if CUDA is available
        if use_cuda:
            inputs = inputs.cuda()

        input = input.clone().detach().float().unsqueeze(0)

        # Feed-forward the network
        output = model(inputs)

        #print("outputs: ", outputs)
        probs, predicted = torch.topk(torch.nn.functional.softmax(output.data[0],0),5)

        out_list = [tail]

        for i in range(5):
            class_score = [classes[int(predicted[i])], round(float(probs[i]), 4)]
            out_list.append(class_score)


        predictions.append(out_list)

    return predictions








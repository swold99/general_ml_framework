import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.functional.classification import *
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from custom_transforms import compose_transforms


def load_show_image(path):
    # Loads and shows single image, not used anywhere
    transform_params = get_default_transform_params()
    image = read_image(path)
    tsfrm = compose_transforms(transform_params=transform_params)
    image = tsfrm((image, path))
    show_tensor_image(image)
    return

def load_imgs(folder, limit=None):
    # Loads all images from specified folder path (relative to current folder)
    images = []
    if limit is None:
        list_folder = os.listdir(folder)
    else:
        list_folder = os.listdir(folder)[:limit]

    for filename in tqdm(sorted(list_folder)):
        image = read_image(os.path.join(folder, filename))
        if image is not None:
            images.append(image)
    return images

def show_image(img):
    # Show png image, not used anywhere
    cv2.imshow('Bildfan', img)
    cv2.waitKey(0)
    return

def show_tensor_image(img, RGB = False, figsize = (100,100), wait_for_button=False):
    plt.figure(figsize=figsize)
    # Show tensor image
    if wait_for_button:
        plt.ion()

    if RGB: # Channels need to be permutated if RGB
        plt.imshow(img.permute(1, 2, 0), cmap='gray')
    else:
        if len(img.shape) == 4: # if image has been resized, it will be a 4d-tensor
            img = img.squeeze(0)
        # Squeeze away the channel dim
        if len(img.shape) == 3:
            img = img.squeeze(0)
        plt.imshow(img,  cmap='gray')
    if wait_for_button:
        plt.waitforbuttonpress()
        plt.close()
    else:
        plt.show()
    return

def visualize_dataset(folder, figsize = (6.4, 4.8), start_image = 1, labels=True, resize=(512, 1024), transform_params=None):
    # Visualizes dataset

    transform_params['resize'] = resize
    tsfrm = compose_transforms(transform_params=transform_params)

    # Loop through folder
    for file_nr, filename in tqdm(enumerate(sorted(os.listdir(folder)))):
        if file_nr >= start_image:

            # Read and transform image
            image = read_image(os.path.join(folder, filename))
            image = tsfrm((image, filename))

            # Show labels if image is labeled
            if labels:
                label_dict = {}
                show_labeled_image(image, label_dict)
            else:
                show_tensor_image(image, figsize=figsize, wait_for_button=True, RGB=True)
            print('\n Name: ', filename, ' \n File number: ', file_nr)
    return

def plot_distribution(img):
    # Plots distribution of pixel values in image, not used anywhere BUT VERY GOOD FOR VISUALIZING AND ANALYZING DATA. REMEMBER THIS FOR THE REPORT

    # Convert tensor image to numpy array
    img_np = np.array(img)

    # plot the pixel values
    plt.hist(img_np.ravel(), bins=50, density=True)
    plt.xlabel("pixel values")
    plt.ylabel("relative frequency")
    plt.title("distribution of pixels")
    plt.show()
    return

def save_model(model,name):
    """Save the state of the model"""
    torch.save(model.state_dict(), os.path.join("weights", name))

def load_model_state(model, name, local=True):
    """Loads the state of the model which is saved in *name*"""
    model.cuda()
    state_dict = torch.load(os.path.join("models", name), map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_single_image(model, img_path):
    # Predicts class of single image, not used anywhere

    transform_params = get_default_transform_params()
    model.eval()
    img = read_image(img_path)
    trans = compose_transforms(transform_params=transform_params)
    input = trans((img, img_path))
    print(input.shape)
    input = input.view(1, input.shape[0], input.shape[1], input.shape[2]).cuda()

    # Feed-forward the network
    outputs = model(input)

    #print("outputs: ", outputs)
    _, predicted = torch.max(outputs.data, 1)
    if predicted == 0:
        label = 'Not metal'
    else:
        label = 'Metal'
    plt.figure()
    plt.title(f"Image path: {img_path}, Predicted label: {label}")
    input = input.squeeze(0)
    show_tensor_image(input.cpu(), RGB=True)
    return label

def get_default_transform_params():
    # Only for the convenience of the function above
    transform_params = {}
    return transform_params



def collate_fn(batch):
    # Really useful function
    return tuple(zip(*batch))

def move_to(obj, device):
    # Move all items of list or dicts to specified device

    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")

def calc_IoU(boxA, boxB, xyxy=True):
    # Calculate intersection over union

    if not xyxy:
        boxA_area = boxA[2]*boxA[3]
        boxBarea = boxB[2]*boxB[3]

        # Get corners of bounding boxes
        boxA_x_min, boxA_x_max = boxA[0] - boxA[2], boxA[0] + boxA[2]
        boxA_y_min, boxA_y_max = boxA[1] - boxA[3], boxA[1] + boxA[3]

        boxB_x_min, boxB_x_max = boxB[0] - boxB[2], boxB[0] + boxB[2]
        boxB_y_min, boxB_y_max = boxB[1] - boxA[3], boxB[1] + boxB[3]
    else:
        boxA_area = abs(boxA[0]-boxA[2])*abs(boxA[1]-boxA[3])
        boxBarea = abs(boxB[0]-boxB[2])*abs(boxB[1]-boxB[3])

        boxA_x_min, boxA_y_min, boxA_x_max, boxA_y_max = boxA[0], boxA[1], boxA[2], boxA[3]
        boxB_x_min, boxB_y_min, boxB_x_max, boxB_y_max = boxB[0], boxB[1], boxB[2], boxB[3]

    # Get corners of intersection rectangle
    inter_x_min, inter_x_max = max(boxA_x_min, boxB_x_min), min(boxA_x_max, boxB_x_max)
    inter_y_min, inter_y_max = max(boxA_y_min, boxB_y_min), min(boxA_y_max, boxB_y_max)

    intersection = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    union = boxA_area + boxBarea - intersection

    iou = intersection/union
    return iou

def remove_duplicates(boxes, scores, labels, iou_threshold):
    # Remove overlapping bounding boxes, only keep the one with the highest score
    best_box_idx = []
    removed_box_idx = []

    # Loop through all boxes
    for i in range(boxes.shape[0]):

        # Check if it has been removed already
        if i not in removed_box_idx:
            intersecting_box_idx = [i]

            # Loop through the rest of the boxes
            for j in range(i+1, boxes.shape[0]):

                # Check if some of the rest of the boxes has been removed
                if j not in removed_box_idx:
                    iou = calc_IoU(boxes[i,:], boxes[j,:])

                    # Add to list of intersecting boxes if there is a overlap
                    if iou > iou_threshold:
                        intersecting_box_idx.append(j)

            # Get the best box
            max_score_idx = torch.argmax(scores[intersecting_box_idx])
            max_score_box = intersecting_box_idx[max_score_idx]
            best_box_idx.append(max_score_box)

            # Remove the rest
            intersecting_box_idx.remove(max_score_box)
            removed_box_idx = removed_box_idx + intersecting_box_idx

    # Return coordinates, labels and scores of best bounding boxes
    unique_boxes = boxes[best_box_idx, :]
    unique_labels = labels[best_box_idx]
    unique_scores = scores[best_box_idx]
    return unique_boxes, unique_labels, unique_scores

def show_bounding_boxes(image, pred_boxes, true_boxes, pred_classes, label_classes, sample_fname, scores):
    # Shows bounding boxes on image
    spaces = []
    for box in pred_boxes:
        width = box[2]-box[0]
        spaces.append(int(width/6))
    pred_classes_and_scores = [" "*spaces[i] + "%s %.3f"  % (pred_classes[i], scores[i]) for i in range(len(pred_classes))]
    drawn_boxes = draw_bounding_boxes((image*255).type(torch.uint8), pred_boxes, labels=pred_classes_and_scores, colors="blue")

    if true_boxes is not None:
        drawn_boxes = draw_bounding_boxes(drawn_boxes, true_boxes,  colors="red")#labels=label_classes, colors="red")
    print(sample_fname)

    if torch.numel(pred_boxes) == 0:
        drawn_boxes = drawn_boxes.to('cpu')

    show_tensor_image(drawn_boxes, RGB=True, wait_for_button=True)
    return

def show_labeled_image(img, label_dict):
    # Shows only labeled bounding boxes along with the image
    img = img*255
    img = img.type(torch.uint8)
    if img.shape[0] == 1:
        img = img.repeat(3,1,1)
    if not label_dict:
        plt.imshow(img.permute(1,2,0))
    else:
        bounding_boxes = label_dict['boxes']
        result = draw_bounding_boxes(img, bounding_boxes)
        plt.imshow(result.permute(1,2,0))
    plt.waitforbuttonpress()
    plt.close()
    return

def save_annotation(img_name, boxes, pred_classes):
    json_dict = {}
    shapes = []
    for idx in range(len(pred_classes)):
        shapes.append({
            "label": pred_classes[idx],
            "points" : [
                [int(boxes[idx][1]), int(boxes[idx][0])],
                [int(boxes[idx][1]), int(boxes[idx][2])],
                [int(boxes[idx][3]), int(boxes[idx][0])],
                [int(boxes[idx][3]), int(boxes[idx][2])]
                ]
            })
    json_dict['shapes'] = shapes
    annot_name = img_name.split(".")[0] + "_annot.json"
    #change where to save annotations, didn't want to save with all other annotations yet
    with open(os.path.join("/courses", "TSBB11", "tsbb11_2022ht_1d-timber-x-ray", "new_test_images", "annotations", annot_name), 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)


def save_model(model, name):
    """Save the state of the model"""
    folder = os.path.join(os.curdir, "weights")
    path = os.path.join(folder, name + ".pth")
    torch.save(model.state_dict(), path)

def save_fig(train_loss, val_loss, f1_train=None, f1_val=None,
             exp_name=None, meas="F1-score"):
    # Saves curve with loss and f1-score from training
    f, (ax1, ax2) = plt.subplots(1, 2)
    x = np.arange(len(train_loss))

    ax1.plot(x, train_loss, label="Training")
    ax1.plot(x, val_loss, label="Validation")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend(loc="upper right")

    if f1_train is not None:
        ax2.plot(x, f1_train, label="Training")
        ax2.plot(x, f1_val, label="Validation")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel(meas)
        ax2.set_title(meas)
        ax1.legend(loc="lower right")
        f.suptitle("Training vs Validation curves")

    fname = "training_curves/" + exp_name + ".png"
    f.savefig(fname)

    return
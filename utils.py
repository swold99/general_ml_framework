import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.functional.classification import confusion_matrix
from torchvision.utils import draw_bounding_boxes


def save_model(model, name):
    """Save the state of the model"""
    folder = os.path.join(os.curdir, "weights")
    path = os.path.join(folder, name + ".pth")
    torch.save(model.state_dict(), path)


def load_model_state(model, name, local=True):
    """Loads the state of the model which is saved in *name*"""
    model.cuda()
    state_dict = torch.load(os.path.join("weights", name), map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collate_fn(batch):
    # Really useful function
    return tuple(zip(*batch))


def custom_collate_fn(batch):
    # Find the maximum height and width in the batch
    max_height = max([img.size(1) for img, _ in batch])
    max_width = max([img.size(2) for img, _ in batch])

    stacked_images = []
    stacked_targets = []
    for img, target in batch:
        # Pad the image to match the maximum height and width
        padded_img = torch.nn.functional.pad(
            img, (0, max_width - img.size(2), 0, max_height - img.size(1)))
        stacked_images.append(padded_img)
        stacked_targets.append(target)

    # Stack the images along a new batch dimension
    stacked_images = torch.stack(stacked_images)

    return stacked_images, stacked_targets


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


def plot_save_conf_matrix(predicted_labels, true_labels, class_labels, filename, num_classes):
    # Assuming you have the predicted labels and true labels as numpy arrays

    # Convert tensors to numpy arrays
    predicted_labels = torch.tensor(predicted_labels)
    true_labels = torch.tensor(true_labels)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels,
                          task='multiclass', num_classes=num_classes)

    # Plot confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    # Fill the matrix with values and labels
    thresh = cm.max() / 2.0
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the confusion matrix as an image
    plt.savefig(os.path.join("conf_matrix", filename + 'conf.png'))

    # Show the confusion matrix
    plt.show()


def show_classification_imgs(inputs, targets, preds):
    for i in range(inputs.shape[0]):
        plt.imshow(inputs[i, ...].permute(1, 2, 0))
        plt.title(f'Label: {targets[i]}, Prediction: {preds[i]}')
        plt.show()


def show_segmentation_imgs(inputs, targets, preds, colormap):
    titles = ['Input', 'Label', 'Prediction']
    for i in range(inputs.shape[0]):
        target_color = apply_colormap(targets[i, ...], colormap)
        pred_color = apply_colormap(preds[i, ...], colormap)
        imgs = [inputs[i, ...].permute(1, 2, 0), target_color, pred_color]
        # Create subplots with 1 row and 3 columns
        fig, axs = plt.subplots(1, 3)
        for j in range(3):
            # Plot your images and set titles
            axs[j].imshow(imgs[j])
            axs[j].set_title(titles[j])

        # Adjust the layout to avoid overlapping titles
        plt.tight_layout()
        plt.show()


def show_detection_imgs(inputs, preds, targets, classes):
    """
    Display images with predicted and ground truth bounding boxes.

    Args:
        inputs (list): List of input images.
        preds (list): List of dictionaries containing predicted bounding boxes, labels, and scores.
        targets (list): List of dictionaries containing ground truth bounding boxes and labels.
        classes (list): List of class labels.

    Returns:
        None
    """
    for i in range(len(inputs)):
        pred_boxes, pred_labels, pred_scores = preds[i]['boxes'], preds[i]['labels'], preds[i]['scores']
        target_boxes, target_labels = targets[i]['boxes'], targets[i]['labels']
        spaces = []
        for box in pred_boxes:
            width = box[2] - box[0]
            spaces.append(int(width / 6))
        pred_classes_and_scores = [" " * spaces[i] + "%s %.3f" % (
            classes[pred_labels[i].item()], pred_scores[i]) for i in range(len(pred_labels))]

        for box in target_boxes:
            width = box[2] - box[0]
            spaces.append(int(width / 6))
        target_labels = [" " * spaces[i] + classes[target_labels[i].item()]
                         for i in range(len(target_labels))]
        drawn_boxes = draw_bounding_boxes(
            (inputs[i] * 255).type(torch.uint8), pred_boxes, labels=pred_classes_and_scores, colors="blue")
        drawn_boxes = draw_bounding_boxes(
            drawn_boxes, target_boxes, labels=target_labels, colors="red")
        plt.imshow(drawn_boxes.permute(1, 2, 0))
        plt.title('Predicted bounding boxes (blue) and ground truth bounding boxes (red)')
        plt.show()



def show_bounding_boxes(image, pred_boxes, true_boxes, pred_classes, label_classes, sample_fname, scores):
    # Shows bounding boxes on image
    spaces = []
    for box in pred_boxes:
        width = box[2]-box[0]
        spaces.append(int(width/6))
    pred_classes_and_scores = [" "*spaces[i] + "%s %.3f" %
                               (pred_classes[i], scores[i]) for i in range(len(pred_classes))]
    drawn_boxes = draw_bounding_boxes(
        (image*255).type(torch.uint8), pred_boxes, labels=pred_classes_and_scores, colors="blue")

    if true_boxes is not None:
        # labels=label_classes, colors="red")
        drawn_boxes = draw_bounding_boxes(
            drawn_boxes, true_boxes,  colors="red")
    print(sample_fname)

    if torch.numel(pred_boxes) == 0:
        drawn_boxes = drawn_boxes.to('cpu')

    return


def apply_colormap(mask, colormap):
    """
    Apply a colormap to a mask image.

    Args:
        mask (ndarray): Mask image.
        colormap (Callable): Colormap function.

    Returns:
        ndarray: RGB image with the applied colormap.
    """
    rgb_image = colormap(mask)[:, :, :3] * 255  # Apply colormap and select RGB channels
    rgb_image = rgb_image.astype(np.uint8)  # Convert image to uint8 data type
    return rgb_image


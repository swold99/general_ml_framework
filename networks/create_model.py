import torch.nn as nn
from torchvision.models import ResNet18_Weights, detection, resnet18
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from networks.unet import UNet


def create_classification_model(network, device, num_classes):
    # Create a classification model based on the specified network, device, and number of classes

    if 'resnet' in network:
        # Create a ResNet18 model with pretrained weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features

        # Replace the fully connected layer with a new one
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def create_segmentation_model(network, device, num_classes):
    # Create a segmentation model based on the specified network, device, and number of classes
    # (Update the model implementation)

    if 'unet' in network:
        # Create a UNet model with the specified number of input and output channels
        model = UNet(in_channels=3, out_channels=num_classes)

    return model


def create_detection_model(network, device, num_classes):
    # Create a detection model based on the specified network, device, and number of classes
    # (Update the model implementation)

    if 'fasterrcnn_resnet50' in network:
        # Create a Faster R-CNN model with ResNet50 backbone and pretrained weights
        model = detection.fasterrcnn_resnet50_fpn(
            weights=detection.faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Replace the box predictor with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

    return model

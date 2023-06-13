import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision.models import ResNet18_Weights, detection, resnet18
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from networks.unet import UNet


def create_classification_model(network, device, num_classes):
    if 'resnet' in network:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features

        # Fully connected layer
        model.fc = nn.Linear(num_ftrs, num_classes)

    if "cuda" in device:
        model.cuda()
        cudnn.benchmark = True

    return model


def create_segmentation_model(network, device, num_classes):
    # UPDATE MODEL
    if 'unet' in network:
        model = UNet(in_channels=3, out_channels=num_classes)

    if "cuda" in device:
        model.cuda()
        cudnn.benchmark = True

    return model


def create_detection_model(network, device, num_classes):
    # UPDATE MODEL
    if 'fasterrcnn_resnet50' in network:
        model = detection.fasterrcnn_resnet50_fpn(
            weights=detection.faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if "cuda" in device:
        model.cuda()
        cudnn.benchmark = True

    return model

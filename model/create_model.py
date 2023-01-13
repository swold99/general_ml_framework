import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights,
                                          fasterrcnn_resnet50_fpn)


def create_model(model="faster-rcnn", use_cuda=True, num_classes=0):

    if model == "faster-rcnn":
        model = create_faster_rcnn_model(num_classes)
        return model

    # UPDATE MODEL
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features

    # Fully connected layer
    model.fc = nn.Linear(num_ftrs, num_classes)
    if use_cuda:
        model.cuda()
        cudnn.benchmark = True

    return model


def create_faster_rcnn_model(num_classes):
    # Load Faster RCNN pre-trained model
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Define a new head for the detector with required number of of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

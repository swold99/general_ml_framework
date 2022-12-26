import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def create_model(use_cuda, num_classes):

    # UPDATE MODEL
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features

    # Fully connected layer
    model.fc = nn.Linear(num_ftrs, num_classes)    
    if use_cuda:
            model.cuda()
            cudnn.benchmark = True
    
    return model
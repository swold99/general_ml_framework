import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

def create_classification_model(network, device, num_classes):
	# UPDATE MODEL
	model = resnet18(weights=ResNet18_Weights.DEFAULT)
	num_ftrs = model.fc.in_features

	# Fully connected layer
	model.fc = nn.Linear(num_ftrs, num_classes)    
	if "cuda" in device:
			model.cuda()
			cudnn.benchmark = True
                        
	return model

def create_model(network, device, num_classes):

    # UPDATE MODEL
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features

    # Fully connected layer
    model.fc = nn.Linear(num_ftrs, num_classes)    
    if "cuda" in device:
            model.cuda()
            cudnn.benchmark = True
    
    return model
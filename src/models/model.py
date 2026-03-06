import torch.nn as nn
from torchvision import models


from torchvision import models
import torch.nn as nn


def get_model(model_name="resnet18", num_classes=101):

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    else:
        raise ValueError("Unknown model")

    return model
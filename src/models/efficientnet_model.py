import torch.nn as nn
from torchvision import models

def get_efficientnet_model(num_classes=101):

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model
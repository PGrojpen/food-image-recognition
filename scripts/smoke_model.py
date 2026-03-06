import torch
from src.model import get_model

model = get_model(num_classes=101)

x = torch.randn(32, 3, 224, 224)
out = model(x)

print(out.shape)
import torch
from src.models import get_resnet_model

def main():
    model = get_resnet_model(num_classes=101)

    x = torch.randn(32, 3, 224, 224)
    out = model(x)

    print(out.shape)

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_dataloaders
from src.models import get_resnet_model


def smoke_train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = get_dataloaders(
        data_root="data/food-101",
        batch_size=64
    )

    model = get_resnet_model(num_classes=101).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    model.train()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        print(f"Smoke train step OK — loss: {loss.item():.4f}")

        break


if __name__ == "__main__":
    smoke_train()
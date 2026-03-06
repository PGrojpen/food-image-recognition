import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import json
import os

from datetime import datetime
from src.data import get_dataloaders
from src.models.model import get_model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = correct / total
            progress_bar.set_postfix(acc=f"{acc:.4f}")

    return correct / total

def main(model_name, batch_size, epochs):

    os.makedirs("artifacts/checkpoints", exist_ok=True)
    os.makedirs("artifacts/metrics", exist_ok=True)

    train_loader, test_loader = get_dataloaders("data/food-101/food-101", batch_size, model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name=model_name).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    checkpoint_path = f"artifacts/checkpoints/{model_name}_best.pth"
    if os.path.exists(checkpoint_path):
        best_global = torch.load(checkpoint_path, map_location="cpu")
        best_global_acc = best_global["val_acc"]
    else:
        best_global_acc = 0.0

    metrics = []

    experiment = {
    "model": model_name,
    "epochs": epochs,
    "batch_size": batch_size,
    "metrics": metrics
}

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_acc = evaluate(
            model, test_loader, device
        )

        metrics.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_acc": val_acc
        })

        if val_acc > best_global_acc:
            best_global_acc = val_acc

            torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_acc": val_acc
            },f"artifacts/checkpoints/{model_name}_best.pth")

            print(f"Checkpoint (acc: {val_acc:.4f})")

        print(f"Epoch {epoch+1} | Loss {train_loss:.4f} | Acc {val_acc:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    metrics_path = f"artifacts/metrics/{model_name}_{timestamp}.json"

    with open(metrics_path, "w") as f:
        json.dump(experiment, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Train a food image classification model"
    )
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18"],)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()

    main(args.model, args.batch_size, args.epochs)
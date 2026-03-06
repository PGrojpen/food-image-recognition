import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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

def main():
    train_loader, test_loader = get_dataloaders("data/food-101/food-101")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3

    best_acc = 0.0

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_acc = evaluate(
            model, test_loader, device
        )
        if val_acc > best_acc:
            best_acc = val_acc

            torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss,
            },"artifacts/checkpoint.pth")

        print(f"Epoch {epoch+1} | Loss {train_loss:.4f} | Acc {val_acc:.4f}")
        print(f"Checkpoint (acc: {val_acc:.4f})")
        

if __name__ == "__main__":
    main()
from src.data import get_dataloaders

def main():
    train_loader, test_loader = get_dataloaders(
        data_root="data/food-101",
        batch_size=32
    )

    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)
    print(labels[:5])

if __name__ == "__main__":
    main()
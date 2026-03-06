from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataset_loader import Food101Dataset


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def get_datasets(data_root):
    train_dataset = Food101Dataset(
        root=data_root,
        split="train",
        transform=get_train_transform()
    )

    test_dataset = Food101Dataset(
        root=data_root,
        split="test",
        transform=get_test_transform()
    )

    return train_dataset, test_dataset


def get_dataloaders(data_root, batch_size=32):
    train_dataset, test_dataset = get_datasets(data_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader
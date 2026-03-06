from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataset_loader import Food101Dataset

def get_train_transform(model_name="resnet18"):
    if model_name == "resnet18":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    raise ValueError(f"Modelo desconhecido: {model_name}")


def get_test_transform(model_name="resnet18"):
    if model_name == "resnet18":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    raise ValueError(f"Modelo desconhecido: {model_name}")

def get_datasets(data_root, model_name="resnet18"):
    train_dataset = Food101Dataset(
        root=data_root,
        split="train",
        transform=get_train_transform(model_name)
    )

    test_dataset = Food101Dataset(
        root=data_root,
        split="test",
        transform=get_test_transform(model_name)
    )

    return train_dataset, test_dataset


def get_dataloaders(data_root, batch_size=32, model_name="resnet18"):
    train_dataset, test_dataset = get_datasets(data_root, model_name)

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
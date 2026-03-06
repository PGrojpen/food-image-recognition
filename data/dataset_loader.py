from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class Food101Dataset(Dataset):

    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform

        meta_file = self.root / "meta" / f"{split}.txt"

        with open(meta_file) as f:
            self.samples = [line.strip() for line in f]

        classes = sorted({s.split("/")[0] for s in self.samples})
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image_path = self.root / "images" / f"{sample}.jpg"
        label_name = sample.split("/")[0]
        label = self.class_to_idx[label_name]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
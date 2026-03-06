from data.dataset_loader import Food101Dataset

dataset = Food101Dataset(
    root="data/food-101/food-101",
    split="train"
)

print("dataset size:", len(dataset))

image, label = dataset[0]

print("label:", label)
print("image size:", image.size)
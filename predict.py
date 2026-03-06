import argparse
import torch
from PIL import Image

from src.data import get_test_transform
from src.models.model import get_model


def load_model(model_name, checkpoint_path, device):
    model = get_model(model_name=model_name)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model


def load_classes(path="data/food-101/food-101/meta/classes.txt"):
    with open(path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f]
    return classes


def predict(image_path, model, device, model_name="resnet18"):
    transform = get_test_transform(model_name)

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)

        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        confidence = probs[0, pred].item()

    classes = load_classes()
    return classes[pred], confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict the class of a food image"
    )

    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--image", type=str, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = f"artifacts/checkpoints/{args.model}_best.pth"

    model = load_model(args.model, checkpoint_path, device)
    prediction, confidence = predict(args.image, model, device, args.model)

    print(f"Prediction: {prediction} ({confidence:.4f})")
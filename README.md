# Food-101 Image Classification
Deep learning computer vision model that identifies dishes from images (Food-101 dataset).

The goal is to train and compare different convolutional neural network models capable of recognizing food categories from images.

## Dataset

This project uses the Food-101 dataset.

Food-101 contains 101 food categories and 101,000 images (~5GB).

Dataset source:
https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

The dataset is stored locally in the `data/` folder and is not included in the repository.

To automatically download and extract the dataset, run:

```bash
python scripts/download.py
```

## Models

The project will compare:

* A baseline CNN implemented from scratch
* ResNet18 using transfer learning
* EfficientNet_B0 using transfer learning

## Testing

## Quick Test

Run the smoke tests to verify that the pipeline works:

```bash
python -m scripts.smoke_dataloader
python -m scripts.smoke_model
python -m scripts.smoke_train_step
```

## Project Structure

```
artifacts/
data/
scripts/
src/
predict.py
requirements.txt
README.md
LICENSE
.gitignore
```

## Tools

* Python
* PyTorch
* Torchvision

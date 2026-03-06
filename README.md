# Food-101 Image Classification
Deep learning computer vision model that identifies dishes from images (Food-101 dataset).

The goal is to train and compare different convolutional neural network models capable of recognizing food categories from images.

## Dataset

Food-101 contains 101 food categories and 101,000 images.

Dataset source:
https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

The dataset is stored locally in the `data/` folder and is not included in the repository.

## Models

The project will compare:

* A baseline CNN implemented from scratch
* ResNet18 using transfer learning
* EfficientNet using transfer learning

## Project Structure

```
data/       dataset (not tracked by Git)
src/        source code
outputs/    trained models and results
main.py     project entry point
```

## Tools

* Python
* PyTorch
* Torchvision

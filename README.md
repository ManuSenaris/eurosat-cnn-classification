# 🛰️ EuroSAT CNN Classification

Satellite image classification using Convolutional Neural Networks in PyTorch.

## Overview

Full classification pipeline for the **EuroSAT** dataset, which contains 27,000 satellite images of 64×64 pixels across 10 balanced classes. The project tackles the problem from two angles: a CNN trained from scratch and Transfer Learning with VGG16.

## Classes

| Label | Class |
|-------|-------|
| 0 | AnnualCrop |
| 1 | Forest |
| 2 | HerbaceousVegetation |
| 3 | Highway |
| 4 | Industrial |
| 5 | Pasture |
| 6 | PermanentCrop |
| 7 | Residential |
| 8 | River |
| 9 | SeaLake |

## Results

| Model | Test Accuracy | Epochs |
|-------|--------------|--------|
| Custom CNN | 93.00% | 20 |
| VGG16 Partial Fine-Tuning | 97.15% | 10 |

## Project Structure

```
eurosat-cnn-classification/
├── EuroSAT/                  # Images organized by class
│   ├── AnnualCrop/
│   ├── Forest/
│   └── ...
├── train.csv                 # Training split
├── validation.csv            # Validation split
├── test.csv                  # Test split
├── EuroSAT_CNN.ipynb         # Main notebook
└── README.md
```

## Pipeline

### 1. Data Preparation
- Custom `EuroSATDataset` class inheriting from `torch.utils.data.Dataset`
- Image paths and labels loaded from CSV files
- Data Augmentation on train set: `RandomHorizontalFlip`, `RandomRotation(15°)`, `ColorJitter`
- Different normalization for Custom CNN (0.5) and VGG16 (ImageNet mean/std)

### 2. Custom CNN
Custom architecture with two convolutional blocks:

```
Input (3 × 64 × 64)
    ↓ Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool
(64 × 32 × 32)
    ↓ Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool
(128 × 16 × 16)
    ↓ Flatten → Linear(32768, 256) → ReLU → Dropout(0.5) → Linear(256, 10)
Output (10 classes)
```

Design decisions:
- **BatchNorm** after every convolution to stabilize gradients
- **Dropout(0.5)** in the fully-connected layer for regularization
- **ReduceLROnPlateau** scheduler to adapt to convergence
- Gradient monitoring to detect vanishing/exploding gradients

### 3. Latent Space Analysis
- Extraction of 256-dimensional feature vectors from the penultimate layer
- Dimensionality reduction with PCA (256D → 50D) followed by t-SNE (50D → 2D)
- Visualization of per-class clusters in the latent space

### 4. Transfer Learning with VGG16
- Model pretrained on ImageNet (1.2M images, 1000 classes)
- Last layer replaced: `Linear(4096, 1000)` → `Linear(4096, 10)`
- Partial Fine-Tuning: first 3 blocks frozen, blocks 4-5 and classifier trainable
- ~29.8M trainable parameters out of 134M total

## Installation

```bash
# Clone the repository
git clone https://github.com/your_username/eurosat-cnn-classification.git
cd eurosat-cnn-classification

# Create virtual environment
conda create -n eurosat python=3.11
conda activate eurosat

# Install dependencies
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm pillow
```

## Usage

Open the main notebook and run the cells in order:

```bash
jupyter notebook EuroSAT_CNN.ipynb
```

The notebook is organized in the following sections:
1. Imports and setup
2. Transforms
3. Dataset and DataLoaders
4. Custom CNN architecture
5. Training and monitoring
6. Latent space analysis
7. Test evaluation (classification report + confusion matrix)
8. Transfer Learning with VGG16
9. Results comparison

## Tech Stack

- Python 3.11
- PyTorch
- torchvision
- scikit-learn
- pandas
- matplotlib
- tqdm

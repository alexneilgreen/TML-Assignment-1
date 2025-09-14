# FGSM Attack Implementation - Assignment 1

## Overview

Implementation of Fast Gradient Sign Method (FGSM) attacks on ResNet-18 and Vision Transformer models using MNIST and CIFAR-10 datasets.

## Requirements

```bash
pip install torch torchvision matplotlib numpy
```

## File Structure

```
├── main.py                   # Main execution file with argument parsing
├── resnet.py                 # ResNet-18 implementation
├── vit.py                    # Vision Transformer implementation
├── fgsm.py                   # FGSM attack functions
├── README.md                 # This file
├── data/                     # Dataset downloads (auto-created)
├── models/                   # Saved model weights (auto-created)
└── results/                  # Output files (auto-created)
```

## Exact Reproduction Commands

### Task 1: Train Models and Show Clean Accuracy

```bash
python main.py --task 1
```

This will:

- Download MNIST and CIFAR-10 datasets
- Train/load ResNet-18 and ViT models for both datasets
- Evaluate clean accuracy on test sets
- Generate accuracy comparison plot (`clean_accuracy.png`)
- Save results to `clean_accuracy_results.json`

### Task 2: Full FGSM Attack Evaluation

```bash
python main.py --task 2
```

This will:

- Load trained models from Task 1
- Evaluate all attack scenarios with ε ∈ {1,2,4,8}/255
- Test untargeted and targeted attacks (random + least-likely targets)
- Generate attack visualizations for each model/dataset combination
- Save complete results to `results/fgsm_task2_results.json`

### Expected Output Files

**Task 1:**

- `clean_accuracy.png` - Accuracy comparison chart
- `clean_accuracy_results.json` - Clean accuracy results
- `models/*.pth` - Trained model weights

**Task 2:**

- `fgsm_visualization_ResNet18_mnist_8.png` - MNIST ResNet attack visualizations
- `fgsm_visualization_ViT_mnist_8.png` - MNIST ViT attack visualizations
- `fgsm_visualization_ResNet18_cifar10_8.png` - CIFAR-10 ResNet attack visualizations
- `fgsm_visualization_ViT_cifar10_8.png` - CIFAR-10 ViT attack visualizations
- `results/fgsm_task2_results.json` - Complete attack evaluation results

## Model Configurations

### ResNet-18

- Standard ResNet-18 architecture adapted for small images
- Input channels: 1 (MNIST), 3 (CIFAR-10)
- Training: 10 epochs (MNIST), 15 epochs (CIFAR-10)
- Learning rate: 0.001

### Vision Transformer (ViT)

- Patch size: 4×4
- Embedding dimension: 192
- Depth: 6 transformer layers
- Attention heads: 3
- Training: 10 epochs (MNIST), 15 epochs (CIFAR-10)
- Learning rate: 0.0005

## FGSM Attack Implementation

### Algorithm

```
Untargeted: x_adv = x + ε × sign(∇_x L(θ, x, y))
Targeted:   x_adv = x - ε × sign(∇_x L(θ, x, y_target))
```

### Evaluation Metrics (Task 2)

- **Clean Accuracy**: Performance on unmodified test images
- **Robust Accuracy**: Performance on adversarial examples per ε
- **Attack Success Rate (ASR)**: Percentage of successful untargeted attacks
- **Targeted ASR**: Success rate for targeted attacks
  - Random target: Randomly chosen different class
  - Least-likely target: Class with minimum logit from clean prediction

### Attack Configurations

- Epsilon values: {1, 2, 4, 8}/255
- Test subset: Fixed 1,000 images per dataset
- Threat model: L∞ bounded perturbations

## Reproducibility Settings

- Fixed seeds: `torch.manual_seed(42)`, `np.random.seed(42)`
- Deterministic CUDA operations enabled
- Fixed test subset using shuffled indices with seed

## Expected Performance Targets

- **MNIST**: ≥95% clean accuracy for both models
- **CIFAR-10**: ResNet-18 ≈85%+, ViT ≈80%+ clean accuracy

## Runtime Estimates

- **Task 1**: ~15-20 minutes total (GPU recommended)
  - Training: ~5-8 minutes per model
- **Task 2**: ~5-10 minutes (assumes models trained in Task 1)
  - Attack evaluation: ~2-3 minutes per model/dataset pair

## Dependencies

- Python 3.7+
- PyTorch 1.9+
- torchvision
- matplotlib
- numpy

## Usage Notes

- Run Task 1 before Task 2 to ensure models are trained
- GPU highly recommended for reasonable training times
- All paths are relative to script directory
- Results include confidence scores and detailed metrics tables

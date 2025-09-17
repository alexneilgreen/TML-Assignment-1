# FGSM Attack Implementation - Assignment 1

## Overview

Implementation of Fast Gradient Sign Method (FGSM) attacks under L∞ threat model on ResNet-18 and Vision Transformer architectures using MNIST and CIFAR-10 datasets. Evaluates targeted vs. untargeted attacks with comprehensive robustness metrics.

## Requirements

```bash
pip install torch torchvision matplotlib numpy
```

## File Structure

```
├── main.py                   # Main execution file with task selection
├── resnet.py                 # ResNet-18 implementation
├── vit.py                    # Vision Transformer implementation  
├── fgsm.py                   # FGSM attack functions and evaluation
├── README.md                 # This file
├── data/                     # Dataset downloads (auto-created)
├── models/                   # Saved model weights (auto-created)
└── results/                  # All output files (auto-created)
```

## Exact Reproduction Commands

### Run Complete Pipeline
```bash
python main.py
```

### Train Models and Evaluate Clean Accuracy
```bash
python main.py --task 1
```

### Full FGSM Attack Evaluation  
```bash
python main.py --task 2
```

## Expected Output Files

**Task 1 generates:**
- `!resnet18_mnist_training.json` - Training parameters
- `!resnet18_cifar10_training.json` - Training parameters
- `!vit_mnist_training.json` - Training parameters
- `!vit_cifar10_training.json` - Training parameters
- `0_clean_accuracy_results.json` - Clean accuracy data
- `1_clean_accuracy.png` - Accuracy comparison chart
- `2_clean_training.png` - Training curves visualization

**Task 2 generates:**
- `0_fgsm_task2_results.json` - Complete attack evaluation results
- `3_fgsm_visualization_ResNet18_mnist_8.png` - Attack visualizations (ε=8/255)
- `3_fgsm_visualization_ViT_mnist_8.png` - Attack visualizations (ε=8/255)
- `3_fgsm_visualization_ResNet18_cifar10_8.png` - Attack visualizations (ε=8/255)
- `3_fgsm_visualization_ViT_cifar10_8.png` - Attack visualizations (ε=8/255)
- `4_table_mnist_results.csv` - Results table for MNIST
- `4_table_cifar10_results.csv` - Results table for CIFAR-10

## Model Configurations

**ResNet-18:**
- Adapted for 28×28 (MNIST) and 32×32 (CIFAR-10) inputs
- Training: 10 epochs (MNIST), 30 epochs (CIFAR-10)
- Learning rate: 0.001, Adam optimizer

**Vision Transformer:**
- Patch size: 4×4, Embed dim: 256, Depth: 8, Heads: 4
- Training: 10 epochs (MNIST), 30 epochs (CIFAR-10) with early stopping
- Learning rate: 0.00025, AdamW optimizer with cosine annealing

## FGSM Implementation

**Untargeted Attack:**
```
x_adv = x + ε × sign(∇_x L(θ, x, y_true))
```

**Targeted Attack:**  
```
x_adv = x - ε × sign(∇_x L(θ, x, y_target))
```

**Epsilon Values:** {1, 2, 4, 8}/255  
**Test Subset:** Fixed 1,000 images per dataset

## Performance Targets

- **MNIST**: ≥95% clean accuracy (both models)
- **CIFAR-10**: ResNet-18 ≈85%+, ViT ≈80%+ clean accuracy

## Evaluation Metrics

- **Clean Accuracy**: Performance on original test images
- **Robust Accuracy**: Performance under adversarial perturbations per ε
- **Attack Success Rate (ASR)**: Untargeted attack success percentage
- **Targeted ASR**: Success rates for:
  - Random target: Different randomly chosen class
  - Least-likely target: Class with minimum clean prediction logit

## FGSM Visualizations

Attack visualizations show 10 examples per model/dataset with:
- **Row 1**: Original images with predictions/confidence
- **Row 2**: Perturbations (×10 magnification for visibility)  
- **Row 3**: Adversarial images with new predictions/confidence

## Reproducibility Settings

- Fixed seeds: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`
- Deterministic CUDA operations enabled
- Fixed test subset using seeded random indices

## Dependencies

- Python 3.7+
- PyTorch 1.9+
- torchvision, matplotlib, numpy

## Usage Notes

- Task 2 requires Task 1 completion (or existing model files)
- GPU highly recommended for training
- All file paths relative to script directory
- Models automatically saved/loaded to avoid retraining
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import random

# Import our models and attack functions
from resnet import ResNet18
from vit import VisionTransformer
from fgsm import evaluate_robustness, fgsm_attack

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_dataloaders(dataset_name: str, batch_size: int = 128):
    """Load and return train/test dataloaders for specified dataset"""
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
    elif dataset_name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, epochs=10, lr=0.001):
    """Train a model and return the trained model"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        print(f'Epoch {epoch+1} completed. Accuracy: {100.*correct/total:.2f}%')
    
    return model

def evaluate_model(model, test_loader):
    """Evaluate model accuracy on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Clean Accuracy: {accuracy:.2f}%')
    return accuracy

def visualize_clean_accuracy(results, save_path='clean_accuracy.png'):
    """Create visualization showing clean accuracy for all models and datasets"""
    datasets = ['MNIST', 'CIFAR-10']
    models = ['ResNet18', 'ViT']
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    resnet_acc = [results['mnist']['ResNet18'], results['cifar10']['ResNet18']]
    vit_acc = [results['mnist']['ViT'], results['cifar10']['ViT']]
    
    bars1 = ax.bar(x - width/2, resnet_acc, width, label='ResNet18', alpha=0.8)
    bars2 = ax.bar(x + width/2, vit_acc, width, label='ViT', alpha=0.8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Clean Accuracy (%)')
    ax.set_title('Clean Accuracy Comparison: ResNet18 vs ViT')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Clean accuracy plot saved to: {save_path}")

def visualize_attacks(model, test_subset, epsilon=8/255, num_samples=10, 
                     dataset_name='cifar10', model_name='ResNet18'):
    """Visualize original, perturbation, and adversarial examples"""
    model.eval()
    
    # Get first batch
    data, target = next(iter(test_subset))
    data, target = data[:num_samples].to(device), target[:num_samples].to(device)
    
    # Generate adversarial examples
    adv_data = fgsm_attack(model, data, target, epsilon, targeted=False)
    perturbation = adv_data - data
    
    # Get predictions
    with torch.no_grad():
        orig_pred = model(data).argmax(dim=1)
        adv_pred = model(adv_data).argmax(dim=1)
        orig_conf = F.softmax(model(data), dim=1).max(dim=1)[0]
        adv_conf = F.softmax(model(adv_data), dim=1).max(dim=1)[0]
    
    # Convert to numpy for visualization
    data_np = data.cpu().numpy()
    adv_data_np = adv_data.cpu().numpy()
    perturbation_np = perturbation.cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Original
        if dataset_name.lower() == 'mnist':
            axes[0, i].imshow(data_np[i, 0], cmap='gray')
        else:
            img = np.transpose(data_np[i], (1, 2, 0))
            # Denormalize for CIFAR-10 visualization
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            axes[0, i].imshow(img)
        axes[0, i].set_title(f'Orig: {orig_pred[i].item()} ({orig_conf[i]:.2f})')
        axes[0, i].axis('off')
        
        # Perturbation (scaled for visibility)
        if dataset_name.lower() == 'mnist':
            axes[1, i].imshow(perturbation_np[i, 0] * 10, cmap='gray', vmin=-0.5, vmax=0.5)
        else:
            pert_vis = np.transpose(perturbation_np[i], (1, 2, 0)) * 10
            pert_vis = np.clip(pert_vis + 0.5, 0, 1)  # Center around 0.5 for visualization
            axes[1, i].imshow(pert_vis)
        axes[1, i].set_title('Perturbation (×10)')
        axes[1, i].axis('off')
        
        # Adversarial
        if dataset_name.lower() == 'mnist':
            axes[2, i].imshow(adv_data_np[i, 0], cmap='gray')
        else:
            adv_img = np.transpose(adv_data_np[i], (1, 2, 0))
            # Denormalize for CIFAR-10 visualization
            adv_img = adv_img * std + mean
            adv_img = np.clip(adv_img, 0, 1)
            axes[2, i].imshow(adv_img)
        axes[2, i].set_title(f'Adv: {adv_pred[i].item()} ({adv_conf[i]:.2f})')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    save_path = f'fgsm_visualization_{model_name}_{dataset_name}_{epsilon*255:.0f}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Attack visualization saved to: {save_path}")

def task1():
    """Task 1: Train models and show clean accuracy"""
    print("="*60)
    print("TASK 1: Training models and evaluating clean accuracy")
    print("="*60)
    
    datasets = ['mnist', 'cifar10']
    clean_results = {}
    
    # Create directories
    Path('./models').mkdir(exist_ok=True)
    
    for dataset_name in datasets:
        print(f"\n{'-'*40}")
        print(f"Processing {dataset_name.upper()} dataset")
        print(f"{'-'*40}")
        
        # Load data
        train_loader, test_loader = get_dataloaders(dataset_name)
        
        # Model configurations
        if dataset_name.lower() == 'mnist':
            in_channels = 1
            img_size = 28
        else:
            in_channels = 3
            img_size = 32
        
        models = {
            'ResNet18': ResNet18(num_classes=10, in_channels=in_channels),
            'ViT': VisionTransformer(img_size=img_size, patch_size=4, in_channels=in_channels,
                                   num_classes=10, embed_dim=192, depth=6, n_heads=3)
        }
        
        dataset_results = {}
        
        for model_name, model in models.items():
            print(f"\n{model_name} on {dataset_name}:")
            
            model_path = f'./models/{model_name}_{dataset_name}.pth'
            
            # Train or load model
            if Path(model_path).exists():
                print(f"Loading existing model from {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=device))
            else:
                print(f"Training new model...")
                epochs = 15 if dataset_name == 'cifar10' else 10
                lr = 0.001 if model_name == 'ResNet18' else 0.0005
                model = train_model(model, train_loader, epochs=epochs, lr=lr)
                torch.save(model.state_dict(), model_path)
            
            # Evaluate clean accuracy
            clean_acc = evaluate_model(model, test_loader)
            dataset_results[model_name] = clean_acc
        
        clean_results[dataset_name] = dataset_results
    
    # Create visualization
    visualize_clean_accuracy(clean_results)
    
    # Save results
    with open('./clean_accuracy_results.json', 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print("\n" + "="*60)
    print("TASK 1 SUMMARY")
    print("="*60)
    for dataset in clean_results:
        print(f"\n{dataset.upper()}:")
        for model in clean_results[dataset]:
            print(f"  {model}: {clean_results[dataset][model]:.2f}%")

def task2():
    """Task 2: Full FGSM evaluation with all metrics"""
    print("="*60)
    print("TASK 2: Full FGSM attack evaluation")
    print("="*60)
    
    datasets = ['mnist', 'cifar10']
    epsilons = [1/255, 2/255, 4/255, 8/255]
    test_subset_size = 1000
    
    # Create directories
    Path('./results').mkdir(exist_ok=True)
    Path('./models').mkdir(exist_ok=True)
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'-'*50}")
        print(f"Processing {dataset_name.upper()} dataset")
        print(f"{'-'*50}")
        
        # Load data
        train_loader, test_loader = get_dataloaders(dataset_name)
        
        # Create test subset
        test_dataset = test_loader.dataset
        test_indices = list(range(len(test_dataset)))
        np.random.shuffle(test_indices)
        subset_indices = test_indices[:test_subset_size]
        test_subset = Subset(test_dataset, subset_indices)
        test_subset_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
        
        # Model configurations
        if dataset_name.lower() == 'mnist':
            in_channels = 1
            img_size = 28
        else:
            in_channels = 3
            img_size = 32
        
        models = {
            'ResNet18': ResNet18(num_classes=10, in_channels=in_channels),
            'ViT': VisionTransformer(img_size=img_size, patch_size=4, in_channels=in_channels,
                                   num_classes=10, embed_dim=192, depth=6, n_heads=3)
        }
        
        dataset_results = {}
        
        for model_name, model in models.items():
            print(f"\n{'-'*30}")
            print(f"Evaluating {model_name} on {dataset_name}")
            print(f"{'-'*30}")
            
            model_path = f'./models/{model_name}_{dataset_name}.pth'
            
            # Load model (should exist from task 1)
            if Path(model_path).exists():
                print(f"Loading model from {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=device))
            else:
                print(f"Model not found! Training new model...")
                epochs = 15 if dataset_name == 'cifar10' else 10
                lr = 0.001 if model_name == 'ResNet18' else 0.0005
                model = train_model(model, train_loader, epochs=epochs, lr=lr)
                torch.save(model.state_dict(), model_path)
            
            # Evaluate robustness
            print(f"\nEvaluating robustness for {model_name} on {dataset_name}")
            results = evaluate_robustness(model, test_subset_loader, epsilons, device)
            dataset_results[model_name] = results
            
            # Create visualizations
            print(f"\nCreating visualizations...")
            visualize_attacks(model, test_subset_loader, epsilon=8/255, 
                            num_samples=10, dataset_name=dataset_name, model_name=model_name)
        
        all_results[dataset_name] = dataset_results
    
    # Print summary table
    print(f"\n{'='*100}")
    print("TASK 2 SUMMARY RESULTS")
    print(f"{'='*100}")
    
    for dataset_name in datasets:
        print(f"\n{dataset_name.upper()} Results:")
        print(f"{'Model':<10} {'Clean':<8} {'Rob(1/255)':<12} {'Rob(2/255)':<12} "
              f"{'Rob(4/255)':<12} {'Rob(8/255)':<12} {'ASR(8/255)':<12}")
        print("-" * 88)
        
        for model_name in ['ResNet18', 'ViT']:
            results = all_results[dataset_name][model_name]
            row = f"{model_name:<10} {results['clean_acc']:<8.2f} "
            for eps in epsilons:
                row += f"{results['robust_acc'][eps]:<12.2f} "
            row += f"{results['untargeted_asr'][8/255]:<12.2f}"
            print(row)
        
        print(f"\nTargeted Attack Success Rates for {dataset_name.upper()}:")
        print(f"{'Model':<10} {'Random(1/255)':<15} {'Random(8/255)':<15} "
              f"{'LeastLikely(1/255)':<18} {'LeastLikely(8/255)':<18}")
        print("-" * 76)
        
        for model_name in ['ResNet18', 'ViT']:
            results = all_results[dataset_name][model_name]
            row = f"{model_name:<10} "
            row += f"{results['targeted_asr_random'][1/255]:<15.2f} "
            row += f"{results['targeted_asr_random'][8/255]:<15.2f} "
            row += f"{results['targeted_asr_least_likely'][1/255]:<18.2f} "
            row += f"{results['targeted_asr_least_likely'][8/255]:<18.2f}"
            print(row)
    
    # Save results
    with open('./results/fgsm_task2_results.json', 'w') as f:
        # Convert numpy floats to regular floats for JSON serialization
        json_results = {}
        for dataset in all_results:
            json_results[dataset] = {}
            for model in all_results[dataset]:
                json_results[dataset][model] = {}
                for key, value in all_results[dataset][model].items():
                    if isinstance(value, dict):
                        json_results[dataset][model][key] = {str(k): float(v) for k, v in value.items()}
                    else:
                        json_results[dataset][model][key] = float(value)
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to ./results/fgsm_task2_results.json")

def main():
    parser = argparse.ArgumentParser(description='FGSM Attack Implementation')
    parser.add_argument('--task', type=int, choices=[1, 2], required=True,
                       help='Task to run: 1 = Train models and show accuracy, 2 = Full FGSM evaluation')
    
    args = parser.parse_args()
    
    if args.task == 1:
        task1()
    elif args.task == 2:
        task2()

if __name__ == "__main__":
    main()
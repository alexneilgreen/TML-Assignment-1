import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import csv
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

def get_model_configs():
    """Centralized model configurations"""
    return {
        'mnist': {
            'in_channels': 1,
            'img_size': 28,
            'ResNet18': lambda: ResNet18(num_classes=10, in_channels=1),
            'ViT': lambda: VisionTransformer(
                img_size=28, patch_size=4, in_channels=1,
                num_classes=10, embed_dim=256, depth=8, n_heads=4, dropout=0.1
            )
        },
        'cifar10': {
            'in_channels': 3,
            'img_size': 32,
            'ResNet18': lambda: ResNet18(num_classes=10, in_channels=3),
            'ViT': lambda: VisionTransformer(
                img_size=32, patch_size=4, in_channels=3,
                num_classes=10, embed_dim=256, depth=8, n_heads=4, dropout=0.1
            )
        }
    }

def train_model(model, train_loader, epochs=10, lr=0.001, model_name='ResNet18', save_name=''):
    """Train a model and return the trained model with training info"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if model_name == 'ViT':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(2*epochs))
        optimizer_type = 'AdamW'
        scheduler_type = 'CosineAnnealingLR'
        weight_decay = 0.05
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        optimizer_type = 'Adam'
        scheduler_type = 'StepLR'
        weight_decay = 1e-4
    
    best_acc = 0
    patience = 5
    patience_counter = 0
    epochs_trained = epochs
    
    # Track per-epoch accuracy
    epoch_accuracies = []
    
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
            
            if model_name == 'ViT':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}\tBatch: {batch_idx}\t'
                      f'Loss: {loss.item():.4f}\tAcc: {100.*correct/total:.2f}%')
        
        epoch_acc = 100.*correct/total
        epoch_accuracies.append(epoch_acc)
        print(f'Epoch: {epoch+1} completed. Accuracy: {epoch_acc:.2f}%\n')
        
        if model_name == 'ViT':
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience and epoch > 10:
                print(f'Early stopping at epoch {epoch+1}. Best accuracy: {best_acc:.2f}%')
                epochs_trained = epoch + 1
                break
        
        scheduler.step()
    
    # Store training parameters
    training_params = {
        'epochs': epochs_trained,
        'learning_rate': lr,
        'optimizer': optimizer_type,
        'scheduler': scheduler_type,
        'weight_decay': weight_decay,
        'criterion': 'CrossEntropyLoss'
    }
    
    if model_name == 'ViT':
        training_params['gradient_clipping'] = 1.0
        training_params['early_stopping'] = True
        training_params['patience'] = patience
    
    # Save training progress
    if save_name:
        Path('./results').mkdir(exist_ok=True)
        training_data = {
            'epoch_accuracies': epoch_accuracies,
            'training_parameters': training_params
        }
        
        with open(f'./results/!{save_name}_training.json', 'w') as f:
            json.dump(training_data, f, indent=2)
    
    return model, training_params

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

def visualize_clean_accuracy(results, save_path='results/1_clean_accuracy.png'):
    """Create visualization showing clean accuracy with target lines"""
    datasets = ['MNIST', 'CIFAR-10']
    models = ['ResNet18', 'ViT']
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    resnet_acc = [results['mnist']['ResNet18'], results['cifar10']['ResNet18']]
    vit_acc = [results['mnist']['ViT'], results['cifar10']['ViT']]
    
    bars1 = ax.bar(x - width/2, resnet_acc, width, label='ResNet18', alpha=0.8)
    bars2 = ax.bar(x + width/2, vit_acc, width, label='ViT', alpha=0.8)
    
    # Add target lines
    # MNIST: 95% for both models
    ax.axhline(y=95, xmin=0, xmax=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    # CIFAR-10 ResNet18: 85%
    ax.axhline(y=85, xmin=0.5, xmax=0.75, color='red', linestyle='--', alpha=0.7, linewidth=2)
    # CIFAR-10 ViT: 80%
    ax.axhline(y=80, xmin=0.75, xmax=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
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
    plt.close()
    print(f"Clean accuracy plot saved to: {save_path}")

def visualize_training_curves():
    """Create training curves visualization"""
    training_files = [
        ('!resnet18_mnist_training.json', 'ResNet18 MNIST', 'red'),
        ('!resnet18_cifar10_training.json', 'ResNet18 CIFAR-10', 'blue'),
        ('!vit_mnist_training.json', 'ViT MNIST', 'green'),
        ('!vit_cifar10_training.json', 'ViT CIFAR-10', 'orange')
    ]
    
    plt.figure(figsize=(12, 8))
    max_epochs = 0
    
    for filename, label, color in training_files:
        filepath = f'./results/{filename}'
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            epochs = list(range(1, len(data['epoch_accuracies']) + 1))
            accuracies = data['epoch_accuracies']
            max_epochs = max(max_epochs, len(epochs))
            
            plt.plot(epochs, accuracies, color=color, label=label, linewidth=2, marker='o', markersize=4)
            
            # Get clean accuracy from results file
            clean_acc_file = './results/0_clean_accuracy_results.json'
            if Path(clean_acc_file).exists():
                with open(clean_acc_file, 'r') as f:
                    clean_results = json.load(f)
                
                # Extract clean accuracy based on filename
                if 'mnist' in filename and 'resnet' in filename:
                    clean_acc = clean_results['mnist']['ResNet18']['clean_accuracy']

                elif 'mnist' in filename and 'vit' in filename:
                    clean_acc = clean_results['mnist']['ViT']['clean_accuracy']

                elif 'cifar' in filename and 'resnet' in filename:
                    clean_acc = clean_results['cifar10']['ResNet18']['clean_accuracy']

                elif 'cifar' in filename and 'vit' in filename:
                    clean_acc = clean_results['cifar10']['ViT']['clean_accuracy']
                    
                else:
                    break
                
                plt.scatter(epochs[-1], clean_acc, color=color, s=100, marker='s', 
                        edgecolors='black', linewidth=1, zorder=5)
                plt.annotate(f'{clean_acc:.1f}%', 
                            xy=(epochs[-1], clean_acc), 
                            xytext=(5, 5), 
                            textcoords='offset points',
                            fontweight='bold',
                color=color)
    
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Training Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_epochs + 1)
    
    plt.tight_layout()
    plt.savefig('./results/2_clean_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Training curves plot saved to: ./results/2_clean_training.png")

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
    adv_data_np = adv_data.detach().cpu().numpy()
    perturbation_np = perturbation.detach().cpu().numpy()
    
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
        axes[1, i].set_title('Perturbation')
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
    save_path = f'./results/3_fgsm_visualization_{model_name}_{dataset_name}_{epsilon*255:.0f}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attack visualization saved to: {save_path}")

def generate_results_table():
    """Generate CSV tables with results for each dataset"""
    
    # Load results
    results_file = './results/0_fgsm_task2_results.json'
    if not Path(results_file).exists():
        print(f"Results file {results_file} not found!")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Define epsilon values and headers
    epsilons = ['0.00392156862745098', '0.00784313725490196', '0.01568627450980392', '0.03137254901960784']  # 1/255, 2/255, 4/255, 8/255
    eps_labels = ['1/255', '2/255', '4/255', '8/255']
    
    headers = ['Model', 'Clean Acc (%)', 'Robust Acc 1/255 (%)', 'Robust Acc 2/255 (%)', 
               'Robust Acc 4/255 (%)', 'Robust Acc 8/255 (%)', 'ASR Untargeted 8/255 (%)', 
               'ASR Random Target 8/255 (%)', 'ASR Least-Likely 8/255 (%)']
    
    # Generate table for each dataset
    for dataset_name in ['mnist', 'cifar10']:
        if dataset_name not in results:
            print(f"Dataset {dataset_name} not found in results!")
            continue
            
        # Create CSV filename
        csv_filename = f'./results/4_table_{dataset_name}_results.csv'
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(headers)
            
            # Write data for each model
            for model_name in ['ResNet18', 'ViT']:
                if model_name not in results[dataset_name]:
                    continue
                    
                model_results = results[dataset_name][model_name]
                
                row = [model_name]
                
                # Clean accuracy
                row.append(f"{model_results['clean_acc']:.2f}")
                
                # Robust accuracy for each epsilon
                for eps in epsilons:
                    if eps in model_results['robust_acc']:
                        row.append(f"{model_results['robust_acc'][eps]:.2f}")
                    else:
                        row.append("N/A")
                
                # ASR Untargeted (8/255)
                eps_8 = epsilons[3]  # 8/255
                if eps_8 in model_results['untargeted_asr']:
                    row.append(f"{model_results['untargeted_asr'][eps_8]:.2f}")
                else:
                    row.append("N/A")
                
                # ASR Random Target (8/255)
                if eps_8 in model_results['targeted_asr_random']:
                    row.append(f"{model_results['targeted_asr_random'][eps_8]:.2f}")
                else:
                    row.append("N/A")
                
                # ASR Least-Likely (8/255)
                if eps_8 in model_results['targeted_asr_least_likely']:
                    row.append(f"{model_results['targeted_asr_least_likely'][eps_8]:.2f}")
                else:
                    row.append("N/A")
                
                writer.writerow(row)
        
        print(f"Table saved to: {csv_filename}")

def task1():
    """Task 1: Train models and show clean accuracy"""
    print("="*60)
    print("TASK 1: Training models and evaluating clean accuracy")
    print("="*60)
    
    datasets = ['mnist', 'cifar10']
    clean_results = {}
    model_configs = get_model_configs()
    
    # Create directories
    Path('./models').mkdir(exist_ok=True)
    Path('./results').mkdir(exist_ok=True)
    
    for dataset_name in datasets:
        print(f"\n{'-'*40}")
        print(f"Processing {dataset_name.upper()} dataset")
        print(f"{'-'*40}")
        
        # Load data
        train_loader, test_loader = get_dataloaders(dataset_name)
        
        # Get model configurations
        config = model_configs[dataset_name]
        
        dataset_results = {}
        
        for model_name in ['ResNet18', 'ViT']:
            print(f"\n{model_name} on {dataset_name}:\n")
            
            model_path = f'./models/{model_name}_{dataset_name}.pth'
            save_name = f'{model_name.lower()}_{dataset_name}'
            
            # Create model
            model = config[model_name]()
            
            # Check if model already exists
            if Path(model_path).exists():
                print(f"Loading existing model from {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=device))
                model = model.to(device)
                
                # Try to load existing training parameters
                training_json_path = f'./results/!{save_name}_training.json'
                if Path(training_json_path).exists():
                    with open(training_json_path, 'r') as f:
                        training_data = json.load(f)
                        training_params = training_data.get('training_parameters', {})
                else:
                    training_params = {}
            else:
                print(f"Training new model...")
                epochs = 30 if dataset_name == 'cifar10' else 10
                lr = 0.001 if model_name == 'ResNet18' else 0.00025
                model, training_params = train_model(model, train_loader, epochs=epochs, 
                                                lr=lr, model_name=model_name, save_name=save_name)
                torch.save(model.state_dict(), model_path)
            
            # Evaluate clean accuracy
            clean_acc = evaluate_model(model, test_loader)
            
            # Store results with training parameters
            dataset_results[model_name] = {
                'clean_accuracy': clean_acc,
                'training_parameters': training_params
            }
        
        clean_results[dataset_name] = dataset_results
    
    # Create visualizations
    viz_results = {}
    for dataset in clean_results:
        viz_results[dataset] = {}
        for model in clean_results[dataset]:
            viz_results[dataset][model] = clean_results[dataset][model]['clean_accuracy']
    
    visualize_clean_accuracy(viz_results)
    visualize_training_curves()
    
    # Save results with training parameters
    with open('./results/0_clean_accuracy_results.json', 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print("\n" + "="*60)
    print("TASK 1 SUMMARY")
    print("="*60)
    for dataset in clean_results:
        print(f"\n{dataset.upper()}:")
        for model in clean_results[dataset]:
            print(f"  {model}: {clean_results[dataset][model]['clean_accuracy']:.2f}%")

def task2():
    """Task 2: Full FGSM evaluation with all metrics"""
    print("="*60)
    print("TASK 2: Full FGSM attack evaluation")
    print("="*60)
    
    datasets = ['mnist', 'cifar10']
    epsilons = [1/255, 2/255, 4/255, 8/255]
    test_subset_size = 1000
    model_configs = get_model_configs()
    
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
        
        # Get model configurations
        config = model_configs[dataset_name]
        
        dataset_results = {}
        
        for model_name in ['ResNet18', 'ViT']:
            print(f"\n{'-'*30}")
            print(f"Evaluating {model_name} on {dataset_name}")
            print(f"{'-'*30}")
            
            model_path = f'./models/{model_name}_{dataset_name}.pth'
            
            # Create model with consistent configuration
            model = config[model_name]()
            
            # Load model (should exist from task 1)
            if Path(model_path).exists():
                print(f"Loading model from {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=device))
                model = model.to(device)
            else:
                print(f"Model not found! Training new model...")
                epochs = 30 if dataset_name == 'cifar10' else 10
                lr = 0.001 if model_name == 'ResNet18' else 0.00025
                save_name = f'{model_name.lower()}_{dataset_name}'
                model, _ = train_model(model, train_loader, epochs=epochs, lr=lr, model_name=model_name, save_name=save_name)
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
    with open('./results/0_fgsm_task2_results.json', 'w') as f:
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
    
    print(f"\nResults saved to ./results/0_fgsm_task2_results.json")
    
    # Generate CSV tables
    print(f"\nGenerating CSV tables...")
    generate_results_table()

def main():
    parser = argparse.ArgumentParser(description='FGSM Attack Implementation')
    parser.add_argument('--task', type=int, choices=[0, 1, 2], default=0,
                       help='Task to run: 0 = Run both tasks, 1 = Train models and show accuracy, 2 = Full FGSM evaluation')
    
    args = parser.parse_args()
    
    if args.task == 0:
        task1()
        task2()
    elif args.task == 1:
        task1()
    elif args.task == 2:
        task2()

if __name__ == "__main__":
    main()
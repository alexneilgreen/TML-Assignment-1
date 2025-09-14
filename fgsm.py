import torch
import torch.nn.functional as F

def fgsm_attack(model, data, target, epsilon, targeted=False, target_class=None):
    """
    Fast Gradient Sign Method attack
    
    Args:
        model: The target model
        data: Input images
        target: True labels
        epsilon: Attack strength
        targeted: If True, perform targeted attack
        target_class: Target class for targeted attack
    
    Returns:
        Adversarial examples
    """
    data = data.clone().detach().requires_grad_(True)
    
    if targeted and target_class is not None:
        # For targeted attack, minimize loss w.r.t. target class
        output = model(data)
        loss = F.cross_entropy(output, target_class)
        loss.backward()
        
        # Gradient descent (minimize loss)
        data_grad = data.grad.data
        perturbed_data = data - epsilon * data_grad.sign()
    else:
        # For untargeted attack, maximize loss w.r.t. true class
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Gradient ascent (maximize loss)
        data_grad = data.grad.data
        perturbed_data = data + epsilon * data_grad.sign()
    
    # Clamp to valid pixel range
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data

def get_least_likely_class(model, data, true_labels):
    """Get least likely class based on model's clean predictions"""
    model.eval()
    with torch.no_grad():
        logits = model(data)
        # Get class with minimum logit value
        least_likely = logits.argmin(dim=1)
    return least_likely

def evaluate_robustness(model, test_subset, epsilons, device):
    """
    Evaluate model robustness under FGSM attacks
    
    Returns:
        Dictionary containing all metrics
    """
    model.eval()
    results = {
        'clean_acc': 0,
        'robust_acc': {},
        'untargeted_asr': {},
        'targeted_asr_random': {},
        'targeted_asr_least_likely': {}
    }
    
    # Evaluate clean accuracy
    correct = 0
    total = 0
    all_data, all_targets = [], []
    
    for data, target in test_subset:
        data, target = data.to(device), target.to(device)
        all_data.append(data)
        all_targets.append(target)
        
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    results['clean_acc'] = 100.0 * correct / total
    print(f"Clean Accuracy: {results['clean_acc']:.2f}%")
    
    # Concatenate all data
    all_data = torch.cat(all_data, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    num_classes = 10
    
    for epsilon in epsilons:
        print(f"\nEvaluating epsilon = {epsilon}")
        
        # Untargeted attack
        adv_data_untargeted = fgsm_attack(model, all_data, all_targets, epsilon, targeted=False)
        
        with torch.no_grad():
            adv_pred = model(adv_data_untargeted).argmax(dim=1)
            robust_correct = adv_pred.eq(all_targets).sum().item()
            attack_success = (~adv_pred.eq(all_targets)).sum().item()
        
        results['robust_acc'][epsilon] = 100.0 * robust_correct / total
        results['untargeted_asr'][epsilon] = 100.0 * attack_success / total
        
        # Targeted attack - random target
        random_targets = torch.randint(0, num_classes, all_targets.shape, device=device)
        # Ensure random targets are different from true labels
        mask = random_targets == all_targets
        random_targets[mask] = (all_targets[mask] + 1) % num_classes
        
        adv_data_targeted_random = fgsm_attack(model, all_data, all_targets, epsilon, 
                                             targeted=True, target_class=random_targets)
        
        with torch.no_grad():
            adv_pred_random = model(adv_data_targeted_random).argmax(dim=1)
            targeted_success_random = adv_pred_random.eq(random_targets).sum().item()
        
        results['targeted_asr_random'][epsilon] = 100.0 * targeted_success_random / total
        
        # Targeted attack - least likely class
        least_likely_targets = get_least_likely_class(model, all_data, all_targets)
        
        adv_data_targeted_ll = fgsm_attack(model, all_data, all_targets, epsilon,
                                         targeted=True, target_class=least_likely_targets)
        
        with torch.no_grad():
            adv_pred_ll = model(adv_data_targeted_ll).argmax(dim=1)
            targeted_success_ll = adv_pred_ll.eq(least_likely_targets).sum().item()
        
        results['targeted_asr_least_likely'][epsilon] = 100.0 * targeted_success_ll / total
        
        print(f"  Robust Acc: {results['robust_acc'][epsilon]:.2f}%")
        print(f"  Untargeted ASR: {results['untargeted_asr'][epsilon]:.2f}%")
        print(f"  Targeted ASR (Random): {results['targeted_asr_random'][epsilon]:.2f}%")
        print(f"  Targeted ASR (Least-likely): {results['targeted_asr_least_likely'][epsilon]:.2f}%")
    
    return results
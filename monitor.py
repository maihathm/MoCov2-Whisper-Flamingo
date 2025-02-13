import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
from tqdm import tqdm

def load_tensorboard_data(log_dir):
    """Load training metrics from TensorBoard logs"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get available tags
    tags = event_acc.Tags()['scalars']
    
    # Load metrics
    metrics = {}
    for tag in tags:
        events = event_acc.Scalars(tag)
        metrics[tag] = {
            'steps': [event.step for event in events],
            'values': [event.value for event in events]
        }
    
    return metrics

def plot_training_curves(metrics, save_dir):
    """Plot and save training curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    if 'train_loss' in metrics:
        plt.plot(metrics['train_loss']['steps'], metrics['train_loss']['values'], label='Train Loss')
    if 'val_loss' in metrics:
        plt.plot(metrics['val_loss']['steps'], metrics['val_loss']['values'], label='Val Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()
    
    # Plot gate attention weights
    gate_metrics = {k: v for k, v in metrics.items() if 'gate' in k}
    if gate_metrics:
        plt.figure(figsize=(12, 6))
        for name, values in gate_metrics.items():
            plt.plot(values['steps'], values['values'], label=name)
        plt.xlabel('Steps')
        plt.ylabel('Gate Value')
        plt.title('Gate Attention Weights')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gate_weights.png'))
        plt.close()
    
    # Plot accuracy if available
    if 'val_accuracy' in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['val_accuracy']['steps'], metrics['val_accuracy']['values'])
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'accuracy.png'))
        plt.close()

def analyze_predictions(results_dir):
    """Analyze model predictions"""
    results = []
    for filename in os.listdir(results_dir):
        if filename.endswith('_results.txt'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                results.extend(f.readlines())
    
    # Extract predictions and targets
    predictions = []
    targets = []
    current = None
    
    for line in results:
        if line.startswith('Pred: '):
            current = line[6:].strip()
        elif line.startswith('Target: ') and current is not None:
            predictions.append(current)
            targets.append(line[8:].strip())
            current = None
    
    return predictions, targets

def compute_statistics(predictions, targets):
    """Compute various statistics from predictions"""
    from collections import Counter
    
    # Length statistics
    pred_lengths = [len(p.split()) for p in predictions]
    target_lengths = [len(t.split()) for t in targets]
    
    length_stats = {
        'avg_pred_length': np.mean(pred_lengths),
        'avg_target_length': np.mean(target_lengths),
        'max_pred_length': max(pred_lengths),
        'max_target_length': max(target_lengths)
    }
    
    # Word frequency analysis
    pred_words = Counter([w for p in predictions for w in p.split()])
    target_words = Counter([w for t in targets for w in t.split()])
    
    # Error analysis
    errors = []
    for pred, target in zip(predictions, targets):
        if pred != target:
            errors.append({
                'prediction': pred,
                'target': target
            })
    
    return {
        'length_stats': length_stats,
        'common_pred_words': pred_words.most_common(10),
        'common_target_words': target_words.most_common(10),
        'error_examples': errors[:10]  # First 10 errors
    }

def main():
    parser = argparse.ArgumentParser(description='Monitor and analyze AVSR training')
    parser.add_argument('--log-dir', type=str, default='logs/avsr_logs',
                      help='Directory containing TensorBoard logs')
    parser.add_argument('--results-dir', type=str, default='results',
                      help='Directory containing prediction results')
    parser.add_argument('--output-dir', type=str, default='analysis',
                      help='Directory to save analysis results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and plot training metrics
    print("Loading training metrics...")
    metrics = load_tensorboard_data(args.log_dir)
    plot_training_curves(metrics, args.output_dir)
    
    # Analyze predictions if available
    if os.path.exists(args.results_dir):
        print("Analyzing predictions...")
        predictions, targets = analyze_predictions(args.results_dir)
        stats = compute_statistics(predictions, targets)
        
        # Save statistics
        with open(os.path.join(args.output_dir, 'analysis.txt'), 'w') as f:
            f.write("=== Length Statistics ===\n")
            for k, v in stats['length_stats'].items():
                f.write(f"{k}: {v:.2f}\n")
            
            f.write("\n=== Common Predicted Words ===\n")
            for word, count in stats['common_pred_words']:
                f.write(f"{word}: {count}\n")
            
            f.write("\n=== Common Target Words ===\n")
            for word, count in stats['common_target_words']:
                f.write(f"{word}: {count}\n")
            
            f.write("\n=== Error Examples ===\n")
            for i, error in enumerate(stats['error_examples'], 1):
                f.write(f"\nError {i}:\n")
                f.write(f"Prediction: {error['prediction']}\n")
                f.write(f"Target: {error['target']}\n")
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()

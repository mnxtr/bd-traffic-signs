#!/usr/bin/env python3
"""
SSD Training Results Visualizer
Plots training and validation loss from training_history.json
"""

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

def plot_ssd_results(json_path, output_dir=None, show=False):
    """
    Plot SSD training results.
    
    Args:
        json_path: Path to training_history.json
        output_dir: Directory to save plots
        show: Whether to show plots
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"❌ Error: File not found: {json_path}")
        return

    # Load history
    with open(json_path, 'r') as f:
        history = json.load(f)
    
    # Set output directory
    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('SSD MobileNet Training Results', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss Plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Time
    ax2.plot(epochs, history['epoch_times'], 'g-', label='Epoch Time', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Time (seconds)', fontsize=11)
    ax2.set_title('Training Time per Epoch', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'ssd_training_results_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize SSD Training Results')
    parser.add_argument('--history', type=str, 
                        default='results/ssd_bd_signs/training_history.json',
                        help='Path to training_history.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--show', action='store_true',
                        help='Show plots')
    
    args = parser.parse_args()
    
    plot_ssd_results(args.history, args.output, args.show)

if __name__ == '__main__':
    main()

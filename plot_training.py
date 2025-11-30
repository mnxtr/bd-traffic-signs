#!/usr/bin/env python3
"""
Training Progress Visualization Script
Generates comprehensive training graphs from YOLOv11 results.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

def plot_training_metrics(csv_path, output_dir=None, show=False):
    """
    Generate training visualization graphs from results.csv
    
    Args:
        csv_path: Path to results.csv file
        output_dir: Directory to save plots (defaults to same dir as csv)
        show: Display plots interactively
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create comprehensive figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('YOLOv11 Training Progress - BD Traffic Signs', fontsize=16, fontweight='bold')
    
    # 1. Training Losses
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train/box_loss'], 'b-', label='Box Loss', linewidth=2)
    ax1.plot(df['epoch'], df['train/cls_loss'], 'r-', label='Class Loss', linewidth=2)
    ax1.plot(df['epoch'], df['train/dfl_loss'], 'g-', label='DFL Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Losses', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Losses
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['val/box_loss'], 'b-', label='Val Box Loss', linewidth=2)
    ax2.plot(df['epoch'], df['val/cls_loss'], 'r-', label='Val Class Loss', linewidth=2)
    ax2.plot(df['epoch'], df['val/dfl_loss'], 'g-', label='Val DFL Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Validation Losses', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision & Recall
    ax3 = axes[0, 2]
    ax3.plot(df['epoch'], df['metrics/precision(B)'], 'b-', label='Precision', linewidth=2, marker='o')
    ax3.plot(df['epoch'], df['metrics/recall(B)'], 'r-', label='Recall', linewidth=2, marker='s')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Precision & Recall', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.05])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. mAP Metrics
    ax4 = axes[1, 0]
    ax4.plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', label='mAP@50', linewidth=2, marker='o')
    ax4.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'r-', label='mAP@50-95', linewidth=2, marker='s')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('mAP Score', fontsize=11)
    ax4.set_title('Mean Average Precision (mAP)', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 1.05])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Learning Rate
    ax5 = axes[1, 1]
    ax5.plot(df['epoch'], df['lr/pg0'], 'b-', label='LR pg0', linewidth=2)
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Learning Rate', fontsize=11)
    ax5.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Training Time
    ax6 = axes[1, 2]
    ax6.plot(df['epoch'], df['time'] / 3600, 'purple', linewidth=2, marker='o')
    ax6.set_xlabel('Epoch', fontsize=11)
    ax6.set_ylabel('Cumulative Time (hours)', fontsize=11)
    ax6.set_title('Training Time Progress', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'training_metrics_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to: {output_file}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    # Generate summary statistics
    latest_epoch = df.iloc[-1]
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY (Latest Epoch)")
    print("="*60)
    print(f"Epoch:              {int(latest_epoch['epoch'])}")
    print(f"Training Time:      {latest_epoch['time']/3600:.2f} hours")
    print(f"\nTraining Losses:")
    print(f"  Box Loss:         {latest_epoch['train/box_loss']:.4f}")
    print(f"  Class Loss:       {latest_epoch['train/cls_loss']:.4f}")
    print(f"  DFL Loss:         {latest_epoch['train/dfl_loss']:.4f}")
    print(f"\nValidation Losses:")
    print(f"  Box Loss:         {latest_epoch['val/box_loss']:.4f}")
    print(f"  Class Loss:       {latest_epoch['val/cls_loss']:.4f}")
    print(f"  DFL Loss:         {latest_epoch['val/dfl_loss']:.4f}")
    print(f"\nMetrics:")
    print(f"  Precision:        {latest_epoch['metrics/precision(B)']:.4f}")
    print(f"  Recall:           {latest_epoch['metrics/recall(B)']:.4f}")
    print(f"  mAP@50:           {latest_epoch['metrics/mAP50(B)']:.4f}")
    print(f"  mAP@50-95:        {latest_epoch['metrics/mAP50-95(B)']:.4f}")
    print("="*60 + "\n")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Plot YOLOv11 training metrics')
    parser.add_argument('--csv', type=str, 
                       default='results/yolov11_bd_signs_20251122_192224/results.csv',
                       help='Path to results.csv file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for plots (default: same as csv)')
    parser.add_argument('--show', action='store_true',
                       help='Display plots interactively')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        return
    
    print(f"📈 Generating training plots from: {csv_path}")
    plot_training_metrics(csv_path, args.output, args.show)

if __name__ == '__main__':
    main()

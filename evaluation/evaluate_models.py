#!/usr/bin/env python3
"""
Model Evaluation and Comparison Script
Compares YOLOv11 and SSD models for Bangladeshi Road Sign Detection
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import cv2
from ultralytics import YOLO

class ModelEvaluator:
    """
    Evaluator for comparing object detection models.
    """
    
    def __init__(self, test_images_dir: str, test_labels_dir: str, class_names: List[str]):
        self.test_images_dir = Path(test_images_dir)
        self.test_labels_dir = Path(test_labels_dir)
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def evaluate_yolov11(
        self, 
        model_path: str, 
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict:
        """
        Evaluate YOLOv11 model.
        
        Args:
            model_path: Path to trained YOLOv11 model
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"🔍 Evaluating YOLOv11 model: {model_path}")
        
        model = YOLO(model_path)
        
        # Run validation
        metrics = model.val(
            data=self.test_labels_dir.parent / 'data.yaml',
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Measure inference speed
        test_images = list(self.test_images_dir.glob('*'))[:100]  # Sample 100 images
        start_time = time.time()
        for img_path in test_images:
            _ = model.predict(str(img_path), conf=conf_threshold, verbose=False)
        inference_time = (time.time() - start_time) / len(test_images)
        fps = 1.0 / inference_time
        
        results = {
            'model_name': 'YOLOv11',
            'map50': float(metrics.box.map50),
            'map50_95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'inference_time_ms': inference_time * 1000,
            'fps': fps,
            'model_size_mb': Path(model_path).stat().st_size / (1024 * 1024)
        }
        
        print(f"   mAP50: {results['map50']:.4f}")
        print(f"   mAP50-95: {results['map50_95']:.4f}")
        print(f"   FPS: {results['fps']:.2f}")
        
        return results
    
    def evaluate_ssd(
        self,
        model_path: str,
        device: str = 'cpu',
        conf_threshold: float = 0.25
    ) -> Dict:
        """
        Evaluate SSD model.
        
        Args:
            model_path: Path to trained SSD model
            device: Device to run inference on
            conf_threshold: Confidence threshold
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"🔍 Evaluating SSD model: {model_path}")
        
        # Keep the original string parameter and construct a torch.device into a new variable
        device_str = device
        torch_device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=torch_device)
        # Note: Model loading needs to be implemented based on saved format
        
        print("⚠️  SSD evaluation implementation pending - requires custom dataset loader")
        
        # Placeholder results
        results = {
            'model_name': 'SSD',
            'map50': 0.0,
            'map50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'inference_time_ms': 0.0,
            'fps': 0.0,
            'model_size_mb': Path(model_path).stat().st_size / (1024 * 1024) if Path(model_path).exists() else 0.0
        }
        
        return results
    
    def compare_models(
        self,
        yolo_results: Dict,
        ssd_results: Dict,
        output_dir: str = '../results/comparison'
    ):
        """
        Generate comparison visualizations and reports.
        
        Args:
            yolo_results: YOLOv11 evaluation results
            ssd_results: SSD evaluation results
            output_dir: Directory to save comparison results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n📊 Generating comparison report...")
        
        # Create comparison table
        comparison = {
            'Metric': ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 
                      'Inference Time (ms)', 'FPS', 'Model Size (MB)'],
            'YOLOv11': [
                yolo_results['map50'],
                yolo_results['map50_95'],
                yolo_results['precision'],
                yolo_results['recall'],
                yolo_results['inference_time_ms'],
                yolo_results['fps'],
                yolo_results['model_size_mb']
            ],
            'SSD': [
                ssd_results['map50'],
                ssd_results['map50_95'],
                ssd_results['precision'],
                ssd_results['recall'],
                ssd_results['inference_time_ms'],
                ssd_results['fps'],
                ssd_results['model_size_mb']
            ]
        }
        
        # Save comparison to JSON
        with open(output_dir / 'comparison.json', 'w') as f:
            json.dump({
                'yolov11': yolo_results,
                'ssd': ssd_results,
                'comparison_table': comparison
            }, f, indent=4)
        
        # Generate visualizations
        self._plot_comparison_charts(comparison, output_dir)
        
        # Print comparison table
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(f"{'Metric':<25} {'YOLOv11':<15} {'SSD':<15}")
        print("-"*60)
        for i, metric in enumerate(comparison['Metric']):
            yolo_val = comparison['YOLOv11'][i]
            ssd_val = comparison['SSD'][i]
            print(f"{metric:<25} {yolo_val:<15.4f} {ssd_val:<15.4f}")
        print("="*60)
        
        print(f"\n✅ Comparison report saved to {output_dir}")
    
    def _plot_comparison_charts(self, comparison: Dict, output_dir: Path):
        """Generate comparison charts."""
        sns.set_style('whitegrid')
        
        # Accuracy metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # mAP comparison
        metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            metric_idx = comparison['Metric'].index(metric)
            values = [comparison['YOLOv11'][metric_idx], comparison['SSD'][metric_idx]]
            ax.bar(['YOLOv11', 'SSD'], values, color=['#4CAF50', '#2196F3'])
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Speed and size comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # FPS comparison
        fps_idx = comparison['Metric'].index('FPS')
        fps_values = [comparison['YOLOv11'][fps_idx], comparison['SSD'][fps_idx]]
        axes[0].bar(['YOLOv11', 'SSD'], fps_values, color=['#FF9800', '#9C27B0'])
        axes[0].set_ylabel('FPS')
        axes[0].set_title('Inference Speed (FPS)')
        
        # Inference time comparison
        time_idx = comparison['Metric'].index('Inference Time (ms)')
        time_values = [comparison['YOLOv11'][time_idx], comparison['SSD'][time_idx]]
        axes[1].bar(['YOLOv11', 'SSD'], time_values, color=['#FF9800', '#9C27B0'])
        axes[1].set_ylabel('Time (ms)')
        axes[1].set_title('Inference Time per Image')
        
        # Model size comparison
        size_idx = comparison['Metric'].index('Model Size (MB)')
        size_values = [comparison['YOLOv11'][size_idx], comparison['SSD'][size_idx]]
        axes[2].bar(['YOLOv11', 'SSD'], size_values, color=['#FF5722', '#3F51B5'])
        axes[2].set_ylabel('Size (MB)')
        axes[2].set_title('Model Size')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Saved comparison charts")

def main():
    parser = argparse.ArgumentParser(description='Evaluate and Compare Models')
    parser.add_argument('--test-images', type=str, required=True,
                        help='Path to test images directory')
    parser.add_argument('--test-labels', type=str, required=True,
                        help='Path to test labels directory')
    parser.add_argument('--classes', type=str, nargs='+', required=True,
                        help='List of class names')
    parser.add_argument('--yolo-model', type=str, required=True,
                        help='Path to trained YOLOv11 model')
    parser.add_argument('--ssd-model', type=str,
                        help='Path to trained SSD model')
    parser.add_argument('--output-dir', type=str, default='../results/comparison',
                        help='Output directory for comparison results')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for inference')
    
    args = parser.parse_args()
    
    print("🚀 Starting model evaluation and comparison...")
    
    evaluator = ModelEvaluator(
        test_images_dir=args.test_images,
        test_labels_dir=args.test_labels,
        class_names=args.classes
    )
    
    # Evaluate YOLOv11
    yolo_results = evaluator.evaluate_yolov11(args.yolo_model)
    
    # Evaluate SSD if model path provided
    ssd_results = None
    if args.ssd_model:
        ssd_results = evaluator.evaluate_ssd(args.ssd_model, args.device)
    
    # Compare models
    if ssd_results:
        evaluator.compare_models(yolo_results, ssd_results, args.output_dir)
    
    print("\n🎉 Evaluation completed successfully!")

if __name__ == '__main__':
    main()

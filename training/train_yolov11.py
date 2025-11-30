#!/usr/bin/env python3
"""
YOLOv11 Training Script for Bangladeshi Road Sign Detection
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

def train_yolov11(
    data_yaml: str,
    model_variant: str = 'yolo11n.pt',
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = 'cpu',
    project: str = '../results',
    name: str = 'yolov11_bd_signs',
    patience: int = 50,
    save_period: int = 10
):
    """
    Train YOLOv11 model for traffic sign detection.
    
    Args:
        data_yaml: Path to data.yaml configuration file
        model_variant: YOLOv11 variant (yolo11n/s/m/l/x.pt)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        device: Device to train on ('cpu' or 'cuda')
        project: Project directory for results
        name: Name of the training run
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
    """
    
    print(f"🚀 Starting YOLOv11 Training")
    print(f"   Model: {model_variant}")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {img_size}")
    print("-" * 50)
    
    # Check if GPU is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Load YOLOv11 model
    model = YOLO(model_variant)
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save_period=save_period,
        plots=True,
        val=True,
        save=True,
        exist_ok=True,
        verbose=True,
        pretrained=True,
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    
    print("\n✅ Training completed!")
    print(f"📁 Results saved to: {project}/{name}")
    
    # Validate the model
    print("\n🔍 Running validation...")
    metrics = model.val()
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    
    return model, results

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 for BD Traffic Signs')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolo11n.pt', 
                        choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 
                                'yolo11l.pt', 'yolo11x.pt'],
                        help='YOLOv11 model variant')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--project', type=str, default='../results', help='Project directory')
    parser.add_argument('--name', type=str, default='yolov11_bd_signs', help='Run name')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Train the model
    model, results = train_yolov11(
        data_yaml=args.data,
        model_variant=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience
    )
    
    print("\n🎉 Training pipeline completed successfully!")

if __name__ == '__main__':
    main()

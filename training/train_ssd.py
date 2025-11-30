#!/usr/bin/env python3
"""
SSD (Single Shot MultiBox Detector) Training Script for Bangladeshi Road Sign Detection
Adapted for BRSSD (Bangladesh Road Sign SSD)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import json
from pathlib import Path
import time
from datetime import datetime

class SSDTrainer:
    """
    Trainer class for SSD models.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        num_classes,
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=0.0005,
        output_dir='../results/ssd_bd_signs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': []
        }
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        print(f"\n📊 Epoch {epoch + 1}")
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            
            total_loss += losses.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                      f"Loss: {losses.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, num_epochs, save_interval=10):
        """
        Full training loop.
        """
        print(f"🚀 Starting SSD Training")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Number of classes: {self.num_classes}")
        print("-" * 50)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            epoch_time = time.time() - start_time
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch_times'].append(epoch_time)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"\n   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"   ✅ Best model saved (val_loss: {val_loss:.4f})")
            
            # Periodic checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch + 1}.pth')
        
        # Save final model
        self.save_checkpoint(num_epochs - 1, 'final_model.pth')
        self.save_training_history()
        
        print("\n✅ Training completed!")
        print(f"📁 Results saved to: {self.output_dir}")
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def save_training_history(self):
        """Save training history to JSON."""
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=4)

def create_ssd_model(backbone='mobilenet', num_classes=91, pretrained=True):
    """
    Create SSD model with specified backbone.
    
    Args:
        backbone: 'mobilenet' or 'vgg'
        num_classes: Number of classes (including background)
        pretrained: Use pretrained weights
    """
    if backbone == 'mobilenet':
        if pretrained:
            model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        else:
            model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes)
    elif backbone == 'vgg':
        if pretrained:
            model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        else:
            model = ssd300_vgg16(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    # Modify the number of classes if needed
    # Note: This is a simplified version. Proper implementation requires
    # modifying the classification head based on the specific architecture.
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train SSD for BD Traffic Signs')
    parser.add_argument('--data-root', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--backbone', type=str, default='mobilenet', 
                        choices=['mobilenet', 'vgg'], help='Backbone architecture')
    parser.add_argument('--num-classes', type=int, default=10, 
                        help='Number of classes (excluding background)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--output-dir', type=str, default='../results/ssd_bd_signs', 
                        help='Output directory')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    
    args = parser.parse_args()
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
    
    print(f"🔧 Creating SSD model with {args.backbone} backbone...")
    
    # Create model (num_classes + 1 for background)
    model = create_ssd_model(
        backbone=args.backbone,
        num_classes=args.num_classes + 1,
        pretrained=args.pretrained
    )
    
    # Note: Data loading needs to be implemented based on your dataset format
    # This is a placeholder - you'll need to implement custom dataset classes
    print("\n⚠️  Note: Please implement custom dataset loaders in the data preprocessing script")
    print("   The train_loader and val_loader need to be created from your dataset")
    
    # Placeholder for data loaders
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print("\n🎯 Model created successfully!")
    print(f"   Backbone: {args.backbone}")
    print(f"   Classes: {args.num_classes}")
    print(f"   Device: {device}")
    
    # Uncomment when data loaders are ready:
    # trainer = SSDTrainer(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     device=device,
    #     num_classes=args.num_classes + 1,
    #     learning_rate=args.lr,
    #     output_dir=args.output_dir
    # )
    # trainer.train(num_epochs=args.epochs)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Data Preprocessing Utilities for Bangladeshi Road Sign Dataset
Handles data splitting, format conversion, and augmentation
"""

import os
import shutil
import argparse
import yaml
import json
from pathlib import Path
from typing import List, Tuple, Dict
import random
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

class DatasetPreprocessor:
    """
    Preprocessor for traffic sign detection datasets.
    """
    
    def __init__(self, raw_data_dir: str, output_dir: str, class_names: List[str]):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def split_dataset(
        self, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.2, 
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        random.seed(seed)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.raw_data_dir.glob(f'**/*{ext}'))
            image_files.extend(self.raw_data_dir.glob(f'**/*{ext.upper()}'))
        
        if not image_files:
            print(f"⚠️  No images found in {self.raw_data_dir}")
            return
        
        print(f"📊 Found {len(image_files)} images")
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Split into train, val, test
        train_size = int(len(image_files) * train_ratio)
        val_size = int(len(image_files) * val_ratio)
        
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]
        
        print(f"   Train: {len(train_files)} images")
        print(f"   Val: {len(val_files)} images")
        print(f"   Test: {len(test_files)} images")
        
        # Create split directories
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            split_dir = self.output_dir / split_name
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            for img_file in files:
                # Copy image
                dst_img = images_dir / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Copy corresponding label if exists
                label_file = img_file.with_suffix('.txt')
                if label_file.exists():
                    dst_label = labels_dir / label_file.name
                    shutil.copy2(label_file, dst_label)
        
        print(f"✅ Dataset split completed")
        print(f"📁 Output directory: {self.output_dir}")
        
        return splits
    
    def create_yolo_yaml(self, output_path: str = None):
        """
        Create data.yaml configuration file for YOLO training.
        
        Args:
            output_path: Path to save the yaml file
        """
        if output_path is None:
            output_path = self.output_dir / 'data.yaml'
        
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': self.num_classes,
            'names': self.class_names
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"✅ Created data.yaml at {output_path}")
        return output_path
    
    def convert_to_coco_format(self, split: str = 'train'):
        """
        Convert YOLO format annotations to COCO format.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
        """
        split_dir = self.output_dir / split
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        coco_format = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for idx, class_name in enumerate(self.class_names):
            coco_format['categories'].append({
                'id': idx,
                'name': class_name,
                'supercategory': 'traffic_sign'
            })
        
        annotation_id = 0
        
        # Process images and annotations
        for img_id, img_file in enumerate(sorted(images_dir.glob('*'))):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            # Read image to get dimensions
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            height, width = img.shape[:2]
            
            # Add image info
            coco_format['images'].append({
                'id': img_id,
                'file_name': img_file.name,
                'width': width,
                'height': height
            })
            
            # Read YOLO format annotations
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        bbox_width = float(parts[3]) * width
                        bbox_height = float(parts[4]) * height
                        
                        # Convert to COCO format (x, y, width, height)
                        x_min = x_center - bbox_width / 2
                        y_min = y_center - bbox_height / 2
                        
                        coco_format['annotations'].append({
                            'id': annotation_id,
                            'image_id': img_id,
                            'category_id': class_id,
                            'bbox': [x_min, y_min, bbox_width, bbox_height],
                            'area': bbox_width * bbox_height,
                            'iscrowd': 0
                        })
                        annotation_id += 1
        
        # Save COCO format
        coco_file = split_dir / f'{split}_coco.json'
        with open(coco_file, 'w') as f:
            json.dump(coco_format, f, indent=4)
        
        print(f"✅ Created COCO format annotations at {coco_file}")
        return coco_file
    
    def augment_dataset(self, split: str = 'train', augmentation_factor: int = 3):
        """
        Apply data augmentation to increase dataset size.
        
        Args:
            split: Dataset split to augment
            augmentation_factor: Number of augmented versions per image
        """
        split_dir = self.output_dir / split
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        aug_images_dir = split_dir / 'images_augmented'
        aug_labels_dir = split_dir / 'labels_augmented'
        aug_images_dir.mkdir(parents=True, exist_ok=True)
        aug_labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Augmenting {split} dataset...")
        
        image_files = list(images_dir.glob('*'))
        for img_file in image_files:
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Read labels
            label_file = labels_dir / f"{img_file.stem}.txt"
            labels = []
            if label_file.exists():
                with open(label_file, 'r') as f:
                    labels = [line.strip() for line in f.readlines()]
            
            # Apply augmentations
            for aug_idx in range(augmentation_factor):
                aug_img = self._apply_augmentation(img)
                
                # Save augmented image
                aug_img_name = f"{img_file.stem}_aug{aug_idx}{img_file.suffix}"
                aug_img_path = aug_images_dir / aug_img_name
                cv2.imwrite(str(aug_img_path), aug_img)
                
                # Copy labels (assuming bounding boxes remain valid)
                if labels:
                    aug_label_path = aug_labels_dir / f"{img_file.stem}_aug{aug_idx}.txt"
                    with open(aug_label_path, 'w') as f:
                        f.write('\n'.join(labels))
        
        print(f"✅ Augmentation completed")
        print(f"📁 Augmented data saved to {aug_images_dir}")
    
    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentations to an image."""
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        # Random contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            img = np.clip(alpha * img, 0, 255).astype(np.uint8)
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # Random rotation (small angle)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))
        
        # Add Gaussian noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return img

def main():
    parser = argparse.ArgumentParser(description='Preprocess BD Traffic Sign Dataset')
    parser.add_argument('--raw-dir', type=str, required=True, 
                        help='Directory containing raw images and annotations')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for processed dataset')
    parser.add_argument('--classes', type=str, nargs='+', required=True,
                        help='List of class names')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Test set ratio')
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation')
    parser.add_argument('--coco-format', action='store_true',
                        help='Convert to COCO format')
    
    args = parser.parse_args()
    
    print("🚀 Starting dataset preprocessing...")
    print(f"   Raw data: {args.raw_dir}")
    print(f"   Output: {args.output_dir}")
    print(f"   Classes: {args.classes}")
    
    preprocessor = DatasetPreprocessor(
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir,
        class_names=args.classes
    )
    
    # Split dataset
    splits = preprocessor.split_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Create YOLO yaml
    preprocessor.create_yolo_yaml()
    
    # Only proceed with format conversion and augmentation if images were found
    if splits:
        # Convert to COCO format if requested
        if args.coco_format:
            for split in ['train', 'val', 'test']:
                preprocessor.convert_to_coco_format(split)
        
        # Apply augmentation if requested
        if args.augment:
            preprocessor.augment_dataset(split='train')
    else:
        print("⚠️  Skipping augmentation and format conversion - no images to process")
    
    print("\n🎉 Preprocessing completed successfully!")

if __name__ == '__main__':
    main()

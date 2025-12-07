#!/usr/bin/env python3
"""
Ensemble Detector for Traffic Sign Detection

Combines predictions from multiple models (YOLOv11, SSD) using Weighted Box Fusion
to achieve more robust and accurate detections.

Features:
- Multi-model ensemble (YOLOv11 + SSD)
- Multi-scale inference for robust detection
- Weighted Box Fusion for prediction merging
- Confidence calibration across models
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import time

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.box_fusion import weighted_box_fusion, non_max_suppression, soft_nms


@dataclass
class Detection:
    """Single detection result."""
    box: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


@dataclass 
class DetectionResult:
    """Complete detection result for an image."""
    detections: List[Detection]
    image: np.ndarray
    inference_time: float
    model_name: str
    

class YOLOv11Detector:
    """YOLOv11 detector wrapper."""
    
    def __init__(
        self, 
        model_path: str = "results/bd_signs_v1/weights/best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        """
        Initialize YOLOv11 detector.
        
        Args:
            model_path: Path to trained model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cpu' or 'cuda')
        """
        from ultralytics import YOLO
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        print(f"🔧 Loading YOLOv11 model from: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"   ✅ Loaded with {len(self.class_names)} classes")
    
    def predict(
        self, 
        image: np.ndarray, 
        conf: Optional[float] = None,
        img_size: int = 640
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on an image.
        
        Args:
            image: Input image (BGR or RGB)
            conf: Override confidence threshold
            img_size: Input image size for model
            
        Returns:
            boxes: (N, 4) array of [x1, y1, x2, y2]
            scores: (N,) array of confidence scores
            labels: (N,) array of class IDs
        """
        conf = conf or self.conf_threshold
        
        results = self.model(
            image, 
            conf=conf, 
            iou=self.iou_threshold,
            imgsz=img_size,
            device=self.device,
            verbose=False
        )[0]
        
        boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else np.array([]).reshape(0, 4)
        scores = results.boxes.conf.cpu().numpy() if len(results.boxes) > 0 else np.array([])
        labels = results.boxes.cls.cpu().numpy().astype(int) if len(results.boxes) > 0 else np.array([])
        
        return boxes, scores, labels
    
    def get_annotated_image(self, image: np.ndarray, conf: Optional[float] = None) -> np.ndarray:
        """Run inference and return annotated image."""
        conf = conf or self.conf_threshold
        results = self.model(image, conf=conf, iou=self.iou_threshold, verbose=False)[0]
        return results.plot()


class SSDDetector:
    """SSD detector wrapper (placeholder for when SSD is trained)."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        device: str = "cpu",
        num_classes: int = 29
    ):
        """
        Initialize SSD detector.
        
        Args:
            model_path: Path to trained model weights (optional)
            conf_threshold: Confidence threshold
            device: Device to run on
            num_classes: Number of classes (excluding background)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.model = None
        self.class_names = None
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("⚠️  SSD model not available - using placeholder")
    
    def _load_model(self, model_path: str):
        """Load SSD model from checkpoint."""
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
        
        print(f"🔧 Loading SSD model from: {model_path}")
        self.model = ssdlite320_mobilenet_v3_large(
            weights=None, 
            num_classes=self.num_classes + 1
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print("   ✅ SSD model loaded")
    
    def predict(
        self, 
        image: np.ndarray,
        conf: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on an image.
        
        Returns empty arrays if model is not loaded.
        """
        if self.model is None:
            return np.array([]).reshape(0, 4), np.array([]), np.array([])
        
        conf = conf or self.conf_threshold
        
        # Preprocess
        import torchvision.transforms.functional as F
        
        img_tensor = F.to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
        
        # Filter by confidence
        mask = predictions['scores'] >= conf
        boxes = predictions['boxes'][mask].cpu().numpy()
        scores = predictions['scores'][mask].cpu().numpy()
        labels = predictions['labels'][mask].cpu().numpy() - 1  # Remove background offset
        
        return boxes, scores, labels
    
    @property
    def is_available(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None


class EnsembleDetector:
    """
    Ensemble detector combining multiple models for robust traffic sign detection.
    
    Supports:
    - YOLOv11 (primary model)
    - SSD (secondary model, if available)
    - Multi-scale inference
    - Weighted Box Fusion for prediction merging
    """
    
    def __init__(
        self,
        yolo_path: str = "results/bd_signs_v1/weights/best.pt",
        ssd_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "cpu",
        use_multi_scale: bool = True,
        scales: List[int] = [480, 640, 800],
        model_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ensemble detector.
        
        Args:
            yolo_path: Path to YOLOv11 weights
            ssd_path: Path to SSD weights (optional)
            conf_threshold: Minimum confidence threshold
            iou_threshold: IoU threshold for box fusion
            device: Device for inference
            use_multi_scale: Whether to use multi-scale inference
            scales: Input sizes for multi-scale inference
            model_weights: Dictionary of weights for each model
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.use_multi_scale = use_multi_scale
        self.scales = scales
        
        # Initialize detectors
        self.yolo = YOLOv11Detector(
            model_path=yolo_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
        
        self.ssd = SSDDetector(
            model_path=ssd_path,
            conf_threshold=conf_threshold,
            device=device
        )
        
        # Get class names from YOLO model
        self.class_names = self.yolo.class_names
        self.num_classes = len(self.class_names)
        
        # Model weights for fusion
        self.model_weights = model_weights or {
            'yolo': 1.0,
            'yolo_multi_scale': 0.8,
            'ssd': 0.7
        }
        
        print(f"\n🎯 Ensemble Detector initialized")
        print(f"   Models: YOLOv11 ✓, SSD {'✓' if self.ssd.is_available else '✗'}")
        print(f"   Multi-scale: {self.use_multi_scale} ({self.scales if self.use_multi_scale else 'N/A'})")
        print(f"   Classes: {self.num_classes}")
    
    def predict(
        self, 
        image: np.ndarray,
        mode: str = "ensemble"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Run ensemble prediction on an image.
        
        Args:
            image: Input image (BGR)
            mode: Detection mode ('yolo', 'ssd', 'ensemble', 'multi_scale')
            
        Returns:
            boxes: (N, 4) array of [x1, y1, x2, y2]
            scores: (N,) array of confidence scores
            labels: (N,) array of class IDs
            inference_time: Total inference time in seconds
        """
        start_time = time.time()
        h, w = image.shape[:2]
        
        if mode == "yolo":
            boxes, scores, labels = self.yolo.predict(image)
            
        elif mode == "ssd":
            if not self.ssd.is_available:
                print("⚠️  SSD not available, falling back to YOLO")
                boxes, scores, labels = self.yolo.predict(image)
            else:
                boxes, scores, labels = self.ssd.predict(image)
                
        elif mode == "multi_scale":
            boxes, scores, labels = self._multi_scale_inference(image)
            
        elif mode == "ensemble":
            boxes, scores, labels = self._ensemble_inference(image)
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'yolo', 'ssd', 'multi_scale', or 'ensemble'")
        
        inference_time = time.time() - start_time
        
        return boxes, scores, labels, inference_time
    
    def _multi_scale_inference(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run YOLO at multiple scales and fuse predictions.
        
        This improves detection of both small and large signs.
        """
        h, w = image.shape[:2]
        
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        
        for scale in self.scales:
            # Run inference at this scale
            boxes, scores, labels = self.yolo.predict(image, img_size=scale)
            
            if len(boxes) > 0:
                # Normalize boxes to [0, 1]
                normalized_boxes = boxes.copy()
                normalized_boxes[:, [0, 2]] /= w
                normalized_boxes[:, [1, 3]] /= h
                
                boxes_list.append(normalized_boxes)
                scores_list.append(scores)
                labels_list.append(labels)
                weights.append(self.model_weights.get('yolo_multi_scale', 0.8))
        
        if len(boxes_list) == 0:
            return np.array([]).reshape(0, 4), np.array([]), np.array([])
        
        # Weighted Box Fusion
        fused_boxes, fused_scores, fused_labels = weighted_box_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights,
            iou_threshold=self.iou_threshold
        )
        
        # Denormalize boxes
        if len(fused_boxes) > 0:
            fused_boxes[:, [0, 2]] *= w
            fused_boxes[:, [1, 3]] *= h
        
        return fused_boxes, fused_scores, fused_labels
    
    def _ensemble_inference(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run full ensemble inference with all available models.
        """
        h, w = image.shape[:2]
        
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        
        # YOLO predictions (primary)
        yolo_boxes, yolo_scores, yolo_labels = self.yolo.predict(image)
        if len(yolo_boxes) > 0:
            normalized = yolo_boxes.copy()
            normalized[:, [0, 2]] /= w
            normalized[:, [1, 3]] /= h
            boxes_list.append(normalized)
            scores_list.append(yolo_scores)
            labels_list.append(yolo_labels)
            weights.append(self.model_weights.get('yolo', 1.0))
        
        # Multi-scale YOLO (if enabled)
        if self.use_multi_scale:
            for scale in self.scales:
                if scale == 640:  # Skip default scale
                    continue
                scale_boxes, scale_scores, scale_labels = self.yolo.predict(image, img_size=scale)
                if len(scale_boxes) > 0:
                    normalized = scale_boxes.copy()
                    normalized[:, [0, 2]] /= w
                    normalized[:, [1, 3]] /= h
                    boxes_list.append(normalized)
                    scores_list.append(scale_scores)
                    labels_list.append(scale_labels)
                    weights.append(self.model_weights.get('yolo_multi_scale', 0.8))
        
        # SSD predictions (if available)
        if self.ssd.is_available:
            ssd_boxes, ssd_scores, ssd_labels = self.ssd.predict(image)
            if len(ssd_boxes) > 0:
                normalized = ssd_boxes.copy()
                normalized[:, [0, 2]] /= w
                normalized[:, [1, 3]] /= h
                boxes_list.append(normalized)
                scores_list.append(ssd_scores)
                labels_list.append(ssd_labels)
                weights.append(self.model_weights.get('ssd', 0.7))
        
        if len(boxes_list) == 0:
            return np.array([]).reshape(0, 4), np.array([]), np.array([])
        
        # Weighted Box Fusion
        fused_boxes, fused_scores, fused_labels = weighted_box_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights,
            iou_threshold=self.iou_threshold
        )
        
        # Denormalize
        if len(fused_boxes) > 0:
            fused_boxes[:, [0, 2]] *= w
            fused_boxes[:, [1, 3]] *= h
        
        return fused_boxes, fused_scores, fused_labels
    
    def draw_detections(
        self, 
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        line_width: int = 2
    ) -> np.ndarray:
        """
        Draw detection boxes on an image.
        
        Args:
            image: Input image (BGR)
            boxes, scores, labels: Detection results
            line_width: Line width for boxes
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Color palette
        colors = [
            (255, 107, 107),  # Red
            (78, 205, 196),   # Teal
            (255, 179, 71),   # Orange
            (170, 166, 255),  # Purple
            (100, 255, 100),  # Green
            (255, 100, 255),  # Magenta
            (100, 200, 255),  # Sky blue
            (255, 255, 100),  # Yellow
        ]
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            color = colors[int(label) % len(colors)]
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)
            
            # Label background
            class_name = self.class_names.get(int(label), f"Class {label}")
            label_text = f"{class_name}: {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(annotated, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 2, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def get_detection_summary(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Get a summary of detections.
        
        Returns:
            Dictionary with detection statistics
        """
        summary = {
            'total_detections': len(boxes),
            'classes_detected': [],
            'avg_confidence': float(scores.mean()) if len(scores) > 0 else 0.0,
            'max_confidence': float(scores.max()) if len(scores) > 0 else 0.0,
            'min_confidence': float(scores.min()) if len(scores) > 0 else 0.0,
        }
        
        if len(labels) > 0:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                class_name = self.class_names.get(int(label), f"Class {label}")
                count = (labels == label).sum()
                summary['classes_detected'].append({
                    'class_id': int(label),
                    'class_name': class_name,
                    'count': int(count)
                })
        
        return summary


def test_ensemble():
    """Test the ensemble detector with a sample image."""
    print("\n" + "="*60)
    print("🧪 Testing Ensemble Detector")
    print("="*60 + "\n")
    
    # Initialize detector
    detector = EnsembleDetector(
        yolo_path="results/bd_signs_v1/weights/best.pt",
        use_multi_scale=True,
        scales=[480, 640, 800]
    )
    
    # Create a test image (gradient for testing)
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    test_image[:, :, 0] = np.linspace(0, 255, 640).astype(np.uint8)
    test_image[:, :, 2] = np.linspace(255, 0, 640).astype(np.uint8)
    
    # Test all modes
    for mode in ["yolo", "multi_scale", "ensemble"]:
        print(f"\n📍 Testing mode: {mode}")
        boxes, scores, labels, time_taken = detector.predict(test_image, mode=mode)
        print(f"   Detections: {len(boxes)}")
        print(f"   Time: {time_taken*1000:.1f}ms")
    
    print("\n✅ Ensemble detector test complete!")


if __name__ == "__main__":
    test_ensemble()

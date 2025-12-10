#!/usr/bin/env python3
"""
Weighted Box Fusion (WBF) Implementation for Ensemble Object Detection

This module implements the WBF algorithm for merging predictions from multiple
object detection models, as well as standard NMS for single model outputs.

Reference: https://arxiv.org/abs/1910.13302
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Non-Maximum Suppression to filter overlapping boxes.

    Args:
        boxes: Array of shape (N, 4) with [x1, y1, x2, y2] format
        scores: Array of shape (N,) with confidence scores
        labels: Array of shape (N,) with class labels
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered boxes, scores, and labels
    """
    if len(boxes) == 0:
        return boxes, scores, labels

    # Sort by confidence score (descending)
    indices = np.argsort(scores)[::-1]
    sorted_boxes = boxes[indices]
    sorted_scores = scores[indices]
    sorted_labels = labels[indices]

    keep = []
    remaining_indices = list(range(len(sorted_boxes)))

    while len(remaining_indices) > 0:
        # Take the first (highest confidence) remaining box
        current_idx = remaining_indices[0]
        keep.append(current_idx)
        remaining_indices = remaining_indices[1:]

        if len(remaining_indices) == 0:
            break

        current_box = sorted_boxes[current_idx]
        current_label = sorted_labels[current_idx]

        # Filter out boxes with high IoU overlap (same class only)
        new_remaining = []
        for idx in remaining_indices:
            if sorted_labels[idx] != current_label:
                new_remaining.append(idx)
            else:
                iou = calculate_iou(current_box, sorted_boxes[idx])
                if iou < iou_threshold:
                    new_remaining.append(idx)
        remaining_indices = new_remaining

    # Return kept detections from sorted arrays
    if len(keep) > 0:
        return sorted_boxes[keep], sorted_scores[keep], sorted_labels[keep]
    else:
        return (np.array([]).reshape(0, 4), np.array([]), np.array([]))


def weighted_box_fusion(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
    iou_threshold: float = 0.55,
    skip_box_threshold: float = 0.0,
    conf_type: str = "avg",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted Box Fusion for merging predictions from multiple models.

    This algorithm clusters boxes from different models based on IoU,
    then computes weighted averages of the box coordinates and confidence scores.

    Args:
        boxes_list: List of box arrays, one per model. Each array has shape (N, 4)
                   with normalized coordinates [x1, y1, x2, y2] in range [0, 1]
        scores_list: List of score arrays, one per model
        labels_list: List of label arrays, one per model
        weights: Optional weights for each model (default: equal weights)
        iou_threshold: IoU threshold for clustering boxes
        skip_box_threshold: Minimum confidence to consider a box
        conf_type: How to aggregate confidence ('avg', 'max', 'box_and_model_avg')

    Returns:
        Fused boxes, scores, and labels
    """
    if len(boxes_list) == 0:
        return np.array([]).reshape(0, 4), np.array([]), np.array([])

    num_models = len(boxes_list)

    if weights is None:
        weights = [1.0] * num_models
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    # Collect all boxes with their metadata
    all_boxes = []
    all_scores = []
    all_labels = []
    model_indices = []

    for model_idx, (boxes, scores, labels) in enumerate(
        zip(boxes_list, scores_list, labels_list)
    ):
        if len(boxes) == 0:
            continue

        # Filter by threshold
        mask = scores >= skip_box_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        all_boxes.append(boxes)
        all_scores.append(scores * weights[model_idx])
        all_labels.append(labels)
        model_indices.extend([model_idx] * len(boxes))

    if len(all_boxes) == 0:
        return np.array([]).reshape(0, 4), np.array([]), np.array([])

    all_boxes = np.vstack(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    model_indices = np.array(model_indices)

    # Get unique labels
    unique_labels = np.unique(all_labels)

    fused_boxes = []
    fused_scores = []
    fused_labels = []

    for label in unique_labels:
        label_mask = all_labels == label
        label_boxes = all_boxes[label_mask]
        label_scores = all_scores[label_mask]
        label_model_indices = model_indices[label_mask]

        # Sort by score
        sorted_indices = np.argsort(label_scores)[::-1]
        label_boxes = label_boxes[sorted_indices]
        label_scores = label_scores[sorted_indices]
        label_model_indices = label_model_indices[sorted_indices]

        # Cluster boxes
        clusters = []
        used = np.zeros(len(label_boxes), dtype=bool)

        for i in range(len(label_boxes)):
            if used[i]:
                continue

            cluster = [(i, label_boxes[i], label_scores[i], label_model_indices[i])]
            used[i] = True

            for j in range(i + 1, len(label_boxes)):
                if used[j]:
                    continue

                iou = calculate_iou(label_boxes[i], label_boxes[j])
                if iou > iou_threshold:
                    cluster.append(
                        (j, label_boxes[j], label_scores[j], label_model_indices[j])
                    )
                    used[j] = True

            clusters.append(cluster)

        # Fuse each cluster
        for cluster in clusters:
            if len(cluster) == 0:
                continue

            boxes_in_cluster = np.array([c[1] for c in cluster])
            scores_in_cluster = np.array([c[2] for c in cluster])
            models_in_cluster = np.array([c[3] for c in cluster])

            # Weighted average of coordinates
            total_weight = scores_in_cluster.sum()
            if total_weight > 0:
                fused_box = (boxes_in_cluster * scores_in_cluster[:, np.newaxis]).sum(
                    axis=0
                ) / total_weight
            else:
                fused_box = boxes_in_cluster.mean(axis=0)

            # Confidence aggregation
            if conf_type == "avg":
                fused_score = scores_in_cluster.mean()
            elif conf_type == "max":
                fused_score = scores_in_cluster.max()
            elif conf_type == "box_and_model_avg":
                # Average across models that detected this box
                unique_models = np.unique(models_in_cluster)
                fused_score = scores_in_cluster.sum() / len(unique_models)
            else:
                fused_score = scores_in_cluster.mean()

            # Boost confidence based on number of models agreeing
            num_models_agreeing = len(np.unique(models_in_cluster))
            confidence_boost = 1.0 + 0.1 * (num_models_agreeing - 1)
            fused_score = min(1.0, fused_score * confidence_boost)

            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
            fused_labels.append(label)

    if len(fused_boxes) == 0:
        return np.array([]).reshape(0, 4), np.array([]), np.array([])

    fused_boxes = np.array(fused_boxes)
    fused_scores = np.array(fused_scores)
    fused_labels = np.array(fused_labels)

    # Sort by confidence
    sorted_indices = np.argsort(fused_scores)[::-1]

    return (
        fused_boxes[sorted_indices],
        fused_scores[sorted_indices],
        fused_labels[sorted_indices],
    )


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Soft Non-Maximum Suppression - reduces confidence of overlapping boxes
    instead of removing them completely.

    Args:
        boxes: Array of shape (N, 4)
        scores: Array of shape (N,)
        labels: Array of shape (N,)
        sigma: Gaussian decay parameter
        score_threshold: Minimum score to keep a box

    Returns:
        Filtered boxes, scores, and labels
    """
    if len(boxes) == 0:
        return boxes, scores, labels

    # Work with copies
    boxes = boxes.copy()
    scores = scores.copy()
    labels = labels.copy()

    indices = list(range(len(boxes)))
    keep = []

    while len(indices) > 0:
        # Find box with highest score
        max_idx = np.argmax(scores[indices])
        max_pos = indices[max_idx]

        keep.append(max_pos)
        indices.pop(max_idx)

        if len(indices) == 0:
            break

        current_box = boxes[max_pos]
        current_label = labels[max_pos]

        # Apply soft suppression to remaining boxes
        for idx in indices:
            if labels[idx] != current_label:
                continue

            iou = calculate_iou(current_box, boxes[idx])

            # Gaussian decay
            weight = np.exp(-(iou**2) / sigma)
            scores[idx] *= weight

        # Remove boxes below threshold
        indices = [i for i in indices if scores[i] >= score_threshold]

    keep = np.array(keep)
    return boxes[keep], scores[keep], labels[keep]

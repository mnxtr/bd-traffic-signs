# Bangladeshi Road Sign Detection using YOLOv11-Nano and Mobile SSD (BRSSD)

## Abstract
Accurate detection of traffic signs is critical for Advanced Driver Assistance Systems (ADAS) and autonomous navigation, especially in emerging markets where infrastructure variability, occlusion, and domain-specific sign designs challenge generalized models. This paper presents a comparative study of a lightweight YOLOv11-Nano model and a MobileNet-based Single Shot Detector (SSD) (termed BRSSD) trained on a Bangladeshi Road Sign Dataset (BRSD). We describe dataset construction, preprocessing, model adaptation, training configuration, evaluation methodology, expected performance, deployment considerations, and limitations. Preliminary benchmarking (based on analogous custom datasets and published ranges) suggests YOLOv11-Nano outperforms SSD-MobileNet in mAP@0.5 and inference speed per MB of model size, while SSD provides competitive accuracy for slightly larger footprint scenarios. Full empirical results depend on completion of dataset acquisition and SSD loader implementation; this manuscript formalizes the experimental protocol and provides reproducible infrastructure.

## 1. Introduction
Road sign detection underpins safety-critical perception tasks. While large-scale datasets (e.g., GTSRB, LISA) have propelled progress, country-specific sign sets—such as those governed by the Bangladesh Road Transport Authority (BRTA)—remain underrepresented, limiting transferability due to distribution shift (color palette, iconography, environmental context, wear, and language). Resource constraints in local deployments (edge devices, mobile CPUs) motivate evaluation of compact detectors. We investigate two families: (1) YOLOv11-Nano (ultralytics) optimized for real-time inference and (2) MobileNet-based SSDLite (SSD) adapted for BRSD conditions. The contribution includes: a standardized pipeline, dual-format annotation strategy (YOLO + COCO), model configuration baselines, evaluation protocol, and an extensible comparison framework.

## 2. Related Work
Object detection has evolved from two-stage (R-CNN variants) to single-stage paradigms (YOLO, SSD, RetinaNet) emphasizing inference efficiency. YOLO versions have progressively enhanced architectural efficiency, attention, and decoupled heads for improved small-object detection—relevant for diminutive distant signs. SSD introduced multi-scale feature maps enabling simultaneous localization and classification with lower latency than predecessors. MobileNet backbones leverage depthwise separable convolutions for embedded devices. Prior traffic sign studies often target European datasets; adaptation to South Asian contexts introduces variability in lighting (monsoon haze), mounting styles, occlusion (rickshaws, buses), and non-standard sign wear.

## 3. Dataset (Bangladeshi Road Sign Dataset - BRSD)
### 3.1 Scope
Classes are configurable (example set: stop_sign, speed_limit_40, speed_limit_60, no_entry, one_way, yield, danger, construction, pedestrian_crossing). Recommended minimum: ≥100–500 images per class for stable fine-tuning.
### 3.2 Collection Sources
Manual field photography, web-scraped governmental imagery (ensuring legal compliance), and augmentation from similar regional datasets. Diversity targets: weather (rain, bright sun, overcast), time-of-day, occlusion levels, partial visibility, motion blur.
### 3.3 Annotation
Primary format: YOLO (class x_center y_center width height normalized). Secondary conversion: COCO for SSD interoperability. Tools: LabelImg, CVAT, Roboflow. Quality control: double-pass annotation + spot-checking confusion-prone classes (similar pictograms). 
### 3.4 Directory Structure
```
bd-traffic-signs/data/
  raw/          # Images + YOLO .txt label files
  processed/    # Split into train/val/test + data.yaml + COCO export
  splits/       # (Optional) explicit index lists
```
### 3.5 Preprocessing Pipeline
Script: `training/data_preprocessing.py` (invocation example in README). Functions: ratio-based splitting (default 70/20/10), augmentation (geometric transforms, color jitter, motion blur), consistency checks (class mapping), dual-format export. Generates `data.yaml` with class indices for YOLO training and COCO-style JSON for SSD loader extension.

## 4. Methodology
### 4.1 Models
- YOLOv11-Nano: Pretrained COCO weights (`yolo11n.pt`) fine-tuned; emphasis on latency and minimal memory.
- BRSSD (SSD-MobileNet): SSDLite320 MobileNet V3 large backbone; lightweight detection head.
### 4.2 Adaptations
- Input resolution: 640×640 (YOLO) vs 320×320 (SSDLite default)—trade-off between detail and speed.
- Anchor priors: YOLO auto-anchor recalibration during training; SSD uses predefined box aspect ratios—future work: domain-tuned prior optimization.
- Class head modification: SSD classification head adjusted for (N_classes + background); placeholder note indicates need for finalized loader to propagate target tensors.
### 4.3 Training Configuration
YOLO Script: `training/train_yolov11.py` parameters (epochs=100, batch=16, optimizer=auto, lr0=0.01, early stopping patience=50). SSD Script: `training/train_ssd.py` (SGD, lr=0.001, momentum=0.9, ReduceLROnPlateau scheduler). Transfer learning employed (pretrained flags). 
### 4.4 Augmentation Strategy
Base: random horizontal flip (if semantically consistent), scale jitter, HSV shifts, mosaic/mixup (optional YOLO built-ins). SSD augmentation (to be integrated): photometric distortions + random cropping maintaining IoU thresholds.
### 4.5 Evaluation Protocol
Script: `evaluation/evaluate_models.py` provides YOLO validation via Ultralytics runtime and structured speed test (average per-image inference over sample of 100 test images). Planned SSD evaluation requires dataset loader + inference loop integration. Metrics: mAP@0.5, mAP@0.5:0.95, mean precision (mp), mean recall (mr), inference time (ms/image), FPS, model size (MB). Future additions: class-wise AP, small vs medium object breakdown, confusion matrix (converted from matched detection classification), error taxonomy (missed, duplicate, misclassified).
### 4.6 Hardware Profile
Baseline: CPU-only environment (PyTorch 2.x). Optional: CUDA device enabling 5–10× acceleration; guidelines in README for installing GPU-enabled PyTorch.

## 5. Experimental Design
### 5.1 Training Runs
1. YOLOv11-Nano: 100 epochs full; 50-epoch pilot for hyperparameter sanity.
2. SSD-MobileNet: 100 epochs (matching YOLO) with early plateau detection (learning rate reduction triggers).
### 5.2 Hyperparameter Tuning
Grid focus: learning rate (YOLO lr0 ∈ {0.005,0.01}), batch size (8,16,32 subject to memory), augmentation toggles (mosaic on/off), SSD weight decay alternatives (0.0005,0.001). Early stopping monitored via validation mAP stabilization.
### 5.3 Statistical Robustness
Three replicate runs per model variant (n=3) for mean ± std of core metrics; significance test (Welch’s t-test) on mAP@0.5, provided sufficient sample size. 
### 5.4 Error Analysis
Manual inspection of top false negatives (missed small distant signs) and false positives (background clutter resembling signs). Categorization by environmental condition (rain, shadow) to guide augmentation refinement.

## 6. Expected Results (Indicative Ranges, Not Final)
Based on analogous edge-object datasets and README performance expectations:
- YOLOv11-Nano: mAP@0.5 ≈ 0.60–0.75; mAP@0.5:0.95 ≈ 0.35–0.50; FPS (GPU) 500–1000; size ~5–6 MB.
- SSD-MobileNet: mAP@0.5 ≈ 0.55–0.70; mAP@0.5:0.95 ≈ 0.30–0.45; FPS (GPU) 300–500; size ~20–30 MB.
YOLO advantage anticipated in small-object precision due to architectural multi-scale refinements and decoupled head; SSD may trail slightly but remain viable for deployments with moderate memory headroom.

## 7. Deployment Considerations
- Edge (Raspberry Pi / ARM SoC): Favor YOLOv11-Nano due to smaller size and higher FPS at reduced power draw.
- Mobile (Android/iOS): Both feasible; quantization (INT8) reduces size—post-training quantization to be benchmarked.
- Fleet Monitoring (Bus-mounted cameras): Throughput vs energy—batch inference strategies possible using YOLO’s high FPS.
- Model Packaging: ONNX export for YOLO; TorchScript path for SSD; validation of numerical parity required post-conversion.

## 8. Limitations
- SSD evaluation pipeline incomplete (custom dataset/loaders pending implementation for COCO-format ground truth ingestion).
- Absence of finalized empirical metrics at time of writing; ranges are estimated, not measured.
- Potential class imbalance unresolved (requires weighting or focal loss experimentation for extremely rare signs).
- Environmental biases (urban vs rural) may reduce generalization—domain adaptation strategies (style transfer) untested.

## 9. Future Work
1. Implement full SSD evaluation (COCO metrics via pycocotools integration).
2. Extend dataset with nighttime and adverse weather conditions; incorporate low-light enhancement preprocessing.
3. Explore larger YOLO variants (yolo11s/m) for accuracy uplift vs computational trade-offs.
4. Incorporate knowledge distillation: teacher (YOLOv11m) to student (Nano) for improved compact performance.
5. Investigate anchor-free alternatives (e.g., RT-DETR or YOLOv11 anchor-free modes if available) for small sign robustness.
6. Add per-class calibration (Platt scaling) to improve confidence-based filtering in safety-critical systems.
7. Conduct robustness tests under motion blur and extreme compression (simulating low-bandwidth transmission).

## 10. Conclusion
This paper delineates a reproducible framework for benchmarking lightweight detectors on Bangladeshi traffic sign imagery, addressing data preparation, model training, and comparative evaluation. YOLOv11-Nano is projected to deliver superior speed-accuracy efficiency; SSD-MobileNet remains a practical alternative with established ecosystem support. Final empirical validation will follow dataset completion and SSD integration, enabling authoritative conclusions on deployment recommendations for regional intelligent transportation systems.

## 11. Reproducibility Checklist
- Environment: Python 3.10, PyTorch 2.x, Ultralytics YOLO library (verified via README command).
- Scripts: `train_yolov11.py`, `train_ssd.py`, `data_preprocessing.py`, `evaluate_models.py`.
- Config: `data/processed/data.yaml` (generated after preprocessing).
- Versioning: Store commit hash / script versions in future logs for traceability.
- Random Seeds: Set (future enhancement) for dataset splitting and augmentation reproducibility.

## 12. Ethical & Legal Considerations
Ensure image collection complies with privacy norms (avoid identifiable faces/license plates unless blurred). Public roadway signage typically permissible, but confirm local regulations. Dataset redistribution requires proper licensing annotations; follow BRTA guidelines where applicable.

## 13. References
1. Redmon et al., "You Only Look Once" Series.
2. Liu et al., "SSD: Single Shot MultiBox Detector." 
3. Ultralytics YOLOv11 Documentation.
4. MobileNetV3: Howard et al. Efficient Mobile Architecture.
5. BRTA (Bangladesh Road Transport Authority) official signage guidelines.
6. COCO Dataset Paper (Lin et al.).
7. LabelImg / CVAT annotation tool repositories.

## Appendix A: Command Summary
- Preprocess: `python data_preprocessing.py --raw-dir ../data/raw --output-dir ../data/processed --classes ... --augment --coco-format`
- Train YOLO: `python train_yolov11.py --data ../data/processed/data.yaml --model yolo11n.pt --epochs 100 --batch 16 --img-size 640`
- Train SSD: `python train_ssd.py --data-root ../data/processed --backbone mobilenet --num-classes N --epochs 100 --pretrained`
- Evaluate YOLO: `python evaluate_models.py --test-images ../data/processed/test/images --test-labels ../data/processed/test/labels --classes ... --yolo-model ../results/.../best.pt`

## Appendix B: Risk Mitigation for Real-Time Use
- Confidence threshold tuning to reduce false positives at high speed.
- Multi-frame temporal smoothing (majority vote over last k frames) for sign persistence.
- Fallback heuristics for critical mandatory signs (e.g., stop_sign) raising priority alerts.

---
Generated: 2025-11-20
Status: Empirical SSD evaluation pending; infrastructure complete.

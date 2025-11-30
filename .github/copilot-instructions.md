<!-- .github/copilot-instructions.md -->
# Copilot / Agent Instructions — bd-traffic-signs

Purpose: give an AI coding agent the minimal, actionable knowledge to be productive in this repository.

1. Big picture
- **Goal**: compare and evaluate object-detection models (YOLOv11 vs SSD/BRSSD) on Bangladeshi road signs.
- **Main flows**: dataset download -> preprocessing -> train -> evaluate -> results/visualize. See `training/download_dataset.py`, `training/data_preprocessing.py`, `training/train_yolov11.py`, `training/train_ssd.py`, and `evaluation/evaluate_models.py`.

2. Key files & where to look first
- `README.md`: project overview and runnable examples (canonical commands).
- `training/download_dataset.py`: downloads and organizes raw images into `data/raw` and suggests next steps.
- `training/data_preprocessing.py`: produces splits and a YOLO `data.yaml` in `data/processed` (used by YOLO training).
- `training/train_yolov11.py`: full YOLOv11 training entrypoint (Ultralytics API). Use this for most experiments.
- `training/train_ssd.py`: SSD training scaffold — requires custom dataset loaders; it's not plug-and-play.
- `evaluation/evaluate_models.py`: how models are compared and which metrics are expected.
- `results/`: training outputs; `results/yolov11_bd_signs/args.yaml` shows canonical run args.

3. Conventions & patterns specific to this repo
- **Data layout**: `data/raw/` (original images + `.txt` labels YOLO format) -> `data/processed/` (split folders: `train/val/test` each with `images/` and `labels/`). `data_preprocessing.py` writes `data/processed/data.yaml` with `path`, `train`, `val`, `test`, `nc`, `names`.
- **Label format**: YOLO text files per image: `class x_center y_center width height` (normalized coords). SSD expects COCO-style JSON (conversion available in `data_preprocessing.py`).
- **Model variants**: YOLOv11 variants are named `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt` — default in scripts is `yolo11n.pt`.
- **Device handling**: training scripts accept `--device` and will fall back to CPU if CUDA is not available (see `train_yolov11.py` and `train_ssd.py`).
- **Results layout**: training writes under `results/<run_name>/` (weights, args.yaml, plots). Don't commit large weights.

4. Dependency & environment notes
- Dependencies in `requirements.txt`. Project expects a Python venv; README shows `source venv/bin/activate`.
- Uses `ultralytics` (YOLOv11), PyTorch, torchvision, and common libs (cv2, tqdm, sklearn).

5. Practical editing rules for agents
- Preserve data paths: prefer relative paths used in scripts (e.g., `../data/processed/data.yaml`, `../results`).
- Do not add or commit large binary weight files. Suggest `.gitignore` entries instead.
- When modifying training code, run a small dry-run with tiny dataset to validate behavior (suggested command shown below).

6. Copy-paste examples (verified against repo scripts)
- Download dataset (uses Zenodo record):
```
cd training
python download_dataset.py --output-dir ../data/raw --download-dir ../data/downloads
```
- Preprocess and create `data.yaml`:
```
cd training
python data_preprocessing.py \
  --raw-dir ../data/raw \
  --output-dir ../data/processed \
  --classes stop_sign speed_limit_40 speed_limit_60 no_entry one_way
```
- Train YOLOv11 (example):
```
cd training
python train_yolov11.py --data ../data/processed/data.yaml --model yolov11n.pt --epochs 50 --batch 8 --device cpu
```
- Train SSD (note: implement data loaders first):
```
cd training
python train_ssd.py --data-root ../data/processed --backbone mobilenet --num-classes 10 --epochs 100
```
- Run evaluation (example from README):
```
cd evaluation
python evaluate_models.py \
  --test-images ../data/processed/test/images \
  --test-labels ../data/processed/test/labels \
  --yolo-model ../results/yolov11_bd_signs/weights/best.pt \
  --ssd-model ../results/ssd_bd_signs/best_model.pth --device cpu
```

7. Known limitations & gotchas
- `train_ssd.py` contains scaffolding and prints: "Please implement custom dataset loaders" — do not assume SSD training will run without adding dataset code.
- `download_dataset.py` expects internet access and the Zenodo API may require retries for large files; avoid re-downloading weights that are already in repo root (`yolo11n.pt` is present).
- Data augmentation in `data_preprocessing.py` assumes labels remain valid for simple augmentations; verify for geometric transforms.

8. What an agent should do first on a new task
- Run the minimal verification steps: activate venv, install deps from `requirements.txt`, and run a tiny preprocessing + single-step train on CPU to validate command-line glue. Use small subset of `data/raw/` or create 2–3 synthetic images with matching `.txt` labels.

9. When to ask the human
- Ask when making changes that affect dataset format, class order, or when proposing to commit large artifacts (weights, full processed datasets).

Feedback
- If anything here is unclear or you want more examples (e.g., unit tests, CI commands, or one-line experiment reproducibility), say which area and I will expand this file.

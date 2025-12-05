# Bangladeshi Road Sign Detection: YOLOv11 vs BRSSD

A comprehensive comparison of YOLOv11 and SSD (BRSSD) models for detecting traffic symbols in Bangladeshi road signs.

## Project Structure

```
bd-traffic-signs/
├── data/                      # Dataset directory
│   ├── raw/                   # Original images
│   ├── processed/             # Annotated datasets
│   └── splits/                # Train/val/test splits
├── models/                    # Model configurations and weights
│   ├── yolov11/              # YOLOv11 models
│   └── brssd/                # SSD models
├── training/                  # Training scripts
│   ├── train_yolov11.py      # YOLOv11 training script
│   ├── train_ssd.py          # SSD training script
│   └── data_preprocessing.py # Data preprocessing utilities
├── evaluation/                # Evaluation and comparison
│   └── evaluate_models.py    # Model comparison script
├── results/                   # Training outputs
├── notebooks/                 # Jupyter notebooks for analysis
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

### 1. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 2. Verify Installation

```bash
python -c "import torch; import ultralytics; print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__)"
```

## Dataset Preparation

### Step 1: Collect and Annotate Data

1. **Collect images** of Bangladeshi traffic signs
   - Manual photography
   - Web scraping (ensure legal compliance)
   - Public datasets

2. **Annotate images** using tools like:
   - [LabelImg](https://github.com/heartexlabs/labelImg) (YOLO format)
   - [CVAT](https://github.com/opencv/cvat) (Multiple formats)
   - [Roboflow](https://roboflow.com/) (Online annotation)

3. **Define your classes**, for example:
   - stop_sign
   - speed_limit_40
   - speed_limit_60
   - speed_limit_80
   - no_entry
   - one_way
   - pedestrian_crossing
   - yield
   - danger
   - construction

### Step 2: Organize Raw Data

Place your annotated images and labels in the `data/raw/` directory:

```
data/raw/
├── image1.jpg
├── image1.txt  # YOLO format: class x_center y_center width height
├── image2.jpg
├── image2.txt
└── ...
```

### Step 3: Preprocess Dataset

```bash
cd training
python data_preprocessing.py \
    --raw-dir ../data/raw \
    --output-dir ../data/processed \
    --classes stop_sign speed_limit_40 speed_limit_60 no_entry one_way \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1 \
    --augment \
    --coco-format
```

This will:
- Split data into train/val/test sets
- Create `data.yaml` for YOLO training
- Convert annotations to COCO format (for SSD)
- Apply data augmentation (optional)

## Training

### Train YOLOv11

```bash
cd training
python train_yolov11.py \
    --data ../data/processed/data.yaml \
    --model yolov11n.pt \
    --epochs 100 \
    --batch 16 \
    --img-size 640 \
    --device cpu \
    --project ../results \
    --name yolov11_bd_signs
```

**Model variants:**
- `yolov11n.pt` - Nano (fastest, smallest)
- `yolov11s.pt` - Small
- `yolov11m.pt` - Medium
- `yolov11l.pt` - Large
- `yolov11x.pt` - Extra large (best accuracy)

### Monitor Training Progress

Generate visualization graphs of training metrics:

```bash
# Quick generate with default settings
bash generate_graph.sh

# Or use the Python script directly with options
source venv/bin/activate
python plot_training.py --csv results/yolov11_bd_signs/results.csv --show

# Generate for specific training run
python plot_training.py --csv results/your_run_name/results.csv --output ./graphs/
```

This generates comprehensive training visualizations including:
- Training and validation losses (Box, Class, DFL)
- Precision and Recall curves
- mAP@50 and mAP@50-95 progression
- Learning rate schedule
- Cumulative training time

The graphs are automatically saved with timestamps in the results directory.

### Train SSD

```bash
cd training
python train_ssd.py \
    --data-root ../data/processed \
    --backbone mobilenet \
    --num-classes 10 \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --device cpu \
    --output-dir ../results/ssd_bd_signs \
    --pretrained
```

**Note:** SSD training requires custom dataset loaders to be implemented based on your specific dataset format.

## Evaluation & Comparison

After training both models, compare their performance:

```bash
cd evaluation
python evaluate_models.py \
    --test-images ../data/processed/test/images \
    --test-labels ../data/processed/test/labels \
    --classes stop_sign speed_limit_40 speed_limit_60 no_entry one_way \
    --yolo-model ../results/yolov11_bd_signs/weights/best.pt \
    --ssd-model ../results/ssd_bd_signs/best_model.pth \
    --output-dir ../results/comparison \
    --device cpu
```

This generates:
- Comparison metrics (mAP, precision, recall, FPS)
- Visualization charts
- JSON report with detailed results

## Results Interpretation

The evaluation will compare models on:

1. **Accuracy Metrics**
   - mAP@0.5 - Mean Average Precision at IoU 0.5
   - mAP@0.5:0.95 - Mean Average Precision across IoU thresholds
   - Precision - Ratio of correct predictions
   - Recall - Ratio of detected objects

2. **Speed Metrics**
   - Inference time (ms per image)
   - FPS (frames per second)

3. **Model Size**
   - Size in MB (important for deployment)

## Using GPU for Training

If you have an NVIDIA GPU and CUDA installed:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Use --device cuda in training commands
python train_yolov11.py --data ... --device cuda
```

## Cloud Training Options

If local training is slow, consider:

1. **Google Colab** (Free GPU)
   - Upload project to Google Drive
   - Use Colab notebooks with T4 GPU

2. **Kaggle** (Free GPU/TPU)
   - Create Kaggle dataset with your images
   - Use Kaggle notebooks

3. **AWS SageMaker / Azure ML / GCP AI Platform**
   - For production-scale training

## Inference on New Images

### YOLOv11 Inference

```bash
cd training
python -c "
from ultralytics import YOLO
model = YOLO('../results/yolov11_bd_signs/weights/best.pt')
results = model.predict('path/to/image.jpg', save=True, conf=0.25)
"
```

### Batch Inference

```bash
yolo detect predict \
    model=../results/yolov11_bd_signs/weights/best.pt \
    source=path/to/images/ \
    conf=0.25 \
    save=True
```

## Tips for Better Performance

1. **Data Quality**
   - Collect diverse images (different lighting, weather, angles)
   - Ensure accurate annotations
   - Minimum 100-500 images per class

2. **Augmentation**
   - Use the `--augment` flag during preprocessing
   - Increases dataset size and model robustness

3. **Hyperparameter Tuning**
   - Adjust learning rate, batch size, epochs
   - Use early stopping to prevent overfitting

4. **Transfer Learning**
   - Both models use pretrained weights from COCO
   - Fine-tuning usually works better than training from scratch

5. **Model Selection**
   - For real-time applications: YOLOv11n or SSD-MobileNet
   - For highest accuracy: YOLOv11x or SSD-VGG16

## Troubleshooting

### Out of Memory Errors
- Reduce batch size
- Use smaller model variant
- Reduce image size

### Low mAP Scores
- Check annotation quality
- Increase training epochs
- Add more diverse training data
- Adjust confidence threshold

### Slow Training
- Use GPU instead of CPU
- Consider cloud training platforms
- Use smaller model variants for experimentation

## Next Steps

1. ✅ Set up environment and dependencies
2. ⏳ Collect and annotate Bangladeshi traffic sign dataset
3. ⏳ Preprocess dataset
4. ⏳ Train YOLOv11 model
5. ⏳ Train SSD model
6. ⏳ Evaluate and compare models
7. ⏳ Deploy best performing model

## Contributing

This is a research/educational project for comparing detection models on Bangladeshi road signs.

## License

This project is for educational purposes. Ensure proper licensing for any datasets used.

## References

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [PyTorch SSD](https://pytorch.org/vision/stable/models.html#object-detection)
- [BRTA - Bangladesh Road Transport Authority](http://www.brta.gov.bd/)

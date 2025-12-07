# Quick Start Guide

## 🚀 Your Environment is Ready!

The project has been successfully set up with all dependencies installed.

## ✅ What's Been Done

1. ✅ Created project directory structure
2. ✅ Set up Python virtual environment
3. ✅ Installed PyTorch (CPU version)
4. ✅ Installed YOLOv11 (Ultralytics)
5. ✅ Installed OpenCV and ML utilities
6. ✅ Created training scripts for both models
7. ✅ Created data preprocessing utilities
8. ✅ Created evaluation and comparison scripts
9. ✅ Generated comprehensive documentation

## 📋 Next Steps

### 1. Activate the Virtual Environment

```bash
cd ~/bd-traffic-signs
source venv/bin/activate
```

### 2. Collect Your Dataset

You need to collect and annotate Bangladeshi traffic sign images. Options:

**Option A: Use an existing dataset**
- Search for "Bangladesh road sign dataset" on Kaggle, Roboflow, or academic sources
- Download and place in `data/raw/`

**Option B: Create your own dataset**
- Use [LabelImg](https://github.com/heartexlabs/labelImg) for annotation
- Install: `pip install labelImg` then run `labelImg`
- Annotate in YOLO format (class x_center y_center width height)
- Save images and labels to `data/raw/`

**Option C: Use a sample dataset for testing**
- Download any traffic sign dataset (GTSRB, LISA, etc.) to test the pipeline
- Adapt class names to your needs

### 3. Prepare Your Data

Once you have raw annotated data in `data/raw/`:

```bash
cd training
python data_preprocessing.py \
    --raw-dir ../data/raw \
    --output-dir ../data/processed \
    --classes stop_sign speed_limit no_entry one_way yield \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1 \
    --augment
```

### 4. Train YOLOv11 (Recommended to Start With)

```bash
cd training
python train_yolov11.py \
    --data ../data/processed/data.yaml \
    --model yolov11n.pt \
    --epochs 50 \
    --batch 8 \
    --img-size 640 \
    --device cpu \
    --project ../results \
    --name yolov11_test
```

**Note:** Start with fewer epochs (50) and smaller batch size (8) for testing on CPU.

### 5. Train SSD (Optional)

The SSD implementation is ready but requires additional dataset loader implementation for your specific format.

### 6. Evaluate Your Model

After training YOLOv11:

```bash
cd evaluation
python evaluate_models.py \
    --test-images ../data/processed/test/images \
    --test-labels ../data/processed/test/labels \
    --classes stop_sign speed_limit no_entry one_way yield \
    --yolo-model ../results/yolov11_test/weights/best.pt \
    --output-dir ../results/comparison
```

## 🎯 Quick Test (No Dataset Needed)

To verify everything works, you can test YOLOv11 with a pretrained COCO model:

```bash
cd ~/bd-traffic-signs
source venv/bin/activate
python -c "
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO

# Load pretrained model
model = YOLO('yolov11n.pt')

# Test on a sample image
results = model.predict('https://ultralytics.com/images/bus.jpg')
print('✅ Model loaded and inference successful!')
print(f'Detected {len(results[0].boxes)} objects')
"
```

## 💡 Tips

1. **Start Small**: Use a small dataset (100-200 images) to test the pipeline first
2. **CPU Training**: Training on CPU is slow. Consider:
   - Using Google Colab (free GPU)
   - Reducing epochs and batch size
   - Using the nano model (yolov11n.pt)
3. **Data Quality**: Good annotations are more important than quantity
4. **Monitor Training**: Check `results/` directory for training plots and metrics

## 📚 Important Files

- `README.md` - Full documentation
- `requirements.txt` - Installed Python packages
- `data.yaml.example` - Template for dataset configuration
- `training/train_yolov11.py` - YOLOv11 training script
- `training/train_ssd.py` - SSD training script
- `training/data_preprocessing.py` - Dataset preparation
- `evaluation/evaluate_models.py` - Model comparison

## 🆘 Need Help?

**Common Issues:**

1. **"No images found"**
   - Check that images are in `data/raw/`
   - Ensure images have extensions: .jpg, .jpeg, .png, .bmp

2. **"CUDA not available"**
   - Normal on CPU systems
   - Training will use CPU (slower but works)

3. **"Out of memory"**
   - Reduce batch size: `--batch 4` or `--batch 2`
   - Use smaller model: `yolov11n.pt`

4. **Slow training**
   - Normal on CPU
   - Consider cloud platforms (Colab, Kaggle)
   - Start with fewer epochs for testing

## 🔗 Useful Resources

- [Ultralytics Docs](https://docs.ultralytics.com/)
- [LabelImg Tutorial](https://github.com/heartexlabs/labelImg)
- [Google Colab](https://colab.research.google.com/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

## 📊 Expected Timeline

- Dataset collection & annotation: 1-7 days (depending on size)
- Data preprocessing: 1-2 hours
- YOLOv11 training (CPU): 2-12 hours (50-100 epochs)
- YOLOv11 training (GPU): 10-60 minutes (50-100 epochs)
- SSD training: Similar to YOLOv11
- Evaluation: 10-30 minutes

## 🎉 Ready to Start!

You're all set! The infrastructure is in place. Now you need to:
1. Get your dataset
2. Run preprocessing
3. Train models
4. Compare results

Good luck with your Bangladeshi traffic sign detection project!

# YOLOv11 Model Export Guide

## Quick Start

Run the export script to convert your trained YOLOv11 model to INT8 quantized TFLite format:

```bash
python3 export_quantized.py
```

This will:
1. Load the best trained model from `results/yolov11_bd_signs_20251122_192224/weights/best.pt`
2. Export it to INT8 quantized TFLite format (optimized for mobile)
3. Save it to `android-app/app/src/main/assets/traffic_signs_yolov11_int8.tflite`

## Model Details

- **Input size**: 320x320 pixels
- **Format**: TFLite with INT8 quantization
- **Optimization**: Reduced model size and faster inference on mobile devices
- **Output**: YOLO detection format (bounding boxes, confidence scores, class predictions)

## Integration in Android App

The quantized model is integrated into the Android app through:
1. **TrafficSignDetector.java**: Handles model loading and inference
2. **MainActivity.java**: Uses detector for real-time traffic sign detection
3. **TensorFlow Lite dependencies**: Added to build.gradle

## Usage in App

1. The app loads the quantized model at startup
2. Uses GPU acceleration if available
3. Processes camera input for real-time detection
4. Speaks detected sign names in Bengali using TTS

## Model Performance

INT8 quantization provides:
- ~4x smaller model size
- ~2-4x faster inference
- Minimal accuracy loss (<1-2%)
- Better battery efficiency on mobile devices

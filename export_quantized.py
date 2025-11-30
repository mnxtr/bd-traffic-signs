"""
Export YOLOv11 model to quantized TFLite format for Android deployment
"""
import os
from ultralytics import YOLO

def export_quantized_model(model_path, output_dir='android-app/app/src/main/assets'):
    """
    Export YOLOv11 model to INT8 quantized TFLite format
    
    Args:
        model_path: Path to .pt model file
        output_dir: Directory to save exported model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    print("Exporting to INT8 quantized TFLite format...")
    print("This may take several minutes...")
    
    # Export with INT8 quantization for better performance on mobile
    export_path = model.export(
        format='tflite',
        int8=True,
        imgsz=320,  # Smaller size for mobile devices
    )
    
    # Move to assets folder
    import shutil
    tflite_name = 'traffic_signs_yolov11_int8.tflite'
    dest_path = os.path.join(output_dir, tflite_name)
    shutil.copy(export_path, dest_path)
    
    print(f"\n✓ Quantized model exported successfully!")
    print(f"  Location: {dest_path}")
    print(f"  Size: {os.path.getsize(dest_path) / (1024*1024):.2f} MB")
    print(f"\nModel ready for Android deployment!")
    
    return dest_path

if __name__ == '__main__':
    model_path = 'results/yolov11_bd_signs_20251122_192224/weights/best.pt'
    export_quantized_model(model_path)

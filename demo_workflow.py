#!/usr/bin/env python3
"""
BD Traffic Signs Detection - Demo Workflow
Demonstrates the complete pipeline with sample/test data
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def verify_environment():
    """Verify system setup"""
    print_section("STEP 1: Environment Verification")
    
    print("✅ Python Environment:")
    print(f"   Python version: {sys.version.split()[0]}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    
    print("\n✅ Project Structure:")
    project_root = Path(__file__).parent
    dirs = ['data/raw', 'data/processed', 'training', 'evaluation', 
            'models/yolov11', 'models/brssd', 'results']
    for dir_path in dirs:
        full_path = project_root / dir_path
        status = "✓" if full_path.exists() else "✗"
        print(f"   [{status}] {dir_path}")
    
    return project_root

def demo_yolov11():
    """Demonstrate YOLOv11 capabilities"""
    print_section("STEP 2: YOLOv11 Model Demo")
    
    print("📦 Loading pretrained YOLOv11 nano model...")
    model = YOLO('yolo11n.pt')
    
    print(f"   Model: YOLOv11n")
    print(f"   Parameters: ~2.6M")
    print(f"   Classes: {len(model.names)} (COCO dataset)")
    print(f"   Input size: 640x640")
    
    print("\n🎯 Running inference on test image...")
    test_url = 'https://ultralytics.com/images/bus.jpg'
    results = model.predict(test_url, verbose=False, save=False)
    
    print(f"   ✓ Detected {len(results[0].boxes)} objects:")
    for i, box in enumerate(results[0].boxes[:8], 1):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        print(f"      {i}. {cls_name:<15} {conf:>6.1%} confidence")
    
    return model

def show_training_workflow():
    """Display the training workflow"""
    print_section("STEP 3: Training Workflow (When Dataset Ready)")
    
    print("📋 Data Preparation:")
    print("   1. Collect Bangladesh traffic sign images")
    print("   2. Annotate using LabelImg or Roboflow")
    print("   3. Organize in data/raw/ directory")
    print()
    print("   Command:")
    print("   $ cd training")
    print("   $ python data_preprocessing.py \\")
    print("       --raw-dir ../data/raw \\")
    print("       --output-dir ../data/processed \\")
    print("       --classes stop_sign speed_limit no_entry one_way yield \\")
    print("       --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 \\")
    print("       --augment")
    
    print("\n🚀 YOLOv11 Training:")
    print("   Command:")
    print("   $ python train_yolov11.py \\")
    print("       --data ../data/processed/data.yaml \\")
    print("       --model yolo11n.pt \\")
    print("       --epochs 100 \\")
    print("       --batch 8 \\")
    print("       --img-size 640 \\")
    print("       --device cpu \\")
    print("       --project ../results \\")
    print("       --name yolov11_bd_signs")
    
    print("\n⚙️  SSD Training (Optional):")
    print("   Command:")
    print("   $ python train_ssd.py \\")
    print("       --data-root ../data/processed \\")
    print("       --backbone mobilenet \\")
    print("       --num-classes 5 \\")
    print("       --epochs 100 \\")
    print("       --batch-size 8 \\")
    print("       --device cpu \\")
    print("       --output-dir ../results/ssd_bd_signs")

def show_evaluation_workflow():
    """Display the evaluation workflow"""
    print_section("STEP 4: Model Evaluation & Comparison")
    
    print("📊 Evaluation Metrics:")
    print("   • mAP@0.5 - Mean Average Precision at IoU threshold 0.5")
    print("   • mAP@0.5:0.95 - mAP across multiple IoU thresholds")
    print("   • Precision - Ratio of correct positive predictions")
    print("   • Recall - Ratio of detected ground truth objects")
    print("   • Inference Time - Speed in FPS (frames per second)")
    print("   • Model Size - Size in MB")
    
    print("\n🔬 Comparison Command:")
    print("   $ cd evaluation")
    print("   $ python evaluate_models.py \\")
    print("       --test-images ../data/processed/test/images \\")
    print("       --test-labels ../data/processed/test/labels \\")
    print("       --classes stop_sign speed_limit no_entry one_way yield \\")
    print("       --yolo-model ../results/yolov11_bd_signs/weights/best.pt \\")
    print("       --ssd-model ../results/ssd_bd_signs/best_model.pth \\")
    print("       --output-dir ../results/comparison \\")
    print("       --device cpu")
    
    print("\n📈 Output:")
    print("   • Comparison metrics (JSON)")
    print("   • Performance charts (PNG)")
    print("   • Confusion matrices")
    print("   • Sample predictions")

def show_deployment_options():
    """Display deployment options"""
    print_section("STEP 5: Deployment Options")
    
    print("🚀 Inference on New Images:")
    print()
    print("   Python API:")
    print("   ```python")
    print("   from ultralytics import YOLO")
    print("   model = YOLO('results/yolov11_bd_signs/weights/best.pt')")
    print("   results = model.predict('image.jpg', conf=0.25)")
    print("   ```")
    
    print("\n   Command Line:")
    print("   $ yolo detect predict \\")
    print("       model=results/yolov11_bd_signs/weights/best.pt \\")
    print("       source=path/to/images/ \\")
    print("       conf=0.25")
    
    print("\n📱 Export Options:")
    print("   • ONNX - For general deployment")
    print("   • TensorRT - For NVIDIA GPUs")
    print("   • CoreML - For iOS devices")
    print("   • TFLite - For Android/mobile")
    print()
    print("   Command:")
    print("   $ yolo export model=best.pt format=onnx")

def show_next_steps():
    """Display next steps"""
    print_section("Next Steps")
    
    print("📋 To continue with this project:")
    print()
    print("1. 📥 Download Dataset")
    print("   Option A: Resume automatic download")
    print("   $ cd training")
    print("   $ python download_dataset.py --output-dir ../data/raw")
    print()
    print("   Option B: Collect manually")
    print("   • Take photos of Bangladesh traffic signs")
    print("   • Use LabelImg for annotation")
    print("   • Save to data/raw/")
    print()
    print("   Option C: Use public dataset for testing")
    print("   • GTSRB (German Traffic Sign Recognition Benchmark)")
    print("   • LISA Traffic Sign Dataset")
    print()
    
    print("2. 🔄 Preprocess Data")
    print("   $ cd training")
    print("   $ python data_preprocessing.py --raw-dir ../data/raw")
    print()
    
    print("3. 🎓 Train Models")
    print("   $ python train_yolov11.py --data ../data/processed/data.yaml")
    print()
    
    print("4. 📊 Evaluate & Compare")
    print("   $ cd ../evaluation")
    print("   $ python evaluate_models.py [options]")
    print()
    
    print("💡 Tips:")
    print("   • Start with small dataset (50-100 images) to test pipeline")
    print("   • Use --epochs 50 for faster initial training")
    print("   • Consider Google Colab for free GPU access")
    print("   • Check results/ directory for training outputs")

def main():
    """Main demo workflow"""
    print("\n" + "🎯"*35)
    print(" "*20 + "BD TRAFFIC SIGNS DETECTION")
    print(" "*25 + "YOLOv11 vs BRSSD")
    print(" "*22 + "Complete Workflow Demo")
    print("🎯"*35)
    
    try:
        # Step 1: Verify environment
        project_root = verify_environment()
        
        # Step 2: Demo YOLOv11
        model = demo_yolov11()
        
        # Step 3: Show training workflow
        show_training_workflow()
        
        # Step 4: Show evaluation workflow
        show_evaluation_workflow()
        
        # Step 5: Show deployment options
        show_deployment_options()
        
        # Show next steps
        show_next_steps()
        
        print_section("✅ Demo Completed Successfully")
        print("All systems are ready for dataset preparation and training!")
        print()
        print("📖 Documentation:")
        print("   • README.md - Full project documentation")
        print("   • QUICKSTART.md - Quick start guide")
        print("   • IMPLEMENTATION_STATUS.md - Current status and plan")
        print()
        print("📧 For questions or issues, refer to the documentation.")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

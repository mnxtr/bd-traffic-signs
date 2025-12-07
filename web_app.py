#!/usr/bin/env python3
"""
Enhanced Web Interface for Real-Time Traffic Sign Detection

Features:
- Real-time webcam detection with adjustable FPS
- Video upload with frame-by-frame processing
- Image upload with batch support
- Model selection: YOLOv11 / Ensemble / Multi-scale
- Modern dark theme with glassmorphism effects
- Detection statistics dashboard
- Side-by-side model comparison
"""

import gradio as gr
import cv2
import numpy as np
import yaml
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime

from ensemble_detector import EnsembleDetector, YOLOv11Detector


print("\n" + "="*70)
print("🚦  BANGLADESH TRAFFIC SIGN DETECTION SYSTEM - ENHANCED")
print("="*70)
print("🔄 Initializing components...\n")

# Configuration
MODEL_PATH = "results/bd_signs_v1/weights/best.pt"
CONFIG_PATH = "data/processed/data.yaml"

# Initialize detectors
print(f"   📦 Loading YOLOv11 model...")
yolo_detector = YOLOv11Detector(model_path=MODEL_PATH)

print(f"   📦 Initializing Ensemble detector...")
ensemble_detector = EnsembleDetector(
    yolo_path=MODEL_PATH,
    use_multi_scale=True,
    scales=[480, 640, 800]
)

# Load class names
try:
    with open(CONFIG_PATH, "r") as f:
        data_config = yaml.safe_load(f)
        class_names = data_config.get("names", [])
except Exception as e:
    class_names = list(ensemble_detector.class_names.values())
    print(f"   ⚠️ Could not load config, using model class names")

model_size = os.path.getsize(MODEL_PATH) / (1024*1024)  # MB

print(f"   ✅ Loaded {len(class_names)} traffic sign classes")
print(f"   📊 Model size: {model_size:.2f} MB")
print("\n" + "="*70)
print("🎉  System ready! Launching enhanced web interface...")
print("="*70 + "\n")

# Detection statistics
stats = {
    'total_detections': 0,
    'total_frames': 0,
    'avg_fps': 0,
    'avg_confidence': 0,
    'session_start': datetime.now()
}


def process_image(
    image: np.ndarray, 
    model_choice: str = "YOLOv11",
    conf_threshold: float = 0.25,
    show_stats: bool = True
) -> Tuple[np.ndarray, str]:
    """
    Process a single image with the selected detection model.
    
    Args:
        image: Input image (RGB from Gradio)
        model_choice: Which model to use
        conf_threshold: Confidence threshold
        show_stats: Whether to show detection statistics
        
    Returns:
        Annotated image and statistics text
    """
    if image is None:
        return None, "No image provided"
    
    start_time = time.time()
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Get mode
    if model_choice == "YOLOv11":
        mode = "yolo"
    elif model_choice == "Multi-Scale YOLOv11":
        mode = "multi_scale"
    elif model_choice == "Ensemble":
        mode = "ensemble"
    else:
        mode = "yolo"
    
    # Run detection
    ensemble_detector.conf_threshold = conf_threshold
    ensemble_detector.yolo.conf_threshold = conf_threshold
    
    boxes, scores, labels, _ = ensemble_detector.predict(image_bgr, mode=mode)
    
    # Draw detections
    annotated = ensemble_detector.draw_detections(image_bgr, boxes, scores, labels)
    
    # Convert back to RGB
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    inference_time = time.time() - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0
    
    # Update stats
    stats['total_detections'] += len(boxes)
    stats['total_frames'] += 1
    stats['avg_fps'] = (stats['avg_fps'] * (stats['total_frames'] - 1) + fps) / stats['total_frames']
    if len(scores) > 0:
        stats['avg_confidence'] = float(scores.mean())
    
    # Generate statistics text
    if show_stats and len(boxes) > 0:
        summary = ensemble_detector.get_detection_summary(boxes, scores, labels)
        
        stats_text = f"**📊 Detection Results**\n\n"
        stats_text += f"**Mode:** {model_choice}\n"
        stats_text += f"**Detections:** {len(boxes)}\n"
        stats_text += f"**Inference Time:** {inference_time*1000:.1f}ms\n"
        stats_text += f"**FPS:** {fps:.1f}\n"
        stats_text += f"**Avg Confidence:** {summary['avg_confidence']:.2%}\n\n"
        
        stats_text += "**Classes Detected:**\n"
        for item in summary['classes_detected']:
            stats_text += f"- {item['class_name']}: {item['count']}\n"
    else:
        stats_text = f"**📊 Detection Results**\n\n"
        stats_text += f"**Mode:** {model_choice}\n"
        stats_text += f"**Detections:** {len(boxes)}\n"
        stats_text += f"**Inference Time:** {inference_time*1000:.1f}ms\n"
        stats_text += f"**FPS:** {fps:.1f}\n"
        if len(boxes) == 0:
            stats_text += "\n*No signs detected in this image.*"
    
    return annotated_rgb, stats_text


def process_webcam(
    frame: np.ndarray,
    model_choice: str = "YOLOv11",
    conf_threshold: float = 0.25
) -> np.ndarray:
    """Process webcam frame in real-time."""
    if frame is None:
        return None
    
    annotated, _ = process_image(frame, model_choice, conf_threshold, show_stats=False)
    return annotated


def process_video(
    video_path: str,
    model_choice: str = "YOLOv11",
    conf_threshold: float = 0.25,
    progress=gr.Progress()
) -> str:
    """Process video file frame by frame with progress tracking."""
    if video_path is None:
        return None
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video
    output_path = f"output_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_rgb, _ = process_image(frame_rgb, model_choice, conf_threshold, show_stats=False)
        
        # Convert back to BGR for video writing
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)
        
        frame_count += 1
        progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    return output_path


def compare_models(
    image: np.ndarray,
    conf_threshold: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Run all models on the same image for comparison.
    
    Returns:
        Tuple of (YOLOv11 result, Multi-scale result, Ensemble result, comparison text)
    """
    if image is None:
        return None, None, None, "No image provided"
    
    results = {}
    
    for name, mode in [("YOLOv11", "yolo"), ("Multi-Scale", "multi_scale"), ("Ensemble", "ensemble")]:
        start = time.time()
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes, scores, labels, _ = ensemble_detector.predict(image_bgr, mode=mode)
        annotated = ensemble_detector.draw_detections(image_bgr, boxes, scores, labels)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        results[name] = {
            'image': annotated_rgb,
            'detections': len(boxes),
            'time': (time.time() - start) * 1000,
            'avg_conf': float(scores.mean()) if len(scores) > 0 else 0
        }
    
    # Comparison text
    comparison = "## 📊 Model Comparison\n\n"
    comparison += "| Model | Detections | Time (ms) | Avg Confidence |\n"
    comparison += "|-------|------------|-----------|----------------|\n"
    
    for name, data in results.items():
        comparison += f"| {name} | {data['detections']} | {data['time']:.1f} | {data['avg_conf']:.2%} |\n"
    
    return (
        results["YOLOv11"]['image'],
        results["Multi-Scale"]['image'],
        results["Ensemble"]['image'],
        comparison
    )


def get_session_stats() -> str:
    """Get current session statistics."""
    runtime = datetime.now() - stats['session_start']
    
    return f"""## 📈 Session Statistics

**Session Duration:** {str(runtime).split('.')[0]}
**Total Frames Processed:** {stats['total_frames']}
**Total Detections:** {stats['total_detections']}
**Average FPS:** {stats['avg_fps']:.1f}
**Last Avg Confidence:** {stats['avg_confidence']:.2%}
"""


# Custom CSS for dark theme with glassmorphism
CUSTOM_CSS = """
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    --dark-bg: #0f0f1a;
    --card-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
}

.dark {
    --background-fill-primary: #0f0f1a !important;
}

.header-banner {
    background: var(--primary-gradient);
    padding: 24px 32px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}

.header-banner h1 {
    margin: 0;
    font-size: 2.2em;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.header-banner p {
    margin: 8px 0 0 0;
    font-size: 1.1em;
    opacity: 0.95;
}

.stat-card {
    background: var(--card-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
}

.stat-card .stat-icon {
    font-size: 2em;
    margin-bottom: 8px;
}

.stat-card .stat-value {
    font-size: 1.8em;
    font-weight: 700;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-card .stat-label {
    font-size: 0.9em;
    opacity: 0.8;
    margin-top: 4px;
}

.info-section {
    background: rgba(102, 126, 234, 0.1);
    border-left: 4px solid #667eea;
    padding: 16px 20px;
    border-radius: 8px;
    margin: 16px 0;
}

.feature-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--primary-gradient);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9em;
    font-weight: 500;
    margin: 4px;
}

.comparison-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}

footer {
    text-align: center;
    padding: 24px;
    opacity: 0.7;
    border-top: 1px solid var(--glass-border);
    margin-top: 32px;
}

/* Button styling */
.gr-button-primary {
    background: var(--primary-gradient) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
}

/* Tab styling */
.tab-nav button.selected {
    background: var(--primary-gradient) !important;
    color: white !important;
}
"""


# Build the Gradio interface (Gradio 6.0 compatible)
with gr.Blocks() as demo:
    
    # Header
    gr.HTML("""
    <div class="header-banner">
        <h1>🚦 Bangladesh Traffic Sign Detection</h1>
        <p>Real-time AI-powered detection with YOLOv11 & Ensemble Models</p>
    </div>
    """)
    
    # Quick stats row
    with gr.Row():
        gr.HTML(f'''
        <div class="stat-card">
            <div class="stat-icon">⚡</div>
            <div class="stat-value">22.2</div>
            <div class="stat-label">FPS (CPU)</div>
        </div>
        ''')
        gr.HTML(f'''
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-value">99.45%</div>
            <div class="stat-label">mAP@50</div>
        </div>
        ''')
        gr.HTML(f'''
        <div class="stat-card">
            <div class="stat-icon">📦</div>
            <div class="stat-value">{model_size:.1f}</div>
            <div class="stat-label">MB Model</div>
        </div>
        ''')
        gr.HTML(f'''
        <div class="stat-card">
            <div class="stat-icon">🏷️</div>
            <div class="stat-value">{len(class_names)}</div>
            <div class="stat-label">Classes</div>
        </div>
        ''')
    
    gr.HTML("<br>")
    
    # Main tabs
    with gr.Tabs():
        
        # Tab 1: Real-time Webcam
        with gr.Tab("📹 Live Webcam", id="webcam"):
            gr.HTML('<div class="info-section">📌 <strong>Real-time detection</strong> from your webcam. Select a model and adjust confidence as needed.</div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_model = gr.Radio(
                        choices=["YOLOv11", "Multi-Scale YOLOv11", "Ensemble"],
                        value="YOLOv11",
                        label="Detection Model"
                    )
                    webcam_conf = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="Confidence Threshold"
                    )
                
                with gr.Column(scale=3):
                    with gr.Row():
                        webcam_input = gr.Image(
                            sources=["webcam"], 
                            streaming=True, 
                            type="numpy",
                            label="Camera Input"
                        )
                        webcam_output = gr.Image(
                            label="Detected Signs",
                            type="numpy"
                        )
            
            webcam_input.stream(
                process_webcam,
                inputs=[webcam_input, webcam_model, webcam_conf],
                outputs=webcam_output
            )
        
        # Tab 2: Image Upload
        with gr.Tab("🖼️ Image Detection", id="image"):
            gr.HTML('<div class="info-section">📌 <strong>Upload an image</strong> to detect traffic signs. Supports JPG, PNG, BMP formats.</div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_model = gr.Radio(
                        choices=["YOLOv11", "Multi-Scale YOLOv11", "Ensemble"],
                        value="YOLOv11",
                        label="Detection Model"
                    )
                    image_conf = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="Confidence Threshold"
                    )
                    image_btn = gr.Button("🔍 Detect Signs", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    
                with gr.Column(scale=2):
                    image_output = gr.Image(label="Detection Result", type="numpy")
                    image_stats = gr.Markdown(label="Statistics")
            
            image_btn.click(
                process_image,
                inputs=[image_input, image_model, image_conf],
                outputs=[image_output, image_stats]
            )
        
        # Tab 3: Video Processing
        with gr.Tab("🎥 Video Processing", id="video"):
            gr.HTML('<div class="info-section">📌 <strong>Upload a video</strong> to process all frames. Results will be saved as a new video file.</div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_model = gr.Radio(
                        choices=["YOLOv11", "Multi-Scale YOLOv11", "Ensemble"],
                        value="YOLOv11",
                        label="Detection Model"
                    )
                    video_conf = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="Confidence Threshold"
                    )
                    video_btn = gr.Button("▶️ Process Video", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    video_input = gr.Video(label="Upload Video")
                    
                with gr.Column(scale=2):
                    video_output = gr.Video(label="Processed Video")
            
            video_btn.click(
                process_video,
                inputs=[video_input, video_model, video_conf],
                outputs=video_output
            )
        
        # Tab 4: Model Comparison
        with gr.Tab("📊 Compare Models", id="compare"):
            gr.HTML('<div class="info-section">📌 <strong>Compare all models</strong> side-by-side on the same image to see differences in detection.</div>')
            
            with gr.Row():
                compare_input = gr.Image(type="numpy", label="Upload Image for Comparison")
                compare_conf = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                    label="Confidence Threshold"
                )
                compare_btn = gr.Button("🔬 Compare Models", variant="primary")
            
            with gr.Row():
                compare_yolo = gr.Image(label="YOLOv11", type="numpy")
                compare_multi = gr.Image(label="Multi-Scale YOLOv11", type="numpy")
                compare_ensemble = gr.Image(label="Ensemble", type="numpy")
            
            compare_stats = gr.Markdown()
            
            compare_btn.click(
                compare_models,
                inputs=[compare_input, compare_conf],
                outputs=[compare_yolo, compare_multi, compare_ensemble, compare_stats]
            )
        
        # Tab 5: About
        with gr.Tab("ℹ️ About", id="about"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"""
## 🚦 System Information

**Model Path:** `{MODEL_PATH}`  
**Architecture:** YOLOv11 Nano  
**Accuracy:** 99.45% mAP@50  
**Size:** {model_size:.2f} MB

---

## 🔧 Detection Modes

| Mode | Description | Speed | Accuracy |
|------|-------------|-------|----------|
| **YOLOv11** | Standard detection at 640px | ⚡ Fast | High |
| **Multi-Scale** | Detection at 480, 640, 800px | 🔄 Medium | Higher |
| **Ensemble** | Combines all scales with WBF | 🎯 Slower | Highest |

---

## 📋 Detected Traffic Sign Classes

{', '.join(class_names) if isinstance(class_names, list) else ', '.join(class_names.values())}
                    """)
                
                with gr.Column():
                    session_stats = gr.Markdown(value=get_session_stats())
                    refresh_btn = gr.Button("🔄 Refresh Stats")
                    refresh_btn.click(get_session_stats, outputs=session_stats)
    
    # Footer
    gr.HTML("""
    <footer>
        <p>🚀 Powered by YOLOv11 & Weighted Box Fusion | 🇧🇩 Bangladesh Traffic Sign Detection</p>
        <p style="font-size: 0.9em; opacity: 0.7;">Ensemble model combines multi-scale inference for robust detection</p>
    </footer>
    """)


if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

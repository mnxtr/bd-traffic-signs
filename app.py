import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import os

print("\n" + "="*70)
print("🚦  BANGLADESH TRAFFIC SIGN DETECTION SYSTEM")
print("="*70)
print("🔄 Initializing components...\n")

# Load the trained YOLOv11 model
MODEL_PATH = "results/bd_signs_v1/weights/best.pt"
print(f"   📦 Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"   ✅ Model loaded successfully!")

# Load class names from data.yaml
config_path = "data/processed/data.yaml"
print(f"   📋 Loading configuration: {config_path}")
with open(config_path, "r") as f:
    data_config = yaml.safe_load(f)
    class_names = data_config.get("names", [])
print(f"   ✅ Loaded {len(class_names)} traffic sign classes")

# Get model info
model_size = os.path.getsize(MODEL_PATH) / (1024*1024)  # MB
print(f"   📊 Model size: {model_size:.2f} MB")
print("\n" + "="*70)
print("🎉  System ready! Launching web interface...\n")
print("="*70 + "\n")

def process_frame(frame):
    """Process a single frame with YOLOv11 detection"""
    if frame is None:
        return None
    
    # Run inference
    results = model(frame, conf=0.25)
    
    # Draw results on frame
    annotated_frame = results[0].plot()
    
    return annotated_frame

def process_video(video):
    """Process video file frame by frame"""
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame = process_frame(frame)
        out.write(annotated_frame)
    
    cap.release()
    out.release()
    
    return output_path

def process_webcam(frame):
    """Process webcam stream in realtime"""
    return process_frame(frame)

# Custom CSS for enhanced styling
CUSTOM_CSS = """
.header-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}
.stat-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    color: white;
    margin: 10px 0;
    font-weight: bold;
}
.info-section {
    background: #f0f2f5;
    padding: 15px;
    border-left: 4px solid #667eea;
    border-radius: 5px;
    margin: 10px 0;
}
.feature-highlight {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 6px;
    margin: 5px;
    display: inline-block;
    font-weight: 500;
}
"""

# Create Gradio interface with enhanced styling
with gr.Blocks(title="BD Traffic Sign Detection", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    # Enhanced header
    gr.HTML("""
    <div class="header-banner">
        <h1 style="margin: 0; font-size: 2.5em;">🚦 Bangladesh Traffic Sign Detection</h1>
        <p style="margin: 8px 0 0 0; font-size: 1.1em; opacity: 0.95;">Real-time AI-powered traffic sign detection using YOLOv11</p>
    </div>
    """)
    
    # Quick stats
    with gr.Row():
        with gr.Column():
            gr.HTML(f'<div class="stat-card">⚡ Speed<br>22.2 FPS (CPU)</div>')
        with gr.Column():
            gr.HTML(f'<div class="stat-card">🎯 Accuracy<br>99.45% mAP</div>')
        with gr.Column():
            gr.HTML(f'<div class="stat-card">📦 Size<br>{model_size:.1f} MB</div>')
        with gr.Column():
            gr.HTML(f'<div class="stat-card">🏷️ Classes<br>{len(class_names)} Signs</div>')
    
    gr.HTML("<br>")
    
    # Main tabs
    with gr.Tab("📹 Live Webcam", elem_id="webcam-tab"):
        with gr.Column():
            gr.HTML('<div class="info-section"><strong>📌 How to use:</strong> Click "Start streaming" to begin real-time detection</div>')
            with gr.Row():
                webcam_input = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Input Stream")
                webcam_output = gr.Image(label="Detected Signs", type="numpy")
            webcam_input.stream(process_webcam, inputs=webcam_input, outputs=webcam_output)
    
    with gr.Tab("🎥 Video Upload", elem_id="video-tab"):
        with gr.Column():
            gr.HTML('<div class="info-section"><strong>📌 How to use:</strong> Upload a video file and click "Process Video" to detect signs</div>')
            with gr.Row():
                video_input = gr.Video(label="Input Video", type="filepath")
                video_output = gr.Video(label="Output Video", type="filepath")
            with gr.Row():
                video_btn = gr.Button("▶️ Process Video", variant="primary", scale=1)
            video_btn.click(process_video, inputs=video_input, outputs=video_output)
    
    with gr.Tab("📷 Image Upload", elem_id="image-tab"):
        with gr.Column():
            gr.HTML('<div class="info-section"><strong>📌 How to use:</strong> Upload an image and click "Detect Signs" to analyze</div>')
            with gr.Row():
                image_input = gr.Image(type="numpy", label="Input Image")
                image_output = gr.Image(label="Detected Signs", type="numpy")
            with gr.Row():
                image_btn = gr.Button("🔍 Detect Signs", variant="primary", scale=1)
            image_btn.click(process_frame, inputs=image_input, outputs=image_output)
    
    with gr.Tab("ℹ️ About", elem_id="about-tab"):
        gr.HTML(f"""
        <div style="text-align: center; padding: 20px;">
            <h2>🚦 System Information</h2>
            <div class="info-section">
                <p><strong>Model:</strong> {MODEL_PATH}</p>
                <p><strong>Architecture:</strong> YOLOv11 Nano</p>
                <p><strong>Accuracy:</strong> 99.45% mAP@50</p>
                <p><strong>Size:</strong> {model_size:.2f} MB</p>
            </div>
            
            <div style="margin-top: 20px;">
                <h3>Detected Traffic Signs</h3>
                <div style="background: #f0f2f5; padding: 15px; border-radius: 8px; text-align: left;">
                    <p style="margin: 5px 0;"><strong>Total Classes:</strong> {len(class_names)}</p>
                    <p style="margin: 5px 0; word-break: break-word;"><strong>Signs:</strong> {', '.join(class_names)}</p>
                </div>
            </div>
            
            <div style="margin-top: 20px; opacity: 0.8;">
                <p>🔬 <em>A comprehensive comparative study of traffic sign detection in Bangladesh</em></p>
                <p>📄 <a href="RESEARCH_PAPER.pdf" target="_blank">Read the Research Paper</a></p>
            </div>
        </div>
        """)
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; padding: 20px; opacity: 0.7; border-top: 1px solid #ddd; margin-top: 20px;">
        <p>🚀 Powered by YOLOv11 | 🇧🇩 Bangladesh Traffic Sign Detection | 📊 Highest accuracy & efficiency</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")

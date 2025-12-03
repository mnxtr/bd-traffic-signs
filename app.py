import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import yaml

# Load the trained YOLOv11 model
MODEL_PATH = "results/bd_signs_v1/weights/best.pt"
model = YOLO(MODEL_PATH)

# Load class names from data.yaml
with open("data/processed/data.yaml", "r") as f:
    data_config = yaml.safe_load(f)
    class_names = data_config.get("names", [])

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

# Create Gradio interface with tabs
with gr.Blocks(title="BD Traffic Sign Detection") as demo:
    gr.Markdown("# 🚦 Bangladesh Traffic Sign Detection")
    gr.Markdown("Real-time traffic sign detection using YOLOv11")
    
    with gr.Tab("📹 Webcam (Realtime)"):
        gr.Markdown("### Live webcam detection")
        with gr.Row():
            webcam_input = gr.Image(sources=["webcam"], streaming=True, type="numpy")
            webcam_output = gr.Image()
        webcam_input.stream(process_webcam, inputs=webcam_input, outputs=webcam_output)
    
    with gr.Tab("🎥 Video Upload"):
        gr.Markdown("### Upload a video file for detection")
        video_input = gr.Video()
        video_output = gr.Video()
        video_btn = gr.Button("Process Video")
        video_btn.click(process_video, inputs=video_input, outputs=video_output)
    
    with gr.Tab("📷 Image Upload"):
        gr.Markdown("### Upload an image for detection")
        image_input = gr.Image(type="numpy")
        image_output = gr.Image()
        image_btn = gr.Button("Detect Signs")
        image_btn.click(process_frame, inputs=image_input, outputs=image_output)
    
    gr.Markdown(f"**Model**: {MODEL_PATH}")
    gr.Markdown(f"**Classes**: {', '.join(class_names)}")

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")

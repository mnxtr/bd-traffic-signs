import gradio as gr
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from ultralytics import YOLO
    model_path = project_root / "assets" / "models" / "yolo11n.pt"
    if model_path.exists():
        print(f"Loading model from {model_path}")
        model = YOLO(str(model_path))
    else:
        print("Model not found, downloading yolov8n.pt as fallback")
        model = YOLO("yolov8n.pt")
        
    def predict(image, conf):
        results = model.predict(image, conf=conf)
        # Visualize the results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            return im_array[..., ::-1]  # Convert BGR to RGB
            
    interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="numpy", label="Input Image"),
            gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Confidence Threshold")
        ],
        outputs=gr.Image(type="numpy", label="Detections"),
        title="🇧🇩 Bangladesh Traffic Sign Detection",
        description="Upload an image to detect traffic signs using YOLOv11.",
        examples=[
            [str(project_root / "assets" / "images" / "bus.jpg"), 0.25]
        ]
    )

except ImportError:
    print("Ultralytics not installed. Running in dummy mode.")
    def predict(image, conf):
        return image
        
    interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="numpy", label="Input Image"),
            gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Confidence Threshold")
        ],
        outputs=gr.Image(type="numpy", label="Echo Image (Model not loaded)"),
        title="🇧🇩 Bangladesh Traffic Sign Detection (Demo Mode)",
        description="Ultralytics not found. This is a placeholder demo.",
    )

if __name__ == "__main__":
    interface.launch(share=False)

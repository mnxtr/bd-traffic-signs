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
        if image is None:
            return None
        results = model.predict(image, conf=conf)
        # Visualize the results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            return im_array[..., ::-1]  # Convert BGR to RGB

    with gr.Blocks(title="🇧🇩 Bangladesh Traffic Sign Detection") as demo:
        gr.Markdown("# 🇧🇩 Bangladesh Traffic Sign Detection")
        gr.Markdown("Upload an image or use your webcam to detect traffic signs using YOLOv11.")
        
        with gr.Tabs():
            with gr.TabItem("Image Upload"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(type="numpy", label="Input Image")
                        img_conf = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Confidence Threshold")
                        img_button = gr.Button("Detect")
                    with gr.Column():
                        img_output = gr.Image(type="numpy", label="Detections")
                
                img_button.click(fn=predict, inputs=[img_input, img_conf], outputs=img_output)
                
                gr.Examples(
                    examples=[[str(project_root / "assets" / "images" / "bus.jpg"), 0.25]],
                    inputs=[img_input, img_conf]
                )
            
            with gr.TabItem("Live Webcam"):
                with gr.Row():
                    with gr.Column():
                        # streaming=True enables real-time processing
                        webcam_input = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Webcam Input")
                        webcam_conf = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Confidence Threshold")
                    with gr.Column():
                        webcam_output = gr.Image(type="numpy", label="Real-time Detections")
                
                # Auto-trigger prediction when webcam image changes
                webcam_input.change(fn=predict, inputs=[webcam_input, webcam_conf], outputs=webcam_output)

except ImportError:
    print("Ultralytics not installed. Running in dummy mode.")
    def predict(image, conf):
        return image
        
    with gr.Blocks(title="🇧🇩 Bangladesh Traffic Sign Detection (Demo)") as demo:
        gr.Markdown("# 🇧🇩 Bangladesh Traffic Sign Detection (Demo Mode)")
        gr.Markdown("Ultralytics not found. This is a placeholder demo.")
        with gr.Tabs():
            with gr.TabItem("Image"):
                input_img = gr.Image()
                output_img = gr.Image()
                btn = gr.Button("Run")
                btn.click(lambda x: x, input_img, output_img)

if __name__ == "__main__":
    demo.launch(share=True)

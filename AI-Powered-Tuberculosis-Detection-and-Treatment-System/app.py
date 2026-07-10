import gradio as gr
import os
import cv2
import numpy as np
from inference import infer_and_visualize
from chatbot import TBMedicalAssistant

# Initialize chatbot
chatbot_instance = TBMedicalAssistant()

def predict_image(image):
    if image is None:
        return "Please upload an X-ray image.", "", "", "", "No image provided."
    
    temp_image_path = "temp_uploaded_image.png"
    try:
        # Ensure image is in BGR format
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_image_path, image_bgr)

        diagnosis, severity, sickness_percentage, drugs, disclaimer = infer_and_visualize(temp_image_path)

        return (
            diagnosis,
            severity,
            f"{sickness_percentage:.2f}%" if isinstance(sickness_percentage, (int, float, np.floating)) else "N/A",
            drugs,
            disclaimer
        )
    except Exception as e:
        return f"Error during inference: {str(e)}", "", "", "", "Failed to process image."
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

def chatbot_response(message, history):
    response = chatbot_instance.get_response(message)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history

# Custom CSS
custom_css = """
body {
    background-color: #fffff;
}
.gradio-container {
    max-width: 1400px;
    margin: auto;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1 {
    color: #ffffff;
    text-align: center;
    font-size: 2.5em;
    margin-bottom: 20px;
}
.gr-button {
    background-color: #007acc;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
}
.gr-button:hover {
    background-color: #005fa3;
}
.gr-textbox, .gr-image {
    border-radius: 5px;
    border: 1px solid #ccc;
}
.chatbot-container {
    height: 600px;
    overflow-y: auto;
    border: 1px solid #dcdcdc;
    border-radius: 5px;
    padding: 10px;
}
.disclaimer {
    font-size: 0.9em;
    color: #7f8c8d;
    font-style: italic;
}
"""

# Gradio Interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🩺 Tuberculosis X-Ray Diagnosis & Medical Chat Assistant")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 X-Ray Analysis")
            image_input = gr.Image(type="numpy", label="Upload Chest X-Ray", height=350)
            analyze_button = gr.Button("Analyze Image")

            diagnosis_output = gr.Textbox(label="Diagnosis")
            severity_output = gr.Textbox(label="Severity")
            percentage_output = gr.Textbox(label="Probability")
            medication_output = gr.Textbox(label="Suggested Medication")
            disclaimer_output = gr.Textbox(label="Disclaimer", lines=3, elem_classes="disclaimer")

            analyze_button.click(
                fn=predict_image,
                inputs=[image_input],
                outputs=[
                    diagnosis_output,
                    severity_output,
                    percentage_output,
                    medication_output,
                    disclaimer_output
                ]
            )

        with gr.Column(scale=1):
            gr.Markdown("### 💬 Chat with Medical Assistant")
            chatbot_ui = gr.Chatbot(label="Medical Assistant", type="messages", elem_classes="chatbot-container")
            msg = gr.Textbox(label="Your Message", placeholder="Ask about TB symptoms, diagnosis, or treatment...")
            clear = gr.Button("Clear Chat")

            msg.submit(
                fn=chatbot_response,
                inputs=[msg, chatbot_ui],
                outputs=[msg, chatbot_ui],
                queue=True
            )
            clear.click(
                fn=lambda: [],
                inputs=None,
                outputs=chatbot_ui,
                queue=False
            )

demo.launch(share=True)
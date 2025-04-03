
"""Image Caption Generator
    https://colab.research.google.com/drive/1kFJp07-mZ_Y1KdmxEbHcJugfnWa8eAg3
"""

!pip install transformers torch torchvision pillow requests gradio

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load BLIP model & processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

import gradio as gr

# Function to generate captions
def generate_caption(image):
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption

# Gradio UI
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🖼️ Image Captioning with BLIP",
    description="Upload an image and get an AI-generated caption.",
    allow_flagging="never"
)

# Launch UI
iface.launch(share=True)

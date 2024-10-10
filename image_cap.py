import gradio as gr
import numpy as np
import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def image_caption(input_image: np.ndarray):
    raw_image=Image.fromarray(input_image).convert("RGB")
    text = "The image of "
    inputs = processor(images=raw_image, text=text, return_tensors="pt")
    output = model.generate(**inputs, max_length=50)
    caption=processor.decode(output[0], skip_special_tokens=True)
    return caption

iface = gr.Interface(
    fn=image_caption, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning Dany2994",
    description="Excercise and approach to image captioning with IA technology, utilizing tools from HuggingFace and Gradio:\
        By using \"AUTOPROCESSOR\"and \"BLIPFORCONDITIONALGENERATION\" from HuggingFace and Python Image Library (PIL) to create the image captioning programm and using \"GRADIO\" to easily create the programm interface.\
            Please upload your image and press \"Submit\",  you should see a brief description of your uploaded image at the right side of the screen.")

iface.launch(share=True)

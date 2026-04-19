import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("deepfake_model.h5")

def predict(img):
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    confidence = pred if pred > 0.5 else (1 - pred)
    label = "Fake" if pred > 0.5 else "Real"

    return f"Prediction: {label}\nConfidence: {confidence*100:.2f}%"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Deepfake Detection System"
)

interface.launch()
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Set up page configuration
st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection System")
st.write("Upload an image of a plant leaf to identify potential diseases.")

# 2. Load the trained model
@st.cache_resource
def load_my_model():
    # compile=False avoids loading training-specific configurations, saving memory
    return tf.keras.models.load_model('plant_disease_mobilenetv2.h5', compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model file: {e}")

# 3. Dynamic list of all 38 classes loaded from your new labels file
try:
    with open('labels.txt', 'r') as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    st.error("Error: 'labels.txt' file not found in the repository! Please upload it to GitHub.")
    CLASS_NAMES = []

# 4. File Uploader UI
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and len(CLASS_NAMES) > 0:
    # Display the uploaded image (Using width='stretch' to avoid deprecated layout warnings)
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width='stretch')
    
    st.write("🔄 Analyzing the image...")
    
    # Preprocess the image exactly like your training pipeline
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # Handle cases where image might be RGBA instead of RGB
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    # Scale pixels (1./255) and add batch dimension
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Make prediction safely using NumPy arrays to prevent UI freezes
    predictions = model.predict(img_array)
    
    # Compute manual softmax using pure numpy to ensure instant UI rendering
    exp_preds = np.exp(predictions[0] - np.max(predictions[0]))
    probabilities = exp_preds / exp_preds.sum()
    
    predicted_class_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_class_idx] * 100
    
    # 6. Output the results instantly
    st.subheader(f"Result: **{CLASS_NAMES[predicted_class_idx]}**")
    st.info(f"Confidence Level: **{confidence:.2f}%**")

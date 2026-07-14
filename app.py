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
    return tf.keras.models.load_model('plant_disease_mobilenetv2.h5', compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model file: {e}")

# 3. Explicit list of all 38 classes
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# 4. File Uploader UI
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image (Updated to 'stretch' to fix the layout warning)
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width='stretch')
    
    st.write("🔄 Analyzing the image...")
    
    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Make prediction safely using NumPy arrays
    predictions = model.predict(img_array)
    
    # Compute probabilities purely using NumPy to prevent UI freezes
    exp_preds = np.exp(predictions[0] - np.max(predictions[0]))
    probabilities = exp_preds / exp_preds.sum()
    
    predicted_class_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_class_idx] * 100
    
    # 6. Output the results instantly
    st.subheader(f"Result: **{CLASS_NAMES[predicted_class_idx]}**")
    st.info(f"Confidence Level: **{confidence:.2f}%**")

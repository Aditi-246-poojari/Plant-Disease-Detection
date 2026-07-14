import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

# Configure page settings
st.set_page_config(
    page_title="Botanical Health Diagnoser", 
    page_icon="🌱", 
    layout="centered"
)

st.title("🌱 Botanical Health Diagnoser")
st.write("Upload a photo of a plant leaf to detect potential diseases instantly.")

# 1. Load the active trained H5 model structure matrix safely
@st.cache_resource
def load_diagnostic_model():
    # Load model weights without rebuilding dynamic training configurations
    model = tf.keras.models.load_model('botanical_diagnostic_mobilenetv2.h5', compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if os.path.exists('botanical_diagnostic_mobilenetv2.h5'):
    classifier_model = load_diagnostic_model()
else:
    st.error("Missing 'botanical_diagnostic_mobilenetv2.h5'. Make sure your model file is pushed to the root of your GitHub repository.")
    st.stop()

# 2. Dynamically load index configurations map
if os.path.exists('botanical_labels.json'):
    with open('botanical_labels.json', 'r') as f:
        labels_dict = json.load(f)
    DISEASE_CLASSES = [k for k, v in sorted(labels_dict.items(), key=lambda item: item[1])]
else:
    st.error("Missing 'botanical_labels.json'. Make sure your labels JSON file is pushed to your GitHub repository.")
    st.stop()

# 3. Handle live image upload assets
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    user_image = Image.open(uploaded_file)
    st.image(user_image, caption="Uploaded Leaf Specimen", use_container_width=True)
    
    with st.spinner("🔄 Running diagnostics matrix inference..."):
        # Standard image pre-processing configurations pipeline
        resized_img = user_image.resize((224, 224)).convert('RGB')
        img_array = np.array(resized_img)
        expanded_tensor = np.expand_dims(img_array, axis=0)
        normalized_tensor = expanded_tensor / 255.0
        
        # Calculate inference probability metrics matrix
        prediction_scores = classifier_model.predict(normalized_tensor)
        highest_idx = np.argmax(prediction_scores[0])
        confidence = prediction_scores[0][highest_idx] * 100
        
        # Clean classification labels to match standard formatting parameters
        predicted_label = DISEASE_CLASSES[highest_idx].replace("___", " - ").replace("_", " ")

    # Display findings layout panels
    st.success(f"**Diagnosis:** {predicted_label}")
    st.info(f"**Confidence Score:** {confidence:.2f}%")
    st.progress(int(confidence))
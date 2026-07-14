import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

st.set_page_config(
    page_title="Botanical Health Diagnoser", 
    page_icon="🌱", 
    layout="centered"
)

st.title("🌱 Botanical Health Diagnoser")
st.write("Upload a photo of a plant leaf to detect potential diseases instantly.")

# Load the TFLite model instead of H5
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="botanical_diagnostic_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

if os.path.exists('botanical_diagnostic_model.tflite'):
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    st.error("Missing 'botanical_diagnostic_model.tflite'. Ensure it is pushed to the root of your GitHub repository.")
    st.stop()

if os.path.exists('botanical_labels.json'):
    with open('botanical_labels.json', 'r') as f:
        labels_dict = json.load(f)
    DISEASE_CLASSES = [k for k, v in sorted(labels_dict.items(), key=lambda item: item[1])]
else:
    st.error("Missing 'botanical_labels.json'. Make sure your labels JSON file is pushed to your GitHub repository.")
    st.stop()

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    user_image = Image.open(uploaded_file)
    st.image(user_image, caption="Uploaded Leaf Specimen", use_container_width=True)
    
    with st.spinner("🔄 Running diagnostics..."):
        resized_img = user_image.resize((224, 224)).convert('RGB')
        img_array = np.array(resized_img, dtype=np.float32)
        expanded_tensor = np.expand_dims(img_array, axis=0)
        normalized_tensor = expanded_tensor / 255.0
        
        interpreter.set_tensor(input_details[0]['index'], normalized_tensor)
        interpreter.invoke()
        
        prediction_scores = interpreter.get_tensor(output_details[0]['index'])
        highest_idx = np.argmax(prediction_scores[0])
        confidence = prediction_scores[0][highest_idx] * 100
        
        predicted_label = DISEASE_CLASSES[highest_idx].replace("___", " - ").replace("_", " ")

    st.success(f"**Diagnosis:** {predicted_label}")
    st.info(f"**Confidence Score:** {confidence:.2f}%")
    st.progress(int(confidence))

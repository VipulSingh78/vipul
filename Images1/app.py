import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests

# Streamlit app title
st.title('Welcome to Apna Electrician')
st.subheader('Upload or capture an image of a product to test model classification.')

# Model URL and local filename
model_url = 'https://github.com/VipulSingh78/vipul/raw/419d4fa1249bd95181d259c202df4e36d873f0c0/Images1/Vipul_Recog_Model.h5'
model_filename = os.path.join('Models', 'Vipul_Recog_Model.h5')
os.makedirs('Models', exist_ok=True)

# Download model if it doesn't exist
def download_model():
    if not os.path.exists(model_filename):
        try:
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                with open(model_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            st.error(f"Error downloading the model: {e}")

download_model()

# Load model with error handling
try:
    model = load_model(model_filename)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Model is None if loading fails

# Product names for validation
product_names = ['Anchor Switch','CCTV CAMERA', 'FAN', 'Switch', 'TV']

# Simplified classification function
def classify_image(image_path):
    if model is None:
        return "Model failed to load."

    # Image preprocessing
    input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    # Predict using the model
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(result)

    # Check confidence level and print class only if confident
    if result[predicted_class_index] < 0.5:
        return "Error: Image doesn't match known products with high confidence."
    
    if 0 <= predicted_class_index < len(product_names):
        return f"The image matches: {product_names[predicted_class_index]}"
    else:
        return "Error: Prediction out of range."

# Streamlit file uploader and camera input
st.markdown("### Upload or capture an image to classify:")
uploaded_file = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])
captured_image = st.camera_input("Capture Image")

# Choose captured image or uploaded file
image_data = uploaded_file if uploaded_file else captured_image

if image_data is not None:
    # Save and display image
    save_path = os.path.join('upload', uploaded_file.name if uploaded_file else "captured_image.png")
    os.makedirs('upload', exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(image_data.getbuffer() if uploaded_file else captured_image.getvalue())

    st.image(image_data, use_column_width=True)

    # Run classification
    try:
        result = classify_image(save_path)
        st.success(result)
    except Exception as e:
        st.error(f"Classification error: {e}")

    # Clear image
    if st.button("Clear Image"):
        uploaded_file = None
        st.experimental_rerun()

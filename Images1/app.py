import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests

# Streamlit app title
st.title('Welcome to Apna Electrician')
st.subheader('Upload or capture an image of a product, and get recommendations!')

# Product names and links
product_names = ['Anchor Switch', 'CCTV CAMERA', 'FAN', 'Switch', 'TV']
product_links = {
    'Anchor Switch': 'https://www.apnaelectrician.com/anchor-switches',
    'CCTV CAMERA': 'https://www.apnaelectrician.com/cctv-cameras',
    'FAN': 'https://www.apnaelectrician.com/fans',
    'Switch': 'https://www.apnaelectrician.com/switches',
    'TV': 'https://www.apnaelectrician.com/tvs'
}

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
                        if chunk:
                            f.write(chunk)
            st.write("Model downloaded successfully.")
        except Exception as e:
            st.error(f"Error downloading the model: {e}")

# Load the model
download_model()
try:
    model = load_model(model_filename)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Function for image classification with added debugging
def classify_images(image_path, confidence_threshold=0.7):
    if model is None:
        st.error("Model is not loaded properly.")
        return None

    try:
        input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)
        st.write("Image loaded and preprocessed successfully.")
    except Exception as e:
        st.error(f"Error loading and preprocessing image: {e}")
        return None

    try:
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        predicted_class_index = np.argmax(result)
        predicted_confidence = result[predicted_class_index]
        
        st.write(f"Prediction Confidence: {predicted_confidence}")
        
        if predicted_confidence < confidence_threshold:
            return "Error: Low confidence for product identification."

        if 0 <= predicted_class_index < len(product_names):
            predicted_class = product_names[predicted_class_index]
        else:
            return "Error: Predicted class index out of range."

        buy_link = product_links.get(predicted_class, 'https://www.apnaelectrician.com/')
        
        return f'The image belongs to {predicted_class}. [Buy here]({buy_link})'
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Capture or upload image and process
st.markdown("### Upload your image below or capture directly from camera:")
uploaded_file = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])
captured_image = st.camera_input("Capture Image")

# Save and display the image
if uploaded_file:
    st.write("Uploaded image selected.")
    image_path = os.path.join('upload', uploaded_file.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
elif captured_image:
    st.write("Captured image selected.")
    image_path = os.path.join('upload', "captured_image.png")
    with open(image_path, 'wb') as f:
        f.write(captured_image.getbuffer())
else:
    st.warning("Please upload or capture an image.")
    image_path = None

# If an image is available, classify it
if image_path:
    st.image(image_path, use_column_width=True)
    try:
        result = classify_images(image_path)
        if result:
            st.success(result)
        else:
            st.error("Error: The model could not classify the image.")
    except Exception as e:
        st.error(f"Error in classification: {e}")
else:
    st.warning("No image available for classification.")

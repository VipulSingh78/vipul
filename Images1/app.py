import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from twilio.rest import Client
import requests

# Twilio credentials
account_sid = 'AC093d4d6255428d338c2f3edc10328cf7'
auth_token = '40d3d53464a816fb6de7855a640c4194'
client = Client(account_sid, auth_token)

# Streamlit app title
st.title('Welcome to Apna Electrician')
st.subheader('Upload or capture an image of a product, and get recommendations!')

# Model URL and local filename
model_url = 'https://github.com/VipulSingh78/vipul/raw/419d4fa1249bd95181d259c202df4e36d873f0c0/Images1/Vipul_Recog_Model.h5'
model_filename = os.path.join('Models', 'Vipul_Recog_Model.h5')

# Ensure the Models and upload directories exist
os.makedirs('Models', exist_ok=True)
os.makedirs('upload', exist_ok=True)

# Function to download the model if it doesn't exist
def download_model():
    if not os.path.exists(model_filename):
        try:
            with st.spinner('Downloading the model...'):
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                with open(model_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            st.success('Model downloaded successfully.')
        except Exception as e:
            st.error(f"Error downloading the model: {e}")

# Download the model
download_model()

# **LOAD THE MODEL** - Load the model globally
try:
    model = load_model(model_filename)  # Load the model from the saved file
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Ensure the model is None if loading fails

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.80

# Define actual class labels (replace these with your model's classes)
class_labels = {
    0: "CCTV CAMERA", 
    1: "Switch", 
    2: "Fan", 
    # Add more classes as needed
}

# Image classification function
def classify_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize for model
    img_array = image.img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)
    
    # Check if predicted class exists in `class_labels` and meets confidence threshold
    if confidence >= CONFIDENCE_THRESHOLD and predicted_class in class_labels:
        class_name = class_labels[predicted_class]
        return class_name, confidence
    else:
        return "Error", confidence

# Streamlit camera input and file uploader
st.markdown("### Upload your image below or capture directly from camera:")
uploaded_file = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])
captured_image = st.camera_input("Capture Image")

# Directory where images will be saved
upload_folder = 'upload'
os.makedirs(upload_folder, exist_ok=True)  # Ensure the upload directory exists

# Function to save image and return the path
def save_image(image_file, filename):
    try:
        image_path = os.path.join(upload_folder, filename)
        with open(image_path, 'wb') as f:
            f.write(image_file.getbuffer())
        st.write(f"Image saved at {image_path}")
        return image_path
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None

# Initialize image_path
image_path = None

# Handle uploaded file
if uploaded_file is not None:
    st.write("Uploaded image selected.")
    image_path = save_image(uploaded_file, uploaded_file.name)

# Handle captured image
elif captured_image is not None:
    st.write("Captured image selected.")
    image_path = save_image(captured_image, "captured_image.png")

# If an image is available, display and classify it
if image_path:
    try:
        # Display the image
        st.image(image_path, caption='Uploaded/Captured Image', use_column_width=True)

        # Classify the image
        class_name, confidence = classify_image(image_path)

        if class_name == "Error":
            st.error("The image doesn't match any known product with high confidence.")
        else:
            st.success(f"The image belongs to {class_name}.")
            st.write(f"Predicted Confidence: {confidence:.2f}")
            
            # Add a "Buy here" link for products
            st.markdown(f"[Buy here](https://example.com/{class_name.replace(' ', '_')})")
    except Exception as e:
        st.error(f"Error processing image: {e}")

    # Clear Image button
    if st.button("Clear Image"):
        # Clear the uploaded and captured image variables
        uploaded_file = None
        captured_image = None
        image_path = None
        st.experimental_rerun()

# If no image is provided, show a warning
else:
    st.warning("Please upload an image or capture one using the camera.")

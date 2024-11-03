import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from twilio.rest import Client
import requests

# Twilio credentials (use secure environment variables in production)
account_sid = 'AC093d4d6255428d338c2f3edc10328cf7'
auth_token = '40d3d53464a816fb6de7855a640c4194'
client = Client(account_sid, auth_token)

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
model_filename = os.path.join('Models', 'Images1/Vipul_Recog_Model.h5')

# Ensure the Models and upload directories exist
os.makedirs('Models', exist_ok=True)
os.makedirs('upload', exist_ok=True)

# Function to download the model if it doesn't exist
def download_model():
    # Check if model exists and attempt to load it; if loading fails, delete and redownload
    if os.path.exists(model_filename):
        try:
            test_model = load_model(model_filename)
            st.success("Model loaded successfully.")
            return  # Model is already downloaded and valid
        except Exception as e:
            st.warning("Corrupted model file detected. Re-downloading...")
            os.remove(model_filename)  # Remove corrupted file
    
    # Download the model if it's not present or if it was corrupted
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

# Image classification function with confidence threshold
def classify_images(image_path, confidence_threshold=0.8):
    if model is None:
        st.error("Model is not loaded properly.")
        return None

    try:
        # Load and preprocess the image
        input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)
        st.write("Image loaded and preprocessed successfully.")
    except Exception as e:
        st.error(f"Error loading and preprocessing image: {e}")
        return None

    try:
        # Predict using the model
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        predicted_class_index = np.argmax(result)
        predicted_confidence = result[predicted_class_index]

        # Display confidence for debugging
        st.write(f"Predicted Confidence: {predicted_confidence:.2f}")

        # Check confidence level and show error if below threshold
        if predicted_confidence < confidence_threshold:
            return f"Error: The image doesn't match any known product with high confidence. (Confidence: {predicted_confidence:.2f})"

        # Retrieve predicted class and corresponding buy link
        if 0 <= predicted_class_index < len(product_names):
            predicted_class = product_names[predicted_class_index]
        else:
            return "Error: Predicted class index out of range."

        buy_link = product_links.get(predicted_class, 'https://www.apnaelectrician.com/')

        # Send WhatsApp message
        send_whatsapp_message(predicted_class, buy_link)

        return f'The image belongs to **{predicted_class}**. [Buy here]({buy_link})'
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# WhatsApp message function
def send_whatsapp_message(predicted_class, buy_link):
    try:
        # Publicly hosted image URL (replace with actual hosted URL)
        # Ensure that the image is uploaded to a public URL accessible by Twilio
        media_url = [f'https://your-public-image-url.com/{os.path.basename(image_path)}']

        message = client.messages.create(
            from_='whatsapp:+14155238886',  # Twilio sandbox number
            body=f"Classification Result: {predicted_class}. Buy here: {buy_link}",
            media_url=media_url,  # Public image URL
            to='whatsapp:+917800905998'  # Replace with your WhatsApp number
        )
        st.write("WhatsApp message sent successfully.")
    except Exception as e:
        st.error(f"Error sending WhatsApp message: {e}")

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
        st.image(image_path, caption='Uploaded/Captured Image', use_column_width=True)
        result = classify_images(image_path)
        if result:
            if "Error" in result:
                st.warning(result)
            else:
                st.success(result)
    except Exception as e:
        st.error(f"Error processing image: {e}")

    # Clear Image button
    if st.button("Clear Image"):
        uploaded_file = None
        captured_image = None
        image_path = None
        st.experimental_rerun()

# If no image is provided, show a warning
else:
    st.warning("Please upload an image or capture one using the camera.")

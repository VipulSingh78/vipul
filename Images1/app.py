import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from twilio.rest import Client
import requests
import base64
from io import BytesIO
from PIL import Image

# Twilio credentials (use secure environment variables in production)
account_sid = 'AC093d4d6255428d338c2f3edc10328cf7'
auth_token = '40d3d53464a816fb6de7855a640c4194'
client = Client(account_sid, auth_token)

# Streamlit app title
st.title('Welcome to Apna Electrician')
st.subheader('Upload an image of a product, and get recommendations!')

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

# Function to download model if it doesn't exist
def download_model():
    if not os.path.exists(model_filename):
        try:
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                with open(model_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            st.error(f"Error downloading the model: {e}")

# Download the model
download_model()

# Load the model
try:
    model = load_model(model_filename)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Classification function
def classify_image(image, confidence_threshold=0.7):
    input_image = image.resize((224, 224))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(result)
    predicted_confidence = result[predicted_class_index]

    if predicted_confidence < confidence_threshold:
        return "Error: The image doesn't match any known product with high confidence."

    predicted_class = product_names[predicted_class_index]
    buy_link = product_links.get(predicted_class, 'https://www.apnaelectrician.com/')

    return f'The image belongs to {predicted_class}. [Buy here]({buy_link})'

# HTML & JavaScript for accessing the camera
st.markdown("""
    <h3>Camera Input</h3>
    <p>Click "Capture" to take a photo and classify it.</p>
    <video id="video" width="100%" height="300" autoplay></video>
    <button onclick="capture()">Capture</button>
    <canvas id="canvas" style="display:none;"></canvas>
    <script>
        var video = document.getElementById('video');
        var canvas = document.getElementById('canvas');
        var context = canvas.getContext('2d');

        // Get access to the camera
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => { video.srcObject = stream; })
            .catch(error => { console.log("Error accessing camera: " + error); });

        function capture() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
            var dataUrl = canvas.toDataURL('image/png');
            var imgBase64 = dataUrl.split(',')[1];
            const imageInput = window.parent.document.getElementById('image_data');
            imageInput.value = imgBase64;
            window.parent.document.getElementById('submit_button').click();
        }
    </script>
""", unsafe_allow_html=True)

# Hidden input for storing the captured image data
image_data = st.text_input("Image Data (Base64)", value="", key="image_data", type="hidden")

# Button for submitting the captured image
if st.button("Classify Image", key="submit_button"):
    if image_data:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Classify the image
        result = classify_image(image)
        st.success(result)
    else:
        st.error("Please capture an image first.")

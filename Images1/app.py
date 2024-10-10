import os
import streamlit as st
import cv2
import numpy as np
import base64
import tempfile
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
st.subheader('Upload an image of a product or use your camera, and get recommendations!')

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

# Function to classify images
def classify_images(image_path):
    if model is None:
        return "Model is not loaded properly."
    
    input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)
    
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(result)
    
    if 0 <= predicted_class_index < len(product_names):
        predicted_class = product_names[predicted_class_index]
    else:
        return "Error: Predicted class index out of range."
    
    buy_link = product_links.get(predicted_class, 'https://www.apnaelectrician.com/')
    send_whatsapp_message(image_path, predicted_class, buy_link)
    
    return f'The image belongs to {predicted_class}. [Buy here]({buy_link})'

# WhatsApp message function
def send_whatsapp_message(image_path, predicted_class, buy_link):
    try:
        media_url = [f'https://your-public-image-url.com/{os.path.basename(image_path)}']
        message = client.messages.create(
            from_='whatsapp:+14155238886',  # Twilio number
            body=f"Classification Result: {predicted_class}. Buy here: {buy_link}",
            media_url=media_url,
            to='whatsapp:+917800905998'
        )
        print("WhatsApp message sent successfully:", message.sid)
    except Exception as e:
        print("Error sending WhatsApp message:", e)

# Camera input
def camera_input():
    st.markdown(
        """
        <video autoplay playsinline></video>
        <button id="capture">Capture</button>
        <canvas style="display:none;"></canvas>

        <script>
        const video = document.querySelector('video');
        const canvas = document.querySelector('canvas');
        const button = document.getElementById('capture');

        navigator.mediaDevices.getUserMedia({ video: true })
          .then(stream => video.srcObject = stream);

        button.addEventListener('click', () => {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext('2d').drawImage(video, 0, 0);

          const dataURL = canvas.toDataURL('image/png');
          window.streamlitAPI.setComponentValue(dataURL);
        });
        </script>
        """,
        unsafe_allow_html=True,
    )

# File uploader for gallery
uploaded_file = st.file_uploader('Choose an image from Gallery', type=['jpg', 'jpeg', 'png'])

# Handle uploaded file or camera input
if uploaded_file is not None:
    save_path = os.path.join('upload', uploaded_file.name)
    os.makedirs('upload', exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, use_column_width=True)

    try:
        result = classify_images(save_path)
        st.success(result)
    except Exception as e:
        st.error(f"Error in classification: {e}")

# Show camera capture option
st.markdown("### Or use your Camera:")
image_data = camera_input()

if image_data:
    image_bytes = image_data.split(",")[1]
    image_array = np.frombuffer(base64.b64decode(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    save_path = os.path.join('upload', 'captured_image.png')
    cv2.imwrite(save_path, image)
    
    st.image(image, caption="Captured Image", use_column_width=True)

    try:
        result = classify_images(save_path)
        st.success(result)
    except Exception as e:
        st.error(f"Error in classification: {e}")

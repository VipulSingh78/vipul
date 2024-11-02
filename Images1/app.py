import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from twilio.rest import Client
import requests

# Twilio credentials (replace with environment variables in production)
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
model_filename = os.path.join('Models', 'Vipul_Recog_Model.h5')

# Ensure directories exist
os.makedirs('Models', exist_ok=True)
os.makedirs('upload', exist_ok=True)

# Download model if not present
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

download_model()

# Load model
try:
    model = load_model(model_filename)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Image classification function
def classify_image(image_path, confidence_threshold=0.8):
    if model is None:
        st.error("Model is not loaded.")
        return None

    try:
        # Load and preprocess the image
        input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        input_image_array = tf.keras.utils.img_to_array(input_image) / 255.0
        input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

        # Predict
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        predicted_class_index = np.argmax(result)
        predicted_confidence = result[predicted_class_index]

        # Debugging: Display prediction details
        st.write(f"Prediction Results: {result}")
        st.write(f"Predicted Class Index: {predicted_class_index}")
        st.write(f"Predicted Confidence: {predicted_confidence:.2f}")

        if predicted_confidence < confidence_threshold:
            return f"Low confidence ({predicted_confidence:.2f}). Unable to classify with high confidence."

        # Get predicted class name and link
        predicted_class = product_names[predicted_class_index]
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
        message = client.messages.create(
            from_='whatsapp:+14155238886',
            body=f"Classification Result: {predicted_class}. Buy here: {buy_link}",
            to='whatsapp:+917800905998'
        )
        st.write("WhatsApp message sent successfully.")
    except Exception as e:
        st.error(f"Error sending WhatsApp message: {e}")

# File uploader and camera input
st.markdown("### Upload your image below or capture directly from camera:")
uploaded_file = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])
captured_image = st.camera_input("Capture Image")

# Directory for saving images
upload_folder = 'upload'
os.makedirs(upload_folder, exist_ok=True)

def save_image(image_file, filename):
    try:
        image_path = os.path.join(upload_folder, filename)
        with open(image_path, 'wb') as f:
            f.write(image_file.getbuffer())
        return image_path
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None

# Handle uploaded or captured image
image_path = None
if uploaded_file is not None:
    image_path = save_image(uploaded_file, uploaded_file.name)
elif captured_image is not None:
    image_path = save_image(captured_image, "captured_image.png")

# Display and classify image if available
if image_path:
    st.image(image_path, caption='Uploaded/Captured Image', use_column_width=True)
    result = classify_image(image_path)

    if result:
        if "Low confidence" in result:
            st.warning(result)
        else:
            st.success(result)
else:
    st.warning("Please upload or capture an image.")

# Clear Image button
if st.button("Clear Image"):
    uploaded_file = None
    captured_image = None
    image_path = None
    st.experimental_rerun()

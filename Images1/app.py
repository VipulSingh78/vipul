import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from twilio.rest import Client
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Twilio credentials
TWILIO_SID = 'AC093d4d6255428d338c2f3edc10328cf7'
TWILIO_AUTH_TOKEN = '40d3d53464a816fb6de7855a640c4194'
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Email credentials
SENDER_EMAIL = "vipulsinghvipul7@gmail.com"  # Replace with your Gmail ID
SENDER_PASSWORD = "zazb kspg ecjd brol"  # Replace with Gmail App Password
RECEIVER_EMAIL = "vipulsinghvipul7@gmail.com"  # Use your email as receiver

# Streamlit app setup
st.title('Welcome to Apna Electrician')
st.subheader('Upload or capture an image of a product, and get recommendations!')

# Product details
PRODUCT_NAMES = ['Anchor Switch', 'CCTV CAMERA', 'FAN', 'Switch', 'TV']
PRODUCT_LINKS = {
    'Anchor Switch': 'https://www.apnaelectrician.com/anchor-switches',
    'CCTV CAMERA': 'https://www.apnaelectrician.com/cctv-cameras',
    'FAN': 'https://www.apnaelectrician.com/fans',
    'Switch': 'https://www.apnaelectrician.com/switches',
    'TV': 'https://www.apnaelectrician.com/tvs'
}

# Model download setup
MODEL_URL = 'https://github.com/VipulSingh78/vipul/raw/419d4fa1249bd95181d259c202df4e36d873f0c0/Images1/Vipul_Recog_Model.h5'
MODEL_PATH = os.path.join('Models', 'Vipul_Recog_Model.h5')
os.makedirs('Models', exist_ok=True)

# Function to download the model
def download_model():
    if not os.path.exists(MODEL_PATH):
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
        except Exception as e:
            st.error(f"Failed to download the model: {e}")

# Download and load the model
download_model()
model = None
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Model loading failed: {e}")

# Function to classify images
def classify_images(image_path, user_message, confidence_threshold=0.5):
    if model is None:
        return "Model is not loaded."
    
    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    confidence_scores = tf.nn.softmax(predictions[0])
    class_index = np.argmax(confidence_scores)
    confidence = confidence_scores[class_index]
    
    if confidence < confidence_threshold:
        return "Error: Low confidence. The image doesn't match any known product."
    
    if class_index < len(PRODUCT_NAMES):
        predicted_class = PRODUCT_NAMES[class_index]
        buy_link = PRODUCT_LINKS.get(predicted_class, 'https://www.apnaelectrician.com/')
        send_email(image_path, predicted_class, buy_link, user_message)
        send_whatsapp_message(predicted_class, buy_link)
        return f"{predicted_class} detected. [Buy here]({buy_link})"
    else:
        return "Error: Predicted class out of range."

# Function to send WhatsApp messages
def send_whatsapp_message(predicted_class, buy_link):
    try:
        client.messages.create(
            from_='whatsapp:+14155238886',
            to='whatsapp:+917800905998',
            body=f"Product detected: {predicted_class}. Buy it here: {buy_link}"
        )
        print("WhatsApp message sent successfully!")
    except Exception as e:
        print(f"Failed to send WhatsApp message: {e}")

# Function to send email
def send_email(image_path, predicted_class, buy_link, user_message):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = "New Product Image Uploaded"
        msg.attach(MIMEText(f"Detected: {predicted_class}. Buy here: {buy_link}\nUser Message: {user_message}", 'plain'))

        with open(image_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
            msg.attach(part)

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# File uploader and camera input
st.markdown("### Upload your image below or capture directly from camera:")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Message box for user input
user_message = st.text_area("Enter a message for the electrician:")

# Handle image upload and capture
if st.button("Capture Image"):
    captured_image = st.camera_input("Capture an image")

    # Assign image_data based on user actions (upload or capture)
    if captured_image:
        image_data = captured_image
    elif uploaded_file:
        image_data = uploaded_file
    else:
        image_data = None  # No image provided or captured

    if image_data:
        os.makedirs('upload', exist_ok=True)
        # Use uploaded file name or a default name if captured image is used
        save_path = os.path.join('upload', uploaded_file.name if uploaded_file else 'captured_image.png')

        with open(save_path, 'wb') as f:
            f.write(image_data.getbuffer())

        st.image(image_data, caption="Uploaded Image", use_column_width=True)
        result = classify_images(save_path, user_message)
        st.success(result)

    else:
        st.error("Please upload or capture an image before proceeding.")

    if st.button("Clear"):
        st.experimental_rerun()

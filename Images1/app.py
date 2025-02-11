import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from twilio.rest import Client
import requests

# Email credentials
SENDER_EMAIL = "vipulsinghvipul7@gmail.com"  # Replace with your Gmail ID
SENDER_PASSWORD = "zazb kspg ecjd brol"  # Replace with Gmail App Password
RECEIVER_EMAIL = "vipulsinghvipul7@gmail.com"  # Use your email as receiver

# Twilio credentials
TWILIO_SID = 'AC093d4d6255428d338c2f3edc10328cf7'
TWILIO_AUTH_TOKEN = '40d3d53464a816fb6de7855a640c4194'
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Streamlit app setup
st.title('Welcome to Apna Electrician')
st.subheader('Upload or capture an image of a product, and get recommendations!')

# Model setup
MODEL_PATH = os.path.join('Models', 'Vipul_Recog_Model.h5')
os.makedirs('Models', exist_ok=True)

def load_classification_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_classification_model()

# Function to classify images
def classify_images(image_path, confidence_threshold=0.5):
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
    
    return f"Detected class index: {class_index}, Confidence: {confidence:.2f}"

# Function to send email with an image attachment
def send_email_with_image(image_path, subject="Product Image Upload", message="Attached is the uploaded image."):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        if os.path.exists(image_path):
            with open(image_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
                msg.attach(part)
        else:
            print("Error: Image file not found.")
            return

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Streamlit interface for file upload
st.markdown("### Upload your image or capture one directly below:")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file and st.button("Upload and Send"):
    os.makedirs('upload', exist_ok=True)
    save_path = os.path.join('upload', uploaded_file.name)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(save_path, caption="Uploaded Image", use_container_width=True)
    result = classify_images(save_path)
    st.success(result)
    send_email_with_image(save_path)
    st.success("Image sent via email!")

if st.button("Clear"):
    st.experimental_rerun()

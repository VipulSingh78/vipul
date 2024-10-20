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

# **LOAD THE MODEL** - Load the model globally
try:
    model = load_model(model_filename)  # Load the model from the saved file
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Ensure the model is None if loading fails

# Image classification function with stricter confidence threshold and class probabilities
def classify_images(image_path, confidence_threshold=0.5):  # Set confidence threshold
    if model is None:
        return "Model is not loaded properly."

    input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_array = tf.keras.applications.mobilenet_v2.preprocess_input(input_image_array)
    input_image_exp_dim = np.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])

    # Print all class probabilities
    st.write("Class Probabilities:")
    for i, prob in enumerate(result):
        st.write(f"{product_names[i]}: {prob*100:.2f}%")
    
    # Get top predicted class and confidence
    predicted_class_index = np.argmax(result)
    confidence = result[predicted_class_index]  # Get confidence for the top prediction

    # If confidence is below the threshold, suggest customer support
    if confidence < confidence_threshold:
        return f"The product is not recognized with high confidence. Please contact customer support at +917800905998."

    # Check if the predicted class index is valid
    if 0 <= predicted_class_index < len(product_names):
        predicted_class = product_names[predicted_class_index]
        buy_link = product_links.get(predicted_class, 'https://www.apnaelectrician.com/')
        send_whatsapp_message(image_path, predicted_class, buy_link)
        return f'The image belongs to {predicted_class}. [Buy here]({buy_link})'
    else:
        return "The product is not recognized. Please contact customer support at +917800905998."

# WhatsApp message function
def send_whatsapp_message(image_path, predicted_class, buy_link):
    try:
        # Publicly hosted image URL (replace with actual hosted URL)
        media_url = [f'https://your-public-image-url.com/{os.path.basename(image_path)}']

        message = client.messages.create(
            from_='whatsapp:+14155238886',  # Twilio number
            body=f"Classification Result: {predicted_class}. Buy here: {buy_link}",
            media_url=media_url,  # Public image URL
            to='whatsapp:+917800905998'
        )
        print("WhatsApp message sent successfully:", message.sid)
    except Exception as e:
        print("Error sending WhatsApp message:", e)

# Streamlit file uploader
st.markdown("### Upload your image below:")
uploaded_file = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])

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

    if st.button("Clear Image"):
        uploaded_file = None
        st.experimental_rerun()

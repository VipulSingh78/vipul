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
    model = load_model(model_filename)  # Load the model from the saved file
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Ensure the model is None if loading fails

# Image classification function with confidence threshold
def classify_image_array(image_array, confidence_threshold=0.5):
    if model is None:
        return "Model is not loaded properly."

    input_image_exp_dim = tf.expand_dims(image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(result)
    predicted_confidence = result[predicted_class_index]
    
    # Check confidence level
    if predicted_confidence < confidence_threshold:
        return "Error: The image doesn't match any known product with high confidence."

    if 0 <= predicted_class_index < len(product_names):
        predicted_class = product_names[predicted_class_index]
    else:
        return "Error: Predicted class index out of range."

    buy_link = product_links.get(predicted_class, 'https://www.apnaelectrician.com/')
    send_whatsapp_message(predicted_class, buy_link)
    
    return f'The image belongs to {predicted_class}. [Buy here]({buy_link})'

# WhatsApp message function
def send_whatsapp_message(predicted_class, buy_link):
    try:
        message = client.messages.create(
            from_='whatsapp:+14155238886',  # Twilio number
            body=f"Classification Result: {predicted_class}. Buy here: {buy_link}",
            to='whatsapp:+917800905998'
        )
        print("WhatsApp message sent successfully:", message.sid)
    except Exception as e:
        print("Error sending WhatsApp message:", e)

# Streamlit file uploader and camera capture button
st.markdown("### Upload your image below or capture directly from camera:")
uploaded_file = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])

# Button to show the camera input widget
show_camera = st.button("Capture Image")

# Display the camera input only if the button is clicked
if show_camera:
    captured_image = st.camera_input("Capture Image")
else:
    captured_image = None

# Check if either an uploaded file or captured image is provided
if uploaded_file or captured_image:
    # Process the uploaded image or captured image
    if uploaded_file:
        image_data = uploaded_file
        image_array = tf.keras.utils.img_to_array(tf.keras.utils.load_img(image_data, target_size=(224, 224)))
    else:
        # Convert captured image data directly to an array
        image_data = captured_image
        image_array = tf.image.decode_image(image_data.read(), channels=3)
        image_array = tf.image.resize(image_array, [224, 224])

    # Display the image
    st.image(image_data, use_column_width=True)
    
    try:
        result = classify_image_array(image_array)
        st.success(result)
    except Exception as e:
        st.error(f"Error in classification: {e}")

    if st.button("Clear Image"):
        uploaded_file = None
        captured_image = None
        st.experimental_rerun()

else:
    st.warning("Please upload an image or capture one using the camera.")

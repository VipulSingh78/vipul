import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
from telegram import Bot, InputFile
from telegram.error import TelegramError

# Telegram bot token and chat ID
bot_token = 'YOUR_TELEGRAM_BOT_TOKEN'
chat_id = 'YOUR_CHAT_ID'

# Initialize the Telegram bot
try:
    bot = Bot(token=bot_token)
    st.write("Telegram Bot initialized successfully!")
except Exception as e:
    st.error(f"Error initializing Telegram Bot: {e}")

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
    model = load_model(model_filename)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Image classification function with confidence threshold
def classify_images(image_path, confidence_threshold=0.5):
    if model is None:
        return "Model is not loaded properly."

    input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

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
    send_telegram_message(image_path, predicted_class, buy_link)
    
    return f'The image belongs to {predicted_class}. [Buy here]({buy_link})'

# Function to send Telegram message with classification and image
def send_telegram_message(image_path, predicted_class, buy_link):
    try:
        # Send classification result and buy link to Telegram
        bot.send_message(
            chat_id=chat_id,
            text=f"Classification Result: {predicted_class}. Buy here: {buy_link}",
            parse_mode='Markdown'
        )
        
        # Send image to Telegram
        with open(image_path, 'rb') as image_file:
            bot.send_photo(chat_id=chat_id, photo=image_file)
        
        st.write("Telegram message sent successfully.")
    except TelegramError as e:
        st.error(f"Error sending Telegram message: {e}")

# Streamlit camera input and file uploader
st.markdown("### Upload your image below or capture directly from camera:")
uploaded_file = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])
captured_image = st.camera_input("Capture Image")

# Choose the captured image or uploaded file if available
image_data = uploaded_file if uploaded_file else captured_image

if image_data is not None:
    # Save and display image
    save_path = os.path.join('upload', uploaded_file.name if uploaded_file else "captured_image.png")
    os.makedirs('upload', exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(image_data.getbuffer() if uploaded_file else captured_image.getvalue())

    st.image(image_data, use_column_width=True)

    try:
        # Run the classify_images function
        result = classify_images(save_path)
        st.success(result)
    except Exception as e:
        st.error(f"Error in classification: {e}")

    if st.button("Clear Image"):
        uploaded_file = None
        st.experimental_rerun()

# Debugging imports
try:
    from telegram import Bot, InputFile
    st.write("telegram module imported successfully!")
except ImportError as e:
    st.error(f"Error importing telegram module: {e}")

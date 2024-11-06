import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
import telegram  # Add the `python-telegram-bot` package for this

# Telegram Bot Token
bot_token = '7608756128:AAEdO8F9kc1W6NDhf6LLXZeZ4USS-rOivok'
chat_id = '<5798688974>'  # Replace with your actual Telegram chat ID
bot = telegram.Bot(token=bot_token)

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

# Image classification function
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
    send_telegram_message(image_path, predicted_class, buy_link)
    
    return f'The image belongs to {predicted_class}. [Buy here]({buy_link})'

# Telegram message function
def send_telegram_message(image_path, predicted_class, buy_link):
    try:
        with open(image_path, 'rb') as img:
            bot.send_photo(
                chat_id=chat_id,
                photo=img,
                caption=f"Classification Result: {predicted_class}. Buy here: {buy_link}"
            )
        print("Telegram message sent successfully.")
    except Exception as e:
        print("Error sending Telegram message:", e)

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

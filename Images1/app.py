import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from twilio.rest import Client
import requests

# Twilio credentials (production me secure environment variables ka use karo)
account_sid = 'AC093d4d6255428d338c2f3edc10328cf7'
auth_token = '40d3d53464a816fb6de7855a640c4194'
client = Client(account_sid, auth_token)

# Streamlit app title
st.title('Welcome to Apna Electrician')
st.subheader('Apna product image upload karo aur recommendations pao!')

# Product ke naam aur links
product_names = ['Anchor Switch', 'CCTV CAMERA', 'FAN', 'Switch', 'TV']
product_links = {
    'Anchor Switch': 'https://www.apnaelectrician.com/anchor-switches',
    'CCTV CAMERA': 'https://www.apnaelectrician.com/cctv-cameras',
    'FAN': 'https://www.apnaelectrician.com/fans',
    'Switch': 'https://www.apnaelectrician.com/switches',
    'TV': 'https://www.apnaelectrician.com/tvs'
}

# Model ka path
model_path = os.path.join('Models', 'Vipul_Recog_Model.h5')

# Agar model exist nahi karta toh download karo
if not os.path.exists(model_path):
    st.info('Model local me nahi mila, download kar rahe hain...')
    try:
        url = 'https://raw.githubusercontent.com/VipulSingh78/vipul/20df1ea393c12e0e1ff97f360e2e281bd594e56c/Images1/Vipul_Recog_Model.h5'
        os.makedirs('Models', exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success('Model successfully download ho gaya!')
    except Exception as e:
        st.error(f"Model download karte waqt error: {e}")

# Model ko load karo, agar error aaye toh usko handle karo
try:
    model = load_model(model_path)
    st.success("Model successfully load ho gaya!")
except Exception as e:
    st.error(f"Model load karte waqt error: {e}")

# Image classify karne ka function
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(result)
    
    if 0 <= predicted_class_index < len(product_names):
        predicted_class = product_names[predicted_class_index]
    else:
        raise IndexError("Predicted class ka index galat range me hai.")
    
    buy_link = product_links.get(predicted_class, 'https://www.apnaelectrician.com/')
    send_whatsapp_message(image_path, predicted_class, buy_link)
    
    return f'Image {predicted_class} se related hai. [Yahan se kharidein]({buy_link})'

# WhatsApp message bhejne ka function
def send_whatsapp_message(image_path, predicted_class, buy_link):
    try:
        media_url = [f'https://your-public-image-url.com/{os.path.basename(image_path)}']
        message = client.messages.create(
            from_='whatsapp:+14155238886',  # Twilio ka number
            body=f"Classification Result: {predicted_class}. Buy here: {buy_link}",
            media_url=media_url,  # Public image URL
            to='whatsapp:+917800905998'
        )
        print("WhatsApp message successfully bheja gaya:", message.sid)
    except Exception as e:
        print("WhatsApp message bhejte waqt error:", e)

# Streamlit file uploader
st.markdown("### Apni image upload karo:")
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
        st.error(f"Image classification me error: {e}")

    if st.button("Image clear karo"):
        uploaded_file = None
        st.experimental_rerun()

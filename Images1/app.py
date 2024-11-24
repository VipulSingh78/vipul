import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from twilio.rest import Client
import requests
import openai  # Import the OpenAI library

# Set up OpenAI API key
openai.api_key = os.getenv('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')  # Ensure this environment variable is set

# Twilio credentials (use secure environment variables in production)
account_sid = os.getenv('TWILIO_ACCOUNT_SID')  # Replace with your Twilio Account SID
auth_token = os.getenv('TWILIO_AUTH_TOKEN')    # Replace with your Twilio Auth Token
client = Client(account_sid, auth_token)

# Streamlit app title
st.title('Welcome to Apna Electrician')
st.subheader('Upload an image of a product, chat with us, and get recommendations!')

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
model_url = 'https://your_model_url.com/Vipul_Recog_Model.h5'  # Update with your model URL
model_filename = os.path.join('Models', 'Vipul_Recog_Model.h5')

os.makedirs('Models', exist_ok=True)

# Function to download model if it doesn't exist
def download_model():
    if not os.path.exists(model_filename):
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(model_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
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

# Image classification function
def classify_images(image_path):
    if model is None:
        return "Model is not loaded properly."

    input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

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
        # Replace with actual hosted URL of the image
        media_url = [f'https://your-public-image-url.com/{os.path.basename(image_path)}']

        message = client.messages.create(
            from_='whatsapp:+14155238886',  # Your Twilio WhatsApp number
            body=f"Classification Result: {predicted_class}. Buy here: {buy_link}",
            media_url=media_url,  # Public image URL
            to='whatsapp:+your_phone_number'  # Replace with your WhatsApp number
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

# Chatbot Interface
st.markdown("### Chat with us:")

if 'messages' not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You:", key="input")

if st.button("Send"):
    if user_input:
        # Append user's message to the session state
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Call OpenAI API for chatbot response
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use 'gpt-4' if available
                messages=st.session_state.messages
            )
            assistant_message = response['choices'][0]['message']['content']
            # Append assistant's response to the session state
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})

            # Display conversation
            for message in st.session_state.messages:
                if message['role'] == 'user':
                    st.write(f"**You:** {message['content']}")
                else:
                    st.write(f"**Assistant:** {message['content']}")
        except Exception as e:
            st.error(f"Error communicating with the chatbot: {e}")

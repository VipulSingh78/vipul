import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from twilio.rest import Client  # Twilio import for WhatsApp message

# Twilio credentials (use environment variables in production for security)
account_sid = 'AC093d4d6255428d338c2f3edc10328cf7'
auth_token = '40d3d53464a816fb6de7855a640c4194'

client = Client(account_sid, auth_token)

# Load your pre-trained model
st.title('Welcome to Apna Electrician')
st.subheader('Upload an image of a product, and get recommendations!')

product_names = ['Anchor Switch', 'CCTV CAMERA', 'FAN', 'Switch', 'TV']
model = load_model('Vipul_Recog_Model.keras')

# Function to classify uploaded images
def classify_images(image_path):
    # Load and preprocess image
    input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    # Predict the class
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_class = product_names[np.argmax(result)]

    # Define product links
    product_links = {
        'Anchor Switch': 'https://www.apnaelectrician.com/anchor-switches',
        'CCTV CAMERA': 'https://www.apnaelectrician.com/cctv-cameras',
        'FAN': 'https://www.apnaelectrician.com/fans',
        'Switch': 'https://www.apnaelectrician.com/switches',
        'TV': 'https://www.apnaelectrician.com/tvs'
    }

    # Generate the dynamic link
    buy_link = product_links.get(predicted_class, 'https://www.apnaelectrician.com/')
    outcome = f'The image belongs to {predicted_class}. [Buy here]({buy_link})'
    
    # Send WhatsApp message with the classification result
    send_whatsapp_message(image_path, predicted_class, buy_link)

    return outcome


# Function to send WhatsApp message
def send_whatsapp_message(image_path, predicted_class, buy_link):
    try:
        # Publicly hosted image URL (replace with your hosted image URL)
        media_url = ['https://your-public-image-url.com/path-to-image.jpg']

        # Create and send the WhatsApp message with the image
        message = client.messages.create(
            from_='whatsapp:+14155238886',  # Twilio sandbox number
            body=f"Classification Result: {predicted_class}. Buy here: {buy_link}",
            media_url=media_url,  # Add the media URL
            to='whatsapp:+917800905998'  # Your verified WhatsApp number
        )
        print("WhatsApp message sent successfully:", message.sid)
    except Exception as e:
        print("Error sending WhatsApp message:", e)


# Streamlit file uploader for image
st.markdown("### Upload your image below:")
uploaded_file = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Save the uploaded file locally
    save_path = os.path.join('upload', uploaded_file.name)
    os.makedirs('upload', exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image with dynamic width
    st.image(uploaded_file, use_column_width=True)

    # Perform classification and display result
    result = classify_images(save_path)
    st.success(result)

    # Option to clear the uploaded image
    if st.button("Clear Image"):
        uploaded_file = None
        st.experimental_rerun()
import streamlit as st

st.title("Hello, Streamlit!")
st.write("This is your first Streamlit app.")

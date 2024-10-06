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

# Load your pre-trained model with error handling
st.title('Welcome to Apna Electrician')
st.subheader('Upload an image of a product, and get recommendations!')

product_names = ['Anchor Switch', 'CCTV CAMERA', 'FAN', 'Switch', 'TV']

# Handle model loading with try-except for better error messages
try:
    model = load_model('Vipul_Recog_Model.keras')
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to classify uploaded images
def classify_images(image_path):
    # Load and preprocess image
    input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    # Predict the class
    predictions = model.predict(input_image_exp_dim)

    # Print the shape and content of predictions for debugging
    print("Predictions shape:", predictions.shape)
    print("Predictions content:", predictions)  # Debugging line

    # Check if predictions are in the expected shape
    if predictions.ndim == 2:  # Ensure that predictions are a 2D array
        result = tf.nn.softmax(predictions[0])
        predicted_class_index = np.argmax(result)

        # Check if the index is within the range of product names
        if 0 <= predicted_class_index < len(product_names):
            predicted_class = product_names[predicted_class_index]
        else:
            raise IndexError("Predicted class index is out of range.")
    else:
        raise ValueError("Unexpected predictions shape: expected 2D array.")

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
        media_url = [f'https://your-public-image-url.com/{os.path.basename(image_path)}']

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

# Ensure only valid image files are uploaded
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
    try:
        result = classify_images(save_path)
        st.success(result)
    except Exception as e:
        st.error(f"Error in classification: {e}")

    # Option to clear the uploaded image
    if st.button("Clear Image"):
        uploaded_file = None
        st.experimental_rerun()

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from twilio.rest import Client
import requests

# Twilio credentials
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

# Load or fine-tune model
model_filename = 'fine_tuned_mobilenet_model.h5'

def fine_tune_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the convolutional base

    # Add classification head for product categories
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(product_names), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Assume the dataset is in folders corresponding to each class
    train_data_dir = 'your_dataset/train'
    validation_data_dir = 'your_dataset/validation'

    # Data augmentation
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load dataset
    train_generator = train_datagen.flow_from_directory(
        train_data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        validation_data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

    # Fine-tune the model
    model.fit(train_generator, epochs=10, validation_data=validation_generator)

    # Save the fine-tuned model
    model.save(model_filename)

    return model

# Check if the model exists, otherwise fine-tune it
if not os.path.exists(model_filename):
    model = fine_tune_model()
else:
    model = load_model(model_filename)

# Classification function with confidence threshold
def classify_images(image_path, confidence_threshold=0.8):
    input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_expanded = np.expand_dims(input_image_array, axis=0) / 255.0  # Normalize

    predictions = model.predict(input_image_expanded)
    result = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(result)
    predicted_confidence = result[predicted_class_index]

    if predicted_confidence < confidence_threshold:
        return "Error: The image doesn't match any known product."

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
        media_url = [f'https://your-public-image-url.com/{os.path.basename(image_path)}']

        message = client.messages.create(
            from_='whatsapp:+14155238886',
            body=f"Classification Result: {predicted_class}. Buy here: {buy_link}",
            media_url=media_url,
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

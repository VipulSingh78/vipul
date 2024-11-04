import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

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
model_filename = os.path.join('Models', 'Vipul_Recog_Model.h5')
os.makedirs('Models', exist_ok=True)

# Load the model
try:
    model = load_model(model_filename)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Image classification function
def classify_images(image_path, confidence_threshold=0.8):
    if model is None:
        return "Model is not loaded properly."

    # Preprocess the image
    input_image = image.load_img(image_path, target_size=(224, 224))
    input_image_array = image.img_to_array(input_image)
    input_image_exp_dim = np.expand_dims(input_image_array, axis=0) / 255.0

    # Predict with model
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(result)
    predicted_confidence = result[predicted_class_index]

    # Check confidence and if it belongs to known classes
    if predicted_confidence < confidence_threshold or predicted_class_index >= len(product_names):
        return "Error: Unknown product."

    predicted_class = product_names[predicted_class_index]
    buy_link = product_links.get(predicted_class, '#')
    
    return f'The image belongs to {predicted_class}. [Buy here]({buy_link})'

# Streamlit input section
st.markdown("### Upload your image below or capture directly from camera:")
uploaded_file = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])
captured_image = st.camera_input("Capture Image")

# Select image source (uploaded or captured)
image_data = uploaded_file if uploaded_file else captured_image

if image_data is not None:
    # Save and display image
    save_path = os.path.join('upload', uploaded_file.name if uploaded_file else "captured_image.png")
    os.makedirs('upload', exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(image_data.getbuffer())

    st.image(image_data, use_column_width=True)

    try:
        # Get classification result
        result = classify_images(save_path)
        if "Error" in result:
            st.error(result)
        else:
            st.success(result)
    except Exception as e:
        st.error(f"Error in classification: {e}")

    if st.button("Clear Image"):
        uploaded_file = None
        st.experimental_rerun()

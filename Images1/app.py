import os
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

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

# Load the model
model_filename = os.path.join('Models', 'Vipul_Recog_Model.h5')
model = load_model(model_filename)

# Image classification function
def classify_images(image):
    input_image = tf.keras.utils.load_img(image, target_size=(224, 224))
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
    
    return f'The image belongs to {predicted_class}. [Buy here]({buy_link})'

# Streamlit file uploader
uploaded_file = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, use_column_width=True)

    # Save uploaded image to a temporary location
    with open(os.path.join('temp', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Classify and display result
    try:
        result = classify_images(uploaded_file)
        st.success(result)
    except Exception as e:
        st.error(f"Error in classification: {e}")

    if st.button("Clear Image"):
        uploaded_file = None
        st.experimental_rerun()

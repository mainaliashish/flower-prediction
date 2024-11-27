import os
import keras
import numpy as np
from keras.models import load_model
import streamlit as st
import tensorflow as tf

st.header('Flower Classification CNN Model')

flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = load_model('/Users/ashish-macbook-pro/Work/python_projects/flower-prediction/Flower_Prediction_Model.h5')

IMAGE_SIZE = 180

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_expand_dim = tf.expand_dims(input_image_array, 0)
    predictions = model.predict(input_image_expand_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = f"The Image belongs to {flower_names[np.argmax(result)]} with a score of {str(np.max(result)*100)}"
    return outcome

uploaded_file = st.file_uploader('Upload an image')
if uploaded_file is not None:
    with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width = 200)

    st.markdown(classify_images(uploaded_file))
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps
import numpy as np


model= tf.keras.models.load_model("potatoes_plant_disease.h5")
class_names= ['Early_blight', 'Late_blight', 'healthy']
image_size= 256

def predict_image(model, img):
    image_arr = image.resize((image_size, image_size))
    image_arr= tf.keras.preprocessing.image.img_to_array(img)
    image_arr= tf.expand_dims(image_arr, 0)
    
    predictions= model.predict(image_arr)
    predicted_class= class_names[np.argmax(predictions[0])]
    confidence= round(100*(np.max(predictions[0])), 2)
    
    return predicted_class, confidence

#Streamlit app
file_upload= st.file_uploader("Upload a potato plant leave to test for a disease.", 
                              type= ["jpg", "png", "jpeg"])
if file_upload is not None:
    image= Image.open(file_upload)
    st.image(image, use_column_width=True)
    predicted_class, confidence = predict_image(model, image)
    st.success(f"Prediction: {predicted_class}.\n Confidence: {confidence}%")

import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import requests
import os


model_url = "https://github.com/Cafffie/potato_disease_app/blob/main/potatoes_plant_model.h5"
local_model_path = "potatoes_plant_model.h5"

if not os.path.exists(local_model_path):
    with open(local_model_path, "wb") as f:
        response = requests.get(model_url)
        f.write(response.content)

model = tf.keras.models.load_model(local_model_path)
        
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
st.write("author: Arebi Olawunmi")
st.write("Date: 24/12/2024")

st.title("Potato Disease Detection App")
st.write("#### This app assists in identifying diseases present in potato plants.")

file_upload= st.file_uploader("Upload an image of a potato plant leave.", 
                              type= ["jpg", "png", "jpeg"])
if file_upload is not None:
    image= Image.open(file_upload)
    st.image(image, use_column_width=True)
    predicted_class, confidence = predict_image(model, image)
    st.success(f"PREDICTION: {predicted_class}.\nCONFIDENCE: {confidence}%")

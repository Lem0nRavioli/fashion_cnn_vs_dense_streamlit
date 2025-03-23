import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('model_cnn.h5')
label_map = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

st.title("🎽 Fashion MNIST - Prédictions")

uploaded_file = st.file_uploader("Upload une image (28x28)", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    label = label_map[np.argmax(prediction)]

    st.image(image, caption=f"Prédiction : {label}", width=150)

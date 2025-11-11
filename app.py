import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# 📦 Cargar el modelo
model = load_model("modelo/modelo_entrenado.h5")

st.title("🚀 Clasificador de imágenes con IA")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(io.BytesIO(uploaded_file.read()), target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    st.image(img, caption="Imagen cargada", use_column_width=True)

    pred = model.predict(img_array)
    st.write("📊 Predicción:", pred)

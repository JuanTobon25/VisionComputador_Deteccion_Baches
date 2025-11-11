import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os

# ---------------------------
# 🔹 CONFIGURACIÓN INICIAL
# ---------------------------
st.set_page_config(page_title="Clasificador de Imágenes con IA", page_icon="🚀", layout="centered")

st.title("🚀 Clasificador de Imágenes con IA")
st.write("Sube una imagen y el modelo la clasificará automáticamente.")

# ---------------------------
# 🔹 CARGA DEL MODELO
# ---------------------------
MODEL_PATH_KERAS = "modelo/modelo_entrenado.keras"
MODEL_PATH_H5 = "modelo/modelo_entrenado.h5"

# Verificar qué formato de modelo existe
if os.path.exists(MODEL_PATH_H5):
    model_path = MODEL_PATH_H5
else:
    st.error("❌ No se encontró el modelo en la carpeta 'modelo/'. Verifica la ruta o el nombre del archivo.")
    st.stop()

try:
    model = load_model(model_path)
    st.success(f"✅ Modelo cargado correctamente desde `{model_path}`")
except Exception as e:
    st.error(f"⚠️ Error al cargar el modelo: {e}")
    st.stop()

# ---------------------------
# 🔹 SUBIDA DE IMAGEN
# ---------------------------
uploaded_file = st.file_uploader("📸 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Leer y procesar la imagen
        img = image.load_img(io.BytesIO(uploaded_file.read()), target_size=(64, 64))
        img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

        st.image(img, caption="🖼️ Imagen cargada", use_column_width=True)

        # ---------------------------
        # 🔹 PREDICCIÓN
        # ---------------------------
        with st.spinner("🔍 Clasificando la imagen..."):
            pred = model.predict(img_array)
        
        # Mostrar el resultado
        pred_label = np.argmax(pred, axis=1)[0] if pred.ndim > 1 else float(pred[0])
        st.subheader("📊 Resultado de la Predicción")
        st.write(f"**Etiqueta predicha:** {pred_label}")
        st.write(f"**Probabilidades:** {pred}")

    except Exception as e:
        st.error(f"⚠️ Error procesando la imagen: {e}")

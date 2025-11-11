import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os

# ============================================================
# ⚙️ CONFIGURACIÓN INICIAL
# ============================================================
st.set_page_config(page_title="Detección de Baches con IA", page_icon="🚧", layout="centered")

st.title("🚗 Detección de Baches con Inteligencia Artificial")
st.write("Sube una imagen de una carretera y el modelo determinará si tiene **baches** o está **en buen estado**.")

# ============================================================
# 💾 CARGA DEL MODELO
# ============================================================
MODEL_PATH = "modelo/modelo_entrenado.h5"

if not os.path.exists(MODEL_PATH):
    st.error("❌ No se encontró el modelo en la carpeta `modelo/`. Verifica la ruta o el nombre del archivo.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    st.success(f"✅ Modelo cargado correctamente desde `{MODEL_PATH}`")
except Exception as e:
    st.error(f"⚠️ Error al cargar el modelo: {e}")
    st.stop()

# ============================================================
# 🏷️ NOMBRES DE CLASES
# ============================================================
# 0 = sin baches, 1 = con baches
class_names = {0: "✅ Sin baches", 1: "🚧 Con baches"}

# ============================================================
# 🖼️ SUBIDA DE IMAGEN
# ============================================================
uploaded_file = st.file_uploader("📸 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Leer imagen sin perder calidad
        img_bytes = uploaded_file.read()
        img = image.load_img(io.BytesIO(img_bytes), target_size=(128, 128))
        img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

        # Mostrar imagen centrada y más pequeña
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="🖼️ Imagen cargada", width=250)

        # ============================================================
        # 🔍 PREDICCIÓN
        # ============================================================
        with st.spinner("🤖 Analizando la imagen..."):
            pred = model.predict(img_array)
            prob = float(pred[0][0])

        # ============================================================
        # 🧠 INTERPRETACIÓN
        # ============================================================
        label = 1 if prob > 0.5 else 0
        class_name = class_names[label]

        # ============================================================
        # 📊 RESULTADOS
        # ============================================================
        st.subheader("📈 Resultado de la Predicción")
        if label == 1:
            st.error(f"{class_name} (probabilidad: {prob:.4f})")
        else:
            st.success(f"{class_name} (probabilidad: {prob:.4f})")

    except Exception as e:
        st.error(f"⚠️ Error procesando la imagen: {e}")


import streamlit as st
import os
import zipfile
import io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# -------------------------------------------------------
# ⚙️ CONFIGURACIÓN DE LA APP
# -------------------------------------------------------
st.set_page_config(page_title="Clasificador de Baches", page_icon="🕳️", layout="centered")

st.title("🕳️ Clasificador de Baches con IA")
st.write("Sube una imagen y el modelo la clasificará automáticamente como **Con bache** o **Sin bache**.")

# -------------------------------------------------------
# ⚙️ CONFIGURACIÓN DE KAGGLE (desde secretos de Streamlit)
# -------------------------------------------------------
try:
    os.makedirs("/root/.kaggle", exist_ok=True)
    with open("/root/.kaggle/kaggle.json", "w") as f:
        json.dump({
            "username": st.secrets["KAGGLE_USERNAME"],
            "key": st.secrets["KAGGLE_KEY"]
        }, f)
    os.chmod("/root/.kaggle/kaggle.json", 600)
    st.info("🔐 Autenticación con Kaggle configurada correctamente.")
except Exception as e:
    st.warning(f"⚠️ No se configuró la autenticación de Kaggle: {e}")

# -------------------------------------------------------
# ⚙️ DESCARGA DIRECTA DEL MODELO DESDE KAGGLE
# -------------------------------------------------------
DATASET_NAME = "juanjostobnvargas/cnn-baches"
MODEL_DIR = "modelo"
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_entrenado.h5")

st.info("📦 Descargando modelo desde Kaggle...")
os.makedirs(MODEL_DIR, exist_ok=True)

# Descargar y descomprimir el modelo desde Kaggle
os.system(f"kaggle datasets download -d {DATASET_NAME} -p {MODEL_DIR}")

# Buscar el zip descargado y extraerlo
for file in os.listdir(MODEL_DIR):
    if file.endswith(".zip"):
        with zipfile.ZipFile(os.path.join(MODEL_DIR, file), "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        os.remove(os.path.join(MODEL_DIR, file))

# -------------------------------------------------------
# ⚙️ CARGA DEL MODELO
# -------------------------------------------------------
try:
    model = load_model(MODEL_PATH)
    st.success("✅ Modelo cargado correctamente.")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# -------------------------------------------------------
# 🧩 SUBIDA DE IMAGEN
# -------------------------------------------------------
uploaded_file = st.file_uploader("📸 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Cargar y procesar la imagen
        img = image.load_img(io.BytesIO(uploaded_file.read()), target_size=(128, 128))
        img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

        # Mostrar imagen centrada
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="🖼️ Imagen cargada", width=220)

        # -------------------------------------------------------
        # 🧠 PREDICCIÓN
        # -------------------------------------------------------
        with st.spinner("🔍 Clasificando..."):
            pred = model.predict(img_array)[0][0]

        # -------------------------------------------------------
        # 📊 RESULTADO
        # -------------------------------------------------------
        st.subheader("📊 Resultado de la Predicción")
        if pred > 0.5:
            st.success(f"🚧 **Con bache** (confianza: {pred:.2f})")
        else:
            st.info(f"🛣️ **Sin bache** (confianza: {1 - pred:.2f})")

    except Exception as e:
        st.error(f"⚠️ Error procesando la imagen: {e}")

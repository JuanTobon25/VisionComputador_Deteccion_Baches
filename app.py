import streamlit as st
import os
import zipfile
import io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import kaggle
import tempfile

# -------------------------------------------------------
# ⚙️ CONFIGURACIÓN DE LA APP
# -------------------------------------------------------
st.set_page_config(page_title="Clasificador de Baches", page_icon="🕳️", layout="centered")

st.title("🕳️ Clasificador de Baches con IA")
st.write("Sube una imagen y el modelo la clasificará automáticamente como **Con bache** o **Sin bache**.")

# -------------------------------------------------------
# ⚙️ AUTENTICACIÓN CON KAGGLE DESDE st.secrets
# -------------------------------------------------------
try:
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
    st.success("🔐 Autenticación con Kaggle configurada correctamente.")
except Exception as e:
    st.error(f"⚠️ No se encontraron las credenciales de Kaggle en `st.secrets`: {e}")
    st.stop()

# -------------------------------------------------------
# ⚙️ DESCARGA DEL MODELO DESDE KAGGLE
# -------------------------------------------------------
DATASET_NAME = "juanjostobnvargas/cnn-baches"

with tempfile.TemporaryDirectory() as tmp_dir:
    st.info("📦 Descargando modelo desde Kaggle...")
    os.system(f"kaggle datasets download -d {DATASET_NAME} -p {tmp_dir}")

    # Buscar el zip descargado y extraerlo
    for file in os.listdir(tmp_dir):
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(tmp_dir, file), "r") as zip_ref:
                zip_ref.extractall(tmp_dir)
            os.remove(os.path.join(tmp_dir, file))

    # Localizar el archivo del modelo
    model_path = None
    for root, _, files in os.walk(tmp_dir):
        for f in files:
            if f.endswith(".h5"):
                model_path = os.path.join(root, f)
                break

    if not model_path:
        st.error("❌ No se encontró el archivo del modelo en el dataset.")
        st.stop()

    # -------------------------------------------------------
    # ⚙️ CARGA DEL MODELO
    # -------------------------------------------------------
    try:
        model = load_model(model_path)
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
        # Cargar y procesar la imagen a (128,128)
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


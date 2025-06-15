import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
#@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_letah.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Fungsi prediksi menggunakan TFLite
def predict_tflite(image, interpreter):
    # Ambil detail input dan output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocessing gambar
    image = tf.image.resize(image, [180, 180])  # Sesuaikan resolusi input model Anda
    image = tf.cast(image, tf.float32) / 255.0  # Normalisasi jika model dilatih dengan skala [0,1]
    image = np.expand_dims(image, axis=0)

    # Set input ke model
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Ambil output prediksi
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Streamlit UI
st.title("Diagnosa Diabetes Mellitus dari Citra Lidah")
st.text("Upload gambar lidah pasien untuk mendapatkan prediksi")

# Upload file gambar
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    

    interpreter = load_tflite_model()
    prediction = predict_tflite(np.array(image), interpreter)

    # Ganti dengan label kelas sesuai dengan model Anda
    class_names = ['Negatif Diabetes', 'Positif Diabetes']
    pred_label = class_names[np.argmax(prediction)]

    st.markdown("### Hasil Prediksi:")
    st.success(f"Hasil analisis menunjukkan bahwa pasien: **{pred_label}**")
    st.image(image, caption="Gambar yang Diunggah", use_column_width=True)
else:
    st.info("Silakan unggah gambar terlebih dahulu.")

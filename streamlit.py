import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu


# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Diabetes", page_icon="🩺", layout="wide")

# Fungsi: Load TFLite model
#@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_letah2.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Fungsi: Prediksi dengan model TFLite
def predict_tflite(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocessing gambar
    image = tf.image.resize(image, [180, 180])
    #image = tf.cast(image, tf.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = tf.nn.softmax(output_data[0]) 
    return output_data

# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        "Menu Utama",
        ["Dashboard", "Pengertian Diabetes", "Deteksi"],
        icons=["house", "info-circle", "search"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#f0f2f6"},
            "icon": {"color": "black", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px"},
            "nav-link-selected": {"background-color": "#02ab21", "color": "white"},
        }
    )

# Konten Berdasarkan Menu yang Dipilih
if selected == "Dashboard":
    st.title("Selamat Datang di Aplikasi Deteksi Diabetes Mellitus")
    st.markdown("""
    Aplikasi ini membantu mendeteksi kemungkinan diabetes mellitus berdasarkan citra lidah pasien 
    menggunakan model deep learning berbasis TensorFlow Lite. Aplikasi ini merupakan hasil penelitian yang berjudul **"PEMANFAATAN ALGORITMA CONVOLUTIONAL NEURAL NETWORK DALAM MEN-DIAGNOSA PENYAKIT DIABETES MELLITUS MENGGUNAKAN CITRA LIDAH"**
    
    ---
    
    ### Abstrak Penelitian
    Diabetes melitus merupakan penyakit metabolik kronis yang ditandai oleh peningkatan kadar glukosa darah sebagai akibat dari gangguan sekresi maupun resistensi insulin. 
    Di Indonesia, jumlah penderita diabetes terus meningkat secara signifikan setiap tahunnya, menjadikannya salah satu masalah kesehatan utama yang perlu segera diatasi. 
    Salah satu gejala fisik yang dapat diamati dari penderita diabetes melitus adalah perubahan kondisi lidah, seperti munculnya lapisan putih kekuningan atau retakan. 
    Penelitian ini bertujuan untuk mengembangkan sistem diagnosis dini penyakit diabetes melitus melalui analisis citra lidah menggunakan algoritma Convolutional Neural Network (CNN), 
    dengan harapan dapat memberikan solusi diagnosis yang cepat, akurat, dan dapat diakses dengan mudah oleh masyarakat. Model CNN dalam penelitian ini berhasil mencapai akurasi sebesar 0.79
    dalam mengklasifikasikan citra lidah penderita diabetes dan non-diabetes. Selain itu, model ini telah diimplementasikan ke dalam platform aplikasi berbasis web, sehingga memungkinkan pengguna umum 
    untuk melakukan diagnosis secara mandiri melalui citra lidah. Hasil dari penelitian ini diharapkan dapat berkontribusi dalam upaya deteksi dini diabetes melitus serta menjadi dasar pengembangan sistem 
    serupa di masa depan yang lebih optimal dan akurat.""")

elif selected == "Pengertian Diabetes":
    st.title("Pengertian Penyakit Diabetes")
    st.markdown("""
### 🩺 Apa Itu Diabetes Mellitus?

**Diabetes Mellitus (DM)** adalah penyakit gangguan metabolik kronis yang ditandai oleh kadar gula darah tinggi akibat gangguan produksi atau fungsi insulin. Penyakit ini merupakan penyebab kematian keenam di dunia.

---

### 🎯 Faktor Risiko Diabetes
- **Tidak dapat dimodifikasi**: Usia, jenis kelamin, dan faktor genetik.
- **Dapat dimodifikasi**: Pola makan, aktivitas fisik, dan gaya hidup.

---

### ⚠️ Gejala Diabetes
- **Akut**: Haus berlebihan, lapar terus-menerus, sering buang air kecil, berat badan turun drastis, mudah lelah.
- **Kronik**: Kesemutan, penglihatan kabur, masalah gigi, gangguan seksual, kelelahan, bahkan keguguran.

---

### 👄 Tanda-Tanda di Rongga Mulut Penderita Diabetes
1. **Mulut Kering (Xerostomia)** – Air liur berkurang, meningkatkan risiko infeksi mulut.
2. **Radang Gusi (Periodontitis)** – Gusi berdarah dan gigi goyah.
3. **Sariawan (Stomatitis Apthosa)** – Infeksi jamur karena kadar gula tinggi.
4. **Oral Thrush** – Lapisan putih kekuningan pada lidah & kerongkongan.
5. **Gigi Berlubang (Dental Caries)** – Karena mulut asam & air liur berkurang.

---

> Kesehatan mulut dapat mencerminkan kondisi sistemik tubuh, termasuk diabetes. Aplikasi ini membantu mendeteksi kemungkinan diabetes melalui analisis citra lidah.
""")


elif selected == "Deteksi":
    st.title("Deteksi Diabetes dari Citra Lidah")
    st.markdown("Upload gambar lidah pasien untuk mengetahui kemungkinan diagnosa diabetes.")
    info = "Upload gambar lidah yang jelas dan fokus dengan pengambilan gambar sekitar 10-15cm dari lidah agar hasil prediksi akurat."
    st.write(f"**{info}**")

    uploaded_file = st.file_uploader("📤 Unggah gambar lidah pasien", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        #st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

        with st.spinner("🔍 Menganalisis gambar..."):
            interpreter = load_tflite_model()
            prediction = predict_tflite(np.array(image), interpreter)

            # Label hasil prediksi
            class_names = ['Positif Diabetes', 'Gambar Lidah Tidak Terdeteksi', 'Negatif Diabetes']
            pred_index = np.argmax(prediction)
            pred_label = class_names[pred_index]
            #confidence = prediction[0][pred_index] * 100
            confidence = np.max(prediction) * 100
        
        if pred_index == 1:
            st.error("Gambar lidah tidak terdeteksi. Mohon mengikuti panduan pengambilan gambar yang benar.")
        else:
            st.markdown("### 🧪 Hasil Prediksi:")
            st.success(f"Pasien terdeteksi: **{pred_label}** dengan tingkat keyakinan {confidence:.2f}%")
            #st.write("Prediksi Mentah:", prediction)
            #st.write("Prediksi Index:", pred_index)
            #st.write("Label:", pred_label)
        st.image(image, caption="Gambar yang Diunggah", use_column_width=True)
        #print(prediction)
        #print(np.sum(prediction))
        #print(np.argmax(prediction))

    else:
        st.info("Silakan unggah gambar terlebih dahulu untuk memulai deteksi.")

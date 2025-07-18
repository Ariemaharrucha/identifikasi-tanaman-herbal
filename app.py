import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Identifikasi Tanaman Herbal 🌿",
    page_icon="🌿",
    layout="wide"
)

# --- CSS Kustom untuk Desain ---
st.markdown("""
<style>
/* Mengubah font utama */
html, body, [class*="st-"] {
    font-family: 'Georgia', serif;
}
/* Style untuk judul utama */
h1 {
    color: #2E8B57; /* SeaGreen */
    text-align: center;
}
/* Style untuk tombol utama 'Cek Gambar' */
.stButton>button {
    color: #FFFFFF;
    background-color: #2E8B57; /* SeaGreen */
    border-radius: 20px;
    border: 1px solid #2E8B57;
    padding: 10px 24px;
    font-weight: bold;
    width: 100%;
}


/* Ubah gaya tombol 'Browse files' */
[data-testid="stFileUploader"] button {
    background-color: #8A2BE2; /* BlueViolet */
    color: white;
    border: none;
    border-radius: 20px;
    padding: 12px 30px;
    font-size: 1.1em;
    font-weight: bold;
}

/* --- STYLE UNTUK TEKS NAMA FILE (PERBAIKAN DENGAN SELECTOR STABIL) --- */
/* Menargetkan div yang berisi tombol hapus file */
div:has(button[data-testid="stFileUploaderDeleteBtn"]) {
    color: #4B0082; /* Ungu gelap (Indigo) */
    font-weight: bold;
    font-size: 1.05rem;
    background-color: rgba(147, 112, 219, 0.1);
    padding: 8px 12px; /* Sedikit menambah padding */
    border-radius: 8px;
    border: 1px solid #9370DB;
    display: flex; /* Menggunakan flex untuk mengatur item di dalamnya */
    align-items: center;
    justify-content: space-between;
}
/* Mengubah gaya tombol X di dalamnya agar serasi */
button[data-testid="stFileUploaderDeleteBtn"] {
    background-color: #9370DB !important;
    border: none;
}

</style>
""", unsafe_allow_html=True)


# --- Informasi Tanaman Herbal ---
HERBAL_INFO = {
    'Daun Jambu Biji': {
        'description': 'Daun dari pohon jambu biji (Psidium guajava) yang dikenal luas karena khasiat obatnya, terutama untuk mengatasi masalah pencernaan.',
        'benefits': ['Membantu mengatasi diare', 'Menurunkan kolesterol', 'Baik untuk penderita diabetes', 'Kaya akan antioksidan']
    },
    'Daun Kari': {
        'description': 'Daun dari pohon kari (Murraya koenigii) yang sering digunakan sebagai bumbu masak dan memiliki aroma yang khas serta kuat.',
        'benefits': ['Baik untuk kesehatan rambut', 'Membantu menurunkan berat badan', 'Mengontrol kadar gula darah', 'Meningkatkan kesehatan mata']
    },
    'Daun Kemangi': {
        'description': 'Dikenal juga sebagai basil, daun ini memiliki aroma wangi yang khas dan sering dijadikan lalapan atau bumbu masakan.',
        'benefits': ['Sebagai antiseptik alami', 'Menjaga kesehatan jantung', 'Mengurangi stres oksidatif', 'Meningkatkan sistem kekebalan tubuh']
    },
    'Daun Mint': {
        'description': 'Daun dari genus Mentha ini memberikan sensasi dingin dan segar, sering digunakan dalam minuman, makanan, dan produk kesehatan.',
        'benefits': ['Meredakan gangguan pencernaan', 'Menyegarkan napas', 'Membantu meredakan sakit kepala', 'Meningkatkan fungsi otak']
    },
    'Daun Pepaya': {
        'description': 'Daun dari pohon pepaya (Carica papaya) yang memiliki rasa pahit namun kaya akan enzim dan nutrisi penting.',
        'benefits': ['Meningkatkan trombosit (membantu penderita DBD)', 'Sebagai anti-malaria', 'Melancarkan pencernaan', 'Memiliki sifat anti-kanker']
    },
    'Daun Sirih': {
        'description': 'Tanaman merambat yang daunnya memiliki nilai budaya dan kesehatan tinggi di Asia Tenggara, terkenal sebagai antiseptik.',
        'benefits': ['Sebagai antiseptik alami', 'Menjaga kesehatan mulut dan gigi', 'Mengatasi mimisan', 'Membantu menyembuhkan luka']
    },
    'Daun Sirsak': {
        'description': 'Daun dari pohon sirsak (Annona muricata) yang dipercaya memiliki banyak khasiat untuk pengobatan, termasuk kanker.',
        'benefits': ['Berpotensi sebagai anti-kanker', 'Membantu menurunkan asam urat', 'Mengatasi rematik', 'Meningkatkan kualitas tidur']
    },
    'Lidah Buaya': {
        'description': 'Dikenal sebagai Aloe vera, tanaman sukulen ini memiliki gel di dalam daunnya yang kaya manfaat untuk kulit dan kesehatan.',
        'benefits': ['Melembapkan dan menyehatkan kulit', 'Mempercepat penyembuhan luka bakar', 'Membantu mengatasi sembelit', 'Menurunkan kadar gula darah']
    },
    'Teh Hijau': {
        'description': 'Daun dari tanaman Camellia sinensis yang diproses minimal, kaya akan antioksidan, dan menjadi salah satu minuman tersehat di dunia.',
        'benefits': ['Kaya akan antioksidan (polifenol)', 'Meningkatkan metabolisme tubuh', 'Meningkatkan fungsi otak', 'Mengurangi risiko penyakit jantung']
    }
}
CLASS_NAMES = list(HERBAL_INFO.keys())


# --- Fungsi Model & Prediksi ---
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# --- Memuat Model ---
@st.cache_resource
def load_keras_model():
    try:
        model = load_model('model_klasifikasi_daun.h5')
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}", icon="🚨")
        return None

model = load_keras_model()

# --- Antarmuka Aplikasi ---
st.title("🌿 Identifikasi Tanaman Herbal")
st.markdown("Unggah gambar daun untuk mengetahui jenis dan manfaatnya!")

# Area Upload
st.header("1. Unggah Gambar Anda", divider='rainbow')
uploaded_file = st.file_uploader(
    "Label ini sekarang disembunyikan oleh CSS",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if model is None:
    st.error("Model tidak dapat dimuat. Aplikasi tidak dapat berfungsi. Pastikan file 'model_klasifikasi_daun.h5' ada di direktori yang sama.")
elif uploaded_file is None:
    st.info("Silakan unggah gambar dengan menekan tombol di atas.", icon="👆")
else:
    # Tata letak 2 kolom untuk gambar dan hasil
    col1, col2 = st.columns(2)

    with col1:
        st.header("2. Pratinjau Gambar", divider='rainbow')
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah.", use_column_width=True)

    with col2:
        st.header("3. Mulai Identifikasi", divider='rainbow')
        if st.button("✨ Cek Gambar Ini!"):
            with st.spinner('Menganalisis gambar... Mohon tunggu...'):
                prediction = predict(image, model)
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                confidence = prediction[0][predicted_class_index] * 100

                info = HERBAL_INFO.get(predicted_class_name, {})
                description = info.get('description', 'Informasi tidak tersedia.')
                benefits = info.get('benefits', [])

                st.success(f"**Hasil Ditemukan!**")
                st.markdown("---")
                st.subheader(f"Jenis Daun: {predicted_class_name}")
                st.write("Tingkat Keyakinan:")
                st.progress(int(confidence), text=f"{confidence:.2f}%")
                st.markdown("---")

                with st.expander("Lihat Deskripsi dan Manfaat"):
                    st.markdown(f"**📝 Deskripsi:**\n> {description}")
                    st.markdown("**🌟 Manfaat Utama:**")
                    for benefit in benefits:
                        st.markdown(f"- {benefit}")

# --- Informasi Tambahan ---
st.markdown("---")
st.header("Tentang Aplikasi")
st.warning(
    "Akurasi model mungkin tidak 100%.",
    icon="⚠️"
)
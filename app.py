import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
import numpy as np

# Custom Sequential class for compatibility (if needed)
class CustomSequential(Sequential):
    pass

# Load model architecture and weights
def load_model():
    with open("model_corn_disease_1.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects={"Sequential": CustomSequential})
    model.load_weights("model_corn_disease_weight.h5")
    return model

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalisasi ke [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch
    return image_array

# Predict disease
def predict_disease(model, image):
    predictions = model.predict(image)
    classes = ['BLIGHT', 'COMMON RUST', 'GRAY LEAF SPOT', 'HEALTHY']
    solutions = {
        'BLIGHT': [
            "Gunakan fungisida berbasis tembaga atau mankozeb.",
            "Pastikan sanitasi ladang dilakukan secara rutin untuk mencegah penyebaran spora.",
            "Hindari irigasi di atas daun untuk mengurangi kelembapan.",
            "Rotasi tanaman dengan tanaman yang tidak rentan terhadap *Blight* seperti kacang tanah atau kedelai."
        ],
        'COMMON RUST': [
            "Gunakan fungisida berbasis sulfur atau azoksistrobin.",
            "Pilih varietas jagung yang tahan terhadap *Rust* untuk penanaman berikutnya.",
            "Kurangi kelembapan di ladang dengan meningkatkan jarak antar tanaman.",
            "Buang dan musnahkan daun yang terinfeksi untuk mencegah penyebaran lebih lanjut."
        ],
        'GRAY LEAF SPOT': [
            "Gunakan fungisida sistemik seperti propikonazol atau tebukonazol.",
            "Terapkan rotasi tanaman dengan tanaman bukan inang seperti gandum atau kedelai.",
            "Pastikan sirkulasi udara di ladang cukup untuk mengurangi kelembapan.",
            "Bersihkan sisa-sisa tanaman setelah panen untuk meminimalkan sumber infeksi di musim berikutnya."
        ],
        'HEALTHY': [
            "Daun jagung sehat. Tidak diperlukan tindakan lebih lanjut.",
            "Pastikan pengelolaan nutrisi tanah dilakukan dengan baik.",
            "Perhatikan irigasi yang tidak berlebihan.",
            "Awasi daun secara rutin untuk mendeteksi tanda-tanda awal penyakit.",
            "Gunakan varietas jagung yang tahan penyakit untuk memastikan hasil panen maksimal."
        ]
    }
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    solution = solutions[predicted_class]
    return predicted_class, confidence, solution

# Streamlit app
st.set_page_config(page_title="Deteksi Penyakit Jagung", page_icon="🌽", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🌽 Deteksi Penyakit Jagung")
    st.write("""
    Aplikasi ini dirancang untuk mendeteksi penyakit pada tanaman jagung berdasarkan gambar daun.
    """)

# Main content
st.markdown(
    """
    <style>
    .main-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        color: #2c3e50;
        text-align: center;
        font-weight: bold;
        font-size: 36px;
    }
    .subtitle {
        color: #34495e;
        text-align: center;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .result-title {
        color: #27ae60;
        font-size: 24px;
        font-weight: bold;
    }
    .solution-title {
        color: #2980b9;
        font-size: 20px;
        font-weight: bold;
    }
    .solution-item {
        color: #fff;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Deteksi Penyakit pada Tanaman Jagung</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Unggah gambar daun jagung untuk mendeteksi status kesehatannya</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Gambar yang diunggah", width=400)  # Atur ukuran gambar

    st.write("⏳ **Sedang memproses...**")
    model = load_model()
    processed_image = preprocess_image(image)
    prediction, confidence, solution = predict_disease(model, processed_image)

    st.markdown(f'<div class="result-title">Prediksi: {prediction}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-title">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
    st.markdown('<div class="solution-title">💡 Solusi yang Disarankan:</div>', unsafe_allow_html=True)
    for idx, step in enumerate(solution, 1):
        st.markdown(f'<div class="solution-item">**{idx}.** {step}</div>', unsafe_allow_html=True)

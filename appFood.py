import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from huggingface_hub import from_pretrained_keras

# Custom CSS Styling
def custom_css():
    st.markdown("""
        <style>
        /* Background utama */
        .stApp {
            background-color: #fafafa;
            font-family: "Segoe UI", sans-serif;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2c3e50, #34495e);
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }

        /* Sidebar Navigation - Hilangkan radio button */
        div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }
        div[role="radiogroup"] label {
            background: transparent !important;
            border: none !important;
            padding: 5px 15px;
            border-radius: 8px;
            margin: 5px 0;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
        }
        div[role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateX(4px);
        }
        div[role="radiogroup"] label p {
            font-size: 16px !important;
            font-weight: 600 !important;
            color: #001f3f !important;
        }
        /* Aktif menu */
        div[role="radiogroup"] label[data-baseweb="radio"] > div:last-child {
            background-color: #ffffff;
            padding: 10px 15px;
            border-radius: 8px;
        }

        /* Judul Utama */
        h1 {
            color: #e74c3c;
            text-align: center;
            font-weight: bold;
        }

        /* Subheader */
        h2, h3 {
            color: #34495e;
        }

        /* Button Style */
        div.stButton > button {
            background: linear-gradient(90deg, #3498db, #2ecc71);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.8em 1.4em;
            font-size: 15px;
            font-weight: 600;
            box-shadow: 0 3px 8px rgba(0,0,0,0.2);
            transition: all 0.3s ease-in-out;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #2980b9, #27ae60);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            transform: translateY(-3px);
        }

        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #e67e22;
        }
        </style>
    """, unsafe_allow_html=True)

custom_css()


# Load Model
@st.cache_resource
def load_food_model():
    model = from_pretrained_keras(
        "aisyahnoviani16/food101",  # ganti repo kamu
        filename="inception_food101 (1).keras"
    )
    return model


# Food101 Labels
food_labels = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari",
    "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
    "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
    "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream",
    "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese",
    "macarons", "miso_soup", "mussels", "nachos", "omelette",
    "onion_rings", "oysters", "pad_thai", "paella", "pancakes",
    "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop",
    "poutine", "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli",
    "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops",
    "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara",
    "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos",
    "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]

# Sidebar Navigation
st.sidebar.title("📚 Menu")
page = st.sidebar.radio("📚 Menu", ["🍴 Food Classification", "📖 Tutorial", "ℹ️ Tentang App"], label_visibility="collapsed")

# Halaman Food Classification
if page == "🍴 Food Classification":
    st.title("🍴 Food Classification App")
    st.write("Unggah foto makanan favoritmu, dan biarkan model menebak menunya! 🍔🍣🍕")

    uploaded_file = st.file_uploader("Pilih gambar makanan...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="📷 Gambar yang diupload", use_container_width=True)

        with col2:
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            top3_idx = np.argsort(preds[0])[-3:][::-1]

            st.subheader("Prediksi")
            for i, idx in enumerate(top3_idx):
                label = food_labels[idx] if idx < len(food_labels) else f"Class {idx}"
                confidence = float(preds[0][idx])
                st.write(f"**{i+1}. {label}** ({confidence:.2%})")
                st.progress(confidence)

            best_idx = top3_idx[0]
            best_label = food_labels[best_idx]
            st.success(f"✅ Prediksi utama: **{best_label}**")

# Halaman Tutorial
elif page == "📖 Tutorial":
    st.title("📖 Tutorial Penggunaan")
    st.markdown("""
    1. Klik tombol **Pilih gambar makanan**.  
    2. Upload file dalam format `.jpg`, `.jpeg`, atau `.png`.  
    3. Tunggu beberapa detik, lalu model akan memberikan **Top-3 prediksi**.  
    4. Lihat confidence bar untuk tingkat keyakinan model.  
    5. Hasil utama ditandai dengan ✅.  

    Tips: gunakan gambar makanan yang jelas & terang agar prediksi lebih akurat!
    """)

# Halaman Tentang App
elif page == "ℹ️ Tentang App":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat untuk **klasifikasi gambar makanan** menggunakan model **InceptionV3** yang dilatih pada dataset **Food-101**.  

    **Fitur Utama:**
    - Upload gambar makanan → prediksi otomatis 🍔🍕🍣  
    - Top-3 hasil prediksi dengan confidence bar 📊  
    - Desain interaktif, sidebar gelap, dan tombol modern  

    Dibuat dengan CINTA❤️ menggunakan **Streamlit + TensorFlow/Keras**.
    """)


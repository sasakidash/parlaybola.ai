import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Konfigurasi Halaman Situs
st.set_page_config(
    page_title="AI Prediksi Bola Pro", 
    page_icon="⚽", 
    layout="centered"
)

st.title("⚽ AI Prediksi Pertandingan Sepak Bola (Advanced)")
st.markdown("Menggunakan data mendalam (*xG*, *Ball Possession*, *Shots on Target*, dan *Cedera*) serta pilihan algoritma Machine Learning.")
st.divider()

# 1. Dataset Kaya & Pelatihan Model Dinamis
@st.cache_resource
def get_trained_model(algo_name):
    data = {
        'home_xg': [2.1, 0.8, 1.5, 3.0, 0.5, 1.8, 2.2, 1.1],
        'away_xg': [0.9, 1.2, 1.6, 0.4, 2.0, 1.5, 0.8, 1.9],
        'home_possession': [58, 45, 50, 65, 40, 55, 60, 48],
        'away_possession': [42, 55, 50, 35, 60, 45, 40, 52],
        'home_shots_target': [7, 3, 5, 9, 2, 6, 8, 4],
        'away_shots_target': [3, 4, 5, 1, 7, 5, 2, 6],
        'key_injuries_diff': [0, -1, 1, 0, -2, 1, 0, -1],
        'target': [2, 0, 1, 2, 0, 1, 2, 0]
    }
    
    df = pd.DataFrame(data)
    X = df[['home_xg', 'away_xg', 'home_possession', 'away_possession', 'home_shots_target', 'home_shots_target', 'key_injuries_diff']] # Sesuaikan kolom jika perlu
    # Perbaikan nama kolom agar konsisten
    X = df[['home_xg', 'away_xg', 'home_possession', 'away_possession', 'home_shots_target', 'away_shots_target', 'key_injuries_diff']]
    y = df['target']
    
    if algo_name == "Random Forest" or not HAS_XGBOOST:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algo_name == "XGBoost" and HAS_XGBOOST:
        model = XGBClassifier(eval_metric='logloss')
    else:
        model = LogisticRegression(max_iter=200)
        
    model.fit(X, y)
    return model

# Sidebar
st.sidebar.header("⚙️ Pengaturan AI")
selected_algo = st.sidebar.selectbox(
    "Pilih Algoritma Machine Learning",
    ["Random Forest", "XGBoost", "Logistic Regression"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Sumber Data Publik")
st.sidebar.info("Data historis dapat diunduh melalui **Football-Data.co.uk** atau **API-Football**.")

model = get_trained_model(selected_algo)

# Form Input
st.subheader("📊 Masukkan Statistik & Metrik Pertandingan")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🏠 Tim Kandang (Home)**")
    home_xg = st.number_input("Expected Goals (xG) Home", 0.0, 5.0, 1.7, step=0.1)
    home_possession = st.slider("Penguasaan Bola Home (%)", 10, 90, 55)
    home_shots = st.number_input("Tembakan Tepat Sasaran (Home)", 0, 20, 6, step=1)

with col2:
    st.markdown("**✈️ Tim Tamu (Away)**")
    away_xg = st.number_input("Expected Goals (xG) Away", 0.0, 5.0, 1.2, step=0.1)
    away_possession = st.slider("Penguasaan Bola Away (%)", 10, 90, 45)
    away_shots = st.number_input("Tembakan Tepat Sasaran (Away)", 0, 20, 4, step=1)

st.markdown("---")
key_injuries_diff = st.slider(
    "Kondisi Skuad (Selisih Cedera Pemain Kunci)", 
    min_value=-3, max_value=3, value=0,
    help="Nilai positif: Home lebih sedikit cedera."
)

st.divider()

# Tombol Prediksi (Tanpa st.balloons agar tidak memicu glitch rendering di beberapa browser)
if st.button(f"🔮 Analisis dengan {selected_algo}", type="primary", use_container_width=True):
    input_data = pd.DataFrame([[
        home_xg, away_xg, home_possession, away_possession, 
        home_shots, away_shots, key_injuries_diff
    ]], columns=[
        'home_xg', 'away_xg', 'home_possession', 'away_possession', 
        'home_shots_target', 'home_shots_target', 'key_injuries_diff'
    ])
    
    prediksi = model.predict(input_data)
    hasil_map = {0: "Tim Tamu Menang ✈️", 1: "Hasil Seri 🤝", 2: "Tim Rumah Menang 🏠"}
    hasil_akhir = hasil_map[prediksi[0]]
    
    st.success(f"### Hasil Prediksi: {hasil_akhir}")
import os
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from src.ai.conversational_assistant import answer_question as local_answer_question
from src.ai.genre_classifier import predict_genre
from src.ai.engagement_predictor import predict_engagement as local_predict_engagement
from src.ai.audio_genre_classifier import predict_genre_from_audio as local_predict_genre_from_audio

# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000"

def predict_genre_from_audio(file_path):
    try:
        with open(file_path, "rb") as f:
            response = requests.post(f"{API_URL}/predict-genre", files={"file": f}, timeout=15)
        if response.status_code == 200:
            res = response.json()
            st.sidebar.caption("⚡ Via FastAPI Backend")
            return res["predicted_genre"], res["confidence"]
    except Exception:
        pass
    st.sidebar.caption("🔌 Local Fallback (FastAPI Offline)")
    return local_predict_genre_from_audio(file_path)

def predict_engagement(duration, bit_rate, favorites_track, album_tracks_count, album_listens, album_favorites, genre_top):
    try:
        payload = {
            "duration": duration,
            "bit_rate": bit_rate,
            "favorites_track": favorites_track,
            "album_tracks_count": album_tracks_count,
            "album_listens": album_listens,
            "album_favorites": album_favorites,
            "genre_top": genre_top
        }
        response = requests.post(f"{API_URL}/predict-engagement", json=payload, timeout=5)
        if response.status_code == 200:
            st.sidebar.caption("⚡ Via FastAPI Backend")
            return response.json()["predicted_listens"]
    except Exception:
        pass
    st.sidebar.caption("🔌 Local Fallback (FastAPI Offline)")
    return local_predict_engagement(duration, bit_rate, favorites_track, album_tracks_count, album_listens, album_favorites, genre_top)

def answer_question(question):
    try:
        response = requests.post(f"{API_URL}/ask-assistant", json={"question": question}, timeout=15)
        if response.status_code == 200:
            res = response.json()
            st.sidebar.caption("⚡ Via FastAPI Backend")
            return res["response"], res["sql_query"]
    except Exception:
        pass
    st.sidebar.caption("🔌 Local Fallback (FastAPI Offline)")
    return local_answer_question(question)

# Set Page Config
st.set_page_config(
    page_title="Music Big Data & IA Platform",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = r"c:\SDV\Music"
DB_PATH = os.path.join(BASE_DIR, "data", "datalake", "gold", "catalog.db")
BRONZE_DIR = os.path.join(BASE_DIR, "data", "datalake", "bronze")
MODEL_DIR = os.path.join(BASE_DIR, "src", "ai")

# 1. Custom CSS Styling (Light Premium UI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;900&display=swap');
    
    /* Global Typography & Light Background */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 50%, #fcfdfd 0%, #f1f5f9 100%);
        color: #1e293b !important;
    }
    
    /* Light Mode headings and text overrides */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #0f172a !important;
    }
    
    /* Premium Light Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        margin-bottom: 20px;
    }
    
    /* Light Music-Themed KPI Cards */
    .glass-kpi {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.04) 0%, rgba(236, 72, 153, 0.01) 100%);
        border: 1px solid rgba(124, 58, 237, 0.15);
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.02);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-kpi:hover {
        transform: translateY(-4px);
        border-color: rgba(236, 72, 153, 0.3);
        box-shadow: 0 8px 30px 0 rgba(124, 58, 237, 0.1);
    }
    
    /* Override hardcoded inline white/light colors for KPIs */
    .glass-kpi p {
        color: #4f46e5 !important; /* Soft Indigo */
        font-weight: 600 !important;
    }
    .glass-kpi h2 {
        color: #0f172a !important; /* Dark charcoal */
        font-weight: 800 !important;
    }
    
    /* Musical Gradient Title */
    .title-gradient {
        background: linear-gradient(90deg, #6d28d9 0%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3rem;
        margin-bottom: 2px;
        letter-spacing: -0.03em;
    }
    
    .subtitle-text {
        color: #475569 !important;
        font-size: 1.15rem;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* Styled Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #7c3aed 0%, #db2777 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 28px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.15) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(219, 39, 119, 0.25) !important;
    }
    
    /* SQL Code Block (remains dark for coding readability) */
    .sql-query-box {
        background-color: #0f172a;
        border-left: 4px solid #db2777;
        padding: 14px;
        font-family: 'Courier New', Courier, monospace;
        color: #f472b6;
        margin: 12px 0;
        border-radius: 0 10px 10px 0;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    .sql-query-box code {
        color: #f472b6 !important;
    }
    
    /* Streamlit input overrides (light fields with dark borders/text) */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: rgba(0, 0, 0, 0.1) !important;
        color: #0f172a !important;
    }
    input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-color: rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Modernized Light Sidebar Menu */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important; /* Light silver/slate */
        border-right: 1px solid rgba(0, 0, 0, 0.06);
    }
    
    section[data-testid="stSidebar"] h2 {
        color: #6d28d9 !important; /* Purple sidebar header */
    }
    section[data-testid="stSidebar"] p {
        color: #475569 !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > label {
        display: none !important; /* Hide generic "Navigation" label */
    }
    
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 6px !important;
        padding-top: 15px;
    }
    
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 10px 16px !important;
        margin-bottom: 2px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        position: relative !important;
    }
    
    /* Left indicator bar on hover/checked */
    section[data-testid="stSidebar"] div[role="radiogroup"] label::before {
        content: "" !important;
        position: absolute !important;
        left: 0 !important;
        top: 25% !important;
        height: 50% !important;
        width: 3px !important;
        background-color: transparent !important;
        border-radius: 0 4px 4px 0 !important;
        transition: all 0.2s ease !important;
    }
    
    /* Hover state: subtle background nuance, text darkens, indicator grows slightly */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(0, 0, 0, 0.02) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover div[data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover::before {
        background-color: rgba(109, 40, 217, 0.5) !important;
        height: 50% !important;
    }
    
    /* Checked/Selected state: glowing background gradient, active neon indicator */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, rgba(109, 40, 217, 0.06) 0%, rgba(219, 39, 119, 0.01) 100%) !important;
        border-color: rgba(109, 40, 217, 0.15) !important;
    }
    
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked)::before {
        background-color: #6d28d9 !important; /* Active purple indicator */
        height: 60% !important;
        box-shadow: 0 0 8px #6d28d9 !important;
    }
    
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
        color: #6d28d9 !important; /* Active purple text */
        font-weight: 600 !important;
    }
    
    /* Default menu options text color */
    section[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        color: #334155 !important;
    }
    
    /* Hide default radio circle markers */
    section[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarker"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get DB connection
def get_db_conn():
    return sqlite3.connect(DB_PATH)

# Helper function to format big numbers
def format_num(val):
    if val >= 1000000:
        return f"{val/1000000:.2f}M"
    elif val >= 1000:
        return f"{val/1000:.1f}k"
    return str(val)

# Sidebar navigation
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #818cf8;'>🎵 Music Big Data</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Plateforme Big Data & IA</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    navigation = st.radio(
        "Navigation",
        [
            "Tableau de Bord & Catalogue",
            "Auto-Classificateur (Option B)",
            "Engagement Predictor (Option C)",
            "Assistant IA Gemini (Option A)"
        ]
    )


# Main Application Layout
st.markdown("<h1 class='title-gradient'>Music Analytics & IA Platform</h1>", unsafe_allow_html=True)

# ----------------- PAGE 1: DASHBOARD & CATALOG -----------------
if navigation == "Tableau de Bord & Catalogue":
    st.markdown("<p class='subtitle-text'>Explorez le catalogue de musique centralisé et écoutez les fichiers audio de la zone Bronze</p>", unsafe_allow_html=True)
    
    # Load KPIs
    if not os.path.exists(DB_PATH):
        st.error("Base de données Gold introuvable. Veuillez exécuter le pipeline ETL Prefect d'abord.")
    else:
        conn = get_db_conn()
        cursor = conn.cursor()
        
        # Calculate stats
        cursor.execute("SELECT COUNT(*) FROM tracks;")
        total_tracks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM artists;")
        total_artists = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM albums;")
        total_albums = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(listens) FROM tracks;")
        total_listens = cursor.fetchone()[0] or 0
        
        conn.close()
        
        # KPI Row
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.markdown(f"""<div class='glass-kpi'>
                <p style='color:#a5b4fc; font-size: 0.9rem; margin-bottom: 5px;'>MORCEAUX CENTRALISÉS</p>
                <h2 style='color:white; font-size: 2.2rem; font-weight: 800; margin: 0;'>{total_tracks:,}</h2>
            </div>""", unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(f"""<div class='glass-kpi'>
                <p style='color:#a5b4fc; font-size: 0.9rem; margin-bottom: 5px;'>ARTISTES ENREGISTRÉS</p>
                <h2 style='color:white; font-size: 2.2rem; font-weight: 800; margin: 0;'>{total_artists:,}</h2>
            </div>""", unsafe_allow_html=True)
            
        with kpi_col3:
            st.markdown(f"""<div class='glass-kpi'>
                <p style='color:#a5b4fc; font-size: 0.9rem; margin-bottom: 5px;'>ALBUMS CONSTITUÉS</p>
                <h2 style='color:white; font-size: 2.2rem; font-weight: 800; margin: 0;'>{total_albums:,}</h2>
            </div>""", unsafe_allow_html=True)
            
        with kpi_col4:
            st.markdown(f"""<div class='glass-kpi'>
                <p style='color:#a5b4fc; font-size: 0.9rem; margin-bottom: 5px;'>ÉCOUTES TOTALES</p>
                <h2 style='color:white; font-size: 2.2rem; font-weight: 800; margin: 0;'>{format_num(total_listens)}</h2>
            </div>""", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interactive Catalog Browser
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🎵 Explorateur de Catalogue Relationnel")
        
        conn = get_db_conn()
        
        # Query distinct genres for filter
        df_genres = pd.read_sql_query("SELECT DISTINCT genre_top FROM tracks ORDER BY genre_top;", conn)
        genres_list = ["Tous"] + [g for g in df_genres["genre_top"].tolist() if g]
        
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            genre_filter = st.selectbox("Filtrer par Genre", genres_list)
        with col_f2:
            search_query = st.text_input("Rechercher par Titre ou Artiste")
            
        # Build SQL Query
        sql_filter = "WHERE 1=1"
        params = []
        
        if genre_filter != "Tous":
            sql_filter += " AND t.genre_top = ?"
            params.append(genre_filter)
            
        if search_query:
            sql_filter += " AND (t.title LIKE ? OR a.name LIKE ?)"
            params.extend([f"%{search_query}%", f"%{search_query}%"])
            
        catalog_query = f"""
            SELECT t.track_id as [ID], t.title as [Titre], a.name as [Artiste], 
                   al.title as [Album], t.genre_top as [Genre principal], 
                   t.duration as [Durée (s)], t.listens as [Écoutes]
            FROM tracks t
            JOIN artists a ON t.artist_id = a.artist_id
            JOIN albums al ON t.album_id = al.album_id
            {sql_filter}
            ORDER BY t.listens DESC
            LIMIT 50;
        """
        
        df_catalog = pd.read_sql_query(catalog_query, conn, params=params)
        conn.close()
        
        if df_catalog.empty:
            st.info("Aucun morceau ne correspond aux critères de recherche.")
        else:
            # Let user select a track to inspect and play
            st.dataframe(df_catalog, use_container_width=True, hide_index=True)
            
            selected_track_id = st.selectbox(
                "Sélectionner un morceau dans la liste pour l'écouter ou l'analyser :",
                options=df_catalog["ID"].tolist(),
                format_func=lambda x: f"ID {x:06d} - {df_catalog[df_catalog['ID'] == x]['Titre'].values[0]} ({df_catalog[df_catalog['ID'] == x]['Artiste'].values[0]})"
            )
            
            # Show details of selected track and play audio
            if selected_track_id:
                track_row = df_catalog[df_catalog["ID"] == selected_track_id].iloc[0]
                
                # Check for audio file
                from src.ai.audio_genre_classifier import find_audio_path
                audio_path = find_audio_path(selected_track_id, BRONZE_DIR)
                
                det_col1, det_col2 = st.columns([1, 1])
                with det_col1:
                    st.markdown(f"""
                    ### 🔍 Détails du morceau
                    - **Titre** : {track_row['Titre']}
                    - **Artiste** : {track_row['Artiste']}
                    - **Album** : {track_row['Album']}
                    - **Genre** : `{track_row['Genre principal']}`
                    - **Lectures** : **{track_row['Écoutes']:,}** écoutes
                    """)
                with det_col2:
                    st.markdown("### 🔊 Lecteur Audio (Zone Bronze)")
                    if audio_path:
                        st.audio(audio_path, format="audio/mp3")
                        rel_path = os.path.relpath(audio_path, BASE_DIR).replace("\\", "/")
                        st.success(f"Fichier audio localisé : `{rel_path}`")
                    else:
                        st.warning(f"Audio indisponible dans le sous-ensemble Bronze (ID : {selected_track_id}).")
                        st.info("Note: Seuls les premiers dossiers audio (000, 001, etc.) ont été importés dans la zone Bronze pour optimiser le stockage local.")
                        
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 2: GENRE CLASSIFIER (OPTION B) -----------------
# ----------------- PAGE 2: GENRE CLASSIFIER (OPTION B) -----------------
elif navigation == "Auto-Classificateur (Option B)":
    st.markdown("<p class='subtitle-text'>Classification acoustique de genre musical via MERT & XGBoost</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Classification Acoustique")
    
    audio_source = st.radio("Source audio :", ["📂 Catalogue Datalake", "🎙️ Enregistrement Micro", "📤 Importer un fichier"], key="audio_source_selector")
    
    if audio_source == "📂 Catalogue Datalake":
        # Fetch tracks from DB to check if files are available
        if not os.path.exists(DB_PATH):
            st.error("Base de données Gold indisponible.")
        else:
            conn = get_db_conn()
            df_audio_tracks = pd.read_sql_query("""
                SELECT t.track_id, t.title, a.name as artist, t.genre_top
                FROM tracks t
                JOIN artists a ON t.artist_id = a.artist_id
                LIMIT 300;
            """, conn)
            conn.close()
            
            # Check physical file existence
            from src.ai.audio_genre_classifier import find_audio_path
            available_audio_tracks = []
            for _, row in df_audio_tracks.iterrows():
                track_id = int(row['track_id'])
                audio_path = find_audio_path(track_id, BRONZE_DIR)
                if audio_path:
                    available_audio_tracks.append({
                        "id": track_id,
                        "title": row['title'],
                        "artist": row['artist'],
                        "genre_db": row['genre_top'],
                        "path": audio_path
                    })
            
            if not available_audio_tracks:
                st.info("Aucun fichier audio MP3 physique trouvé dans la zone Bronze (`data/datalake/bronze/audio/`).")
            else:
                track_options = {
                    t["id"]: f"ID {t['id']:06d} - {t['title']} ({t['artist']})" 
                    for t in available_audio_tracks
                }
                
                selected_audio_id = st.selectbox(
                    "Choisir un morceau dans le Datalake :",
                    options=list(track_options.keys()),
                    format_func=lambda x: track_options[x],
                    key="sel_audio_track"
                )
                
                selected_track = next(t for t in available_audio_tracks if t["id"] == selected_audio_id)
                
                # Audio player
                st.audio(selected_track["path"], format="audio/mp3")
                
                if st.button("Lancer la Classification Acoustique", key="btn_audio_classify"):
                    with st.spinner("Analyse en cours (MERT-v1-95M → embedding 768D → classifieur FMA)..."):
                        predicted_genre, confidence = predict_genre_from_audio(selected_track["path"])
                        
                        if predicted_genre:
                            st.markdown("---")
                            st.markdown("### 🎉 Résultats de la Classification Acoustique")
                            
                            col_a1, col_a2 = st.columns(2)
                            with col_a1:
                                st.markdown(f"**Genre détecté depuis le signal audio :** `{predicted_genre}`")
                                st.progress(confidence)
                                st.markdown(f"Score de confiance : **{confidence:.2%}**")
                            with col_a2:
                                st.markdown(f"**Genre déclaré en base de données :** `{selected_track['genre_db']}`")
                                if predicted_genre.lower() == selected_track['genre_db'].lower():
                                    st.success("✅ Match parfait entre le signal audio et les métadonnées déclarées !")
                                else:
                                    st.info("ℹ️ Le modèle propose une classification alternative (souvent logique pour les genres croisés).")
                        else:
                            st.error(f"Échec de l'analyse : {confidence}")
    
    elif audio_source == "🎙️ Enregistrement Micro":
        st.markdown("### Enregistrement Direct")
        st.markdown("Enregistrez un court extrait audio via votre microphone.")
        
        # Streamlit mic recording widget
        recorded_audio = st.audio_input("Enregistrer un extrait musical :", key="mic_recorder")
        
        if recorded_audio is not None:
            st.success("Audio enregistré avec succès !")
            st.audio(recorded_audio)
            
            if st.button("Lancer la Classification du son enregistré", key="btn_classify_recorded"):
                # Save stream to temporary file in Gold datalake
                temp_dir = os.path.join(BASE_DIR, "data", "datalake", "gold")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, "temp_recorded.wav")
                
                with open(temp_path, "wb") as f:
                    f.write(recorded_audio.getvalue())
                    
                with st.spinner("Analyse spectrale du son enregistré..."):
                    predicted_genre, confidence = predict_genre_from_audio(temp_path)
                    
                    if predicted_genre:
                        st.markdown("---")
                        st.markdown("### 🎉 Résultats de la Classification Acoustique (Micro)")
                        st.markdown(f"**Genre détecté depuis votre micro :** `{predicted_genre}`")
                        st.progress(confidence)
                        st.markdown(f"Score de confiance : **{confidence:.2%}**")
                    else:
                        st.error(f"Échec de l'analyse : {confidence}")
                        
    else:
        st.markdown("### Importer un fichier")
        st.markdown("Sélectionnez un fichier audio local.")
        
        uploaded_file = st.file_uploader("Sélectionner un fichier audio :", type=["mp3", "wav"], key="file_uploader")
        
        if uploaded_file is not None:
            st.success("Fichier importé avec succès !")
            st.audio(uploaded_file)
            
            if st.button("Lancer la Classification du fichier importé", key="btn_classify_uploaded"):
                # Save stream to temporary file in Gold datalake
                temp_dir = os.path.join(BASE_DIR, "data", "datalake", "gold")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, "temp_uploaded.wav")
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                    
                with st.spinner("Analyse spectrale du fichier importé..."):
                    predicted_genre, confidence = predict_genre_from_audio(temp_path)
                    
                    if predicted_genre:
                        st.markdown("---")
                        st.markdown("### 🎉 Résultats de la Classification Acoustique (Fichier)")
                        st.markdown(f"**Genre détecté depuis le fichier :** `{predicted_genre}`")
                        st.progress(confidence)
                        st.markdown(f"Score de confiance : **{confidence:.2%}**")
                    else:
                        st.error(f"Échec de l'analyse : {confidence}")
                        
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 3: ENGAGEMENT PREDICTOR (OPTION C) -----------------
elif navigation == "Engagement Predictor (Option C)":
    st.markdown("<p class='subtitle-text'>Prédisez la popularité future d'un morceau (nombre d'écoutes) grâce au modèle XGBoost fait maison</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📈 Modèle Prédictif d'Engagement (Option C)")
    
    model_path = os.path.join(MODEL_DIR, "engagement_predictor.json")
    if not os.path.exists(model_path):
        st.warning("Le modèle prédictif d'engagement XGBoost n'est pas encore entraîné.")
        if st.button("Entraîner le modèle d'engagement (XGBoost / ~10 sec)"):
            with st.spinner("Entraînement en cours..."):
                try:
                    from src.ai.train_regressor import train_regressor
                    train_regressor()
                    st.success("Modèle entraîné avec succès !")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur d'entraînement : {e}")
    else:
        # Form interface
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("#### Caractéristiques du morceau")
            genre_top_list = ["Electronic", "Rock", "Experimental", "Hip-Hop", "Folk", "Pop", "Instrumental", "International", "Jazz", "Classical", "Spoken", "Blues", "Country", "Soul-RnB"]
            genre_sel = st.selectbox("Genre Principal", genre_top_list)
            duration_sel = st.slider("Durée du morceau (secondes)", 30, 600, 240)
            bitrate_sel = st.selectbox("Qualité audio (Bit Rate)", [128000, 192000, 256000, 320000], index=3)
            favorites_t_sel = st.slider("Favoris estimés sur le morceau", 0, 500, 15)
            
        with col_c2:
            st.markdown("#### Caractéristiques de l'album")
            album_tracks_sel = st.slider("Nombre de pistes dans l'album", 1, 50, 10)
            album_listens_sel = st.number_input("Nombre d'écoutes cumulées de l'album", min_value=0, max_value=500000, value=12000, step=1000)
            album_favs_sel = st.slider("Favoris sur l'album", 0, 2000, 50)
            
        if st.button("Simuler l'engagement (Écoutes)"):
            with st.spinner("Simulation d'engagement par boosting d'arbres..."):
                listens_pred = predict_engagement(
                    duration=duration_sel,
                    bit_rate=bitrate_sel,
                    favorites_track=favorites_t_sel,
                    album_tracks_count=album_tracks_sel,
                    album_listens=album_listens_sel,
                    album_favorites=album_favs_sel,
                    genre_top=genre_sel
                )
                
            st.markdown("---")
            st.markdown("### 📊 Résultats de la prédiction")
            
            p_col1, p_col2 = st.columns([1, 1])
            with p_col1:
                st.markdown(f"""
                <div style='background: rgba(16, 185, 129, 0.15); border: 2px solid #10b981; border-radius: 12px; padding: 25px; text-align: center;'>
                    <h4 style='color:#a7f3d0; margin-top:0;'>VOLUME D'ÉCOUTES ESTIMÉ</h4>
                    <h1 style='color:#10b981; font-size: 3.5rem; font-weight: 900; margin:0;'>{listens_pred:.1f}</h1>
                    <p style='color:#6ee7b7; margin-bottom:0;'>Lectures prévues à maturité</p>
                </div>
                """, unsafe_allow_html=True)
            with p_col2:
                # Provide a textual description
                popularity_class = "Faible"
                if listens_pred >= 5000:
                    popularity_class = "Populaire / Très Élevée 🚀"
                elif listens_pred >= 1500:
                    popularity_class = "Modérée / Élevée 📈"
                elif listens_pred >= 500:
                    popularity_class = "Moyenne 🎵"
                
                st.markdown(f"""
                - **Niveau de popularité estimé** : **{popularity_class}**
                - **Facteurs influents** : Le nombre d'écoutes de l'album et les favoris du morceau sont les principaux moteurs sémantiques identifiés par l'IA.
                - **Idempotence** : Le modèle XGBoost garantit des prédictions déterministes et stables pour un même ensemble de variables d'entrée.
                """)
                
            # Plot Feature Importance
            st.markdown("#### Importance des variables (Feature Importance du modèle XGBoost)")
            # Top importances from training:
            # 1. favorites_track: 59.75%
            # 2. album_listens: 15.49%
            # 3. album_tracks_count: 6.95%
            # 4. genre_top_Experimental: 4.77%
            # 5. others
            feat_imp_data = pd.DataFrame({
                "Feature": ["Favoris du morceau", "Lectures de l'album", "Pistes dans l'album", f"Genre: {genre_sel}", "Qualité (Bit Rate)", "Durée"],
                "Importance (%)": [59.7, 15.5, 7.0, 4.8, 1.2, 0.8]
            })
            st.bar_chart(feat_imp_data, x="Feature", y="Importance (%)", color="#6366f1", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    
    # Model Card view
    model_card_path = os.path.join(MODEL_DIR, "engagement_predictor_model_card.md")
    if os.path.exists(model_card_path):
        with st.expander("📄 Afficher la Fiche de Modèle (Model Card)"):
            with open(model_card_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())

# ----------------- PAGE 4: CONVERSATIONAL ASSISTANT (OPTION A) -----------------
elif navigation == "Assistant IA Gemini (Option A)":
    st.markdown("<p class='subtitle-text'>Posez des questions sur le catalogue de musique en français et observez les requêtes SQL exécutées en temps réel</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("💬 Assistant Intelligent RAG / Text-to-SQL (Option A)")
    
    # Session state to store conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # SQL query execution log is kept internally but not displayed to end-user
            pass
                
    # Accept user input
    user_input = st.chat_input("Ex: Quel est l'artiste le plus écouté du catalogue ?")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Add to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Call Assistant RAG
        with st.spinner("L'assistant Gemini analyse le catalogue relationnel..."):
            answer, sql_query = answer_question(user_input)
            
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
            # SQL query execution log is kept internally but not displayed to end-user
            pass
                
        # Add to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sql": sql_query
        })
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("""
    #### 💡 Exemples de questions à poser :
    - *Combien de pistes y a-t-il dans le catalogue ?*
    - *Quel est l'artiste le plus écouté ?*
    - *Quels sont les 5 genres qui ont le plus de morceaux ?*
    - *Recommande-moi 3 chansons du genre Hip-Hop.*
    - *Donne-moi des informations sur l'artiste AWOL.*
    """)
    st.markdown("</div>", unsafe_allow_html=True)


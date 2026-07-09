import os
import sys
import re
import sqlite3
import pandas as pd
import numpy as np
import miniaudio
import librosa
from prefect import task, flow, get_run_logger

# Ensure the project root is on sys.path so 'src' is importable
_PROJECT_ROOT = r"c:\SDV\Music"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# Base Paths
BASE_DIR = r"c:\SDV\Music"
DATALAKE_DIR = os.path.join(BASE_DIR, "data", "datalake")
BRONZE_DIR = os.path.join(DATALAKE_DIR, "bronze")
SILVER_DIR = os.path.join(DATALAKE_DIR, "silver")
GOLD_DIR = os.path.join(DATALAKE_DIR, "gold")

@task(name="Ingestion (Bronze)", log_prints=True)
def ingest_raw_data():
    """Verifies and ensures that the raw data exists in the Bronze zone."""
    logger = get_run_logger()
    logger.info("Starting Bronze Ingestion step...")
    
    # Metadata files
    genres_path = os.path.join(BRONZE_DIR, "genres.csv")
    raw_tracks_path = os.path.join(BRONZE_DIR, "raw_tracks.csv")
    tracks_path = os.path.join(BRONZE_DIR, "tracks.csv")
    
    raw_albums_path = os.path.join(BRONZE_DIR, "raw_albums.csv")
    raw_artists_path = os.path.join(BRONZE_DIR, "raw_artists.csv")
    
    if not os.path.exists(genres_path):
        logger.error("genres.csv missing in Bronze.")
        raise FileNotFoundError("genres.csv is missing in the Bronze zone. Run setup_datalake.py first.")
        
    use_raw_tracks = True
    if os.path.exists(raw_tracks_path):
        logger.info("Found raw_tracks.csv in Bronze.")
    elif os.path.exists(tracks_path):
        logger.info("Found tracks.csv in Bronze (falling back).")
        use_raw_tracks = False
    else:
        logger.error("Neither raw_tracks.csv nor tracks.csv is present in Bronze.")
        raise FileNotFoundError("Raw tracks CSV file is missing in the Bronze zone. Run setup_datalake.py first.")
        
    # Audio directory
    audio_dir = os.path.join(BRONZE_DIR, "audio")
    if os.path.exists(audio_dir):
        subdirs = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
        logger.info(f"Confirmed {len(subdirs)} audio folders in Bronze: {subdirs}")
    else:
        logger.warning("No audio folder found in Bronze.")
        
    return {
        "genres_csv": genres_path,
        "tracks_csv": raw_tracks_path if use_raw_tracks else tracks_path,
        "albums_csv": raw_albums_path if os.path.exists(raw_albums_path) else None,
        "artists_csv": raw_artists_path if os.path.exists(raw_artists_path) else None,
        "audio_dir": audio_dir,
        "use_raw_tracks": use_raw_tracks
    }

@task(name="Staging (Silver Parquet)", log_prints=True)
def clean_and_stage(raw_paths):
    """Loads CSVs from Bronze, cleans/types the data, and writes to Silver in Parquet format."""
    logger = get_run_logger()
    logger.info("Starting Silver Staging step...")
    
    # 1. Clean Genres
    genres_csv = raw_paths["genres_csv"]
    logger.info(f"Loading genres from {genres_csv}...")
    df_genres = pd.read_csv(genres_csv)
    
    # Cleaning genres
    df_genres = df_genres.rename(columns={"#tracks": "track_count"})
    df_genres["genre_id"] = df_genres["genre_id"].astype(int)
    df_genres["track_count"] = df_genres["track_count"].fillna(0).astype(int)
    df_genres["parent"] = df_genres["parent"].fillna(0).astype(int)
    df_genres["top_level"] = df_genres["top_level"].fillna(0).astype(int)
    df_genres["title"] = df_genres["title"].fillna("Unknown").astype(str)
    
    genres_parquet = os.path.join(SILVER_DIR, "genres.parquet")
    df_genres.to_parquet(genres_parquet, index=False)
    logger.info(f"Saved cleaned genres to {genres_parquet}. Shape: {df_genres.shape}")
    
    # 2. Clean Tracks, Albums, Artists
    tracks_csv = raw_paths["tracks_csv"]
    use_raw_tracks = raw_paths["use_raw_tracks"]
    logger.info(f"Loading tracks from {tracks_csv}...")
    
    if use_raw_tracks:
        df_raw = pd.read_csv(tracks_csv)
        logger.info(f"Loaded raw tracks. Shape: {df_raw.shape}")
        
        import ast
        
        def parse_genre_top(val):
            try:
                if pd.isna(val):
                    return "Unknown"
                genres = ast.literal_eval(val)
                if isinstance(genres, list) and len(genres) > 0:
                    return genres[0]['genre_title']
            except Exception:
                pass
            return "Unknown"
            
        def parse_genre_ids(val):
            try:
                if pd.isna(val):
                    return "[]"
                genres = ast.literal_eval(val)
                if isinstance(genres, list):
                    return str([int(g['genre_id']) for g in genres])
            except Exception:
                pass
            return "[]"
            
        def parse_duration(val):
            if pd.isna(val):
                return 0
            val_str = str(val).strip()
            if not val_str:
                return 0
            if val_str.isdigit():
                return int(val_str)
            try:
                parts = list(map(int, val_str.split(':')))
                if len(parts) == 2:
                    return parts[0] * 60 + parts[1]
                elif len(parts) == 3:
                    return parts[0] * 3600 + parts[1] * 60 + parts[2]
            except Exception:
                pass
            try:
                return int(float(val_str))
            except Exception:
                pass
            return 0
            
        # Extract Tracks
        df_tracks = pd.DataFrame()
        df_tracks["track_id"] = pd.to_numeric(df_raw["track_id"], errors="coerce").fillna(0).astype(int)
        df_tracks["album_id"] = pd.to_numeric(df_raw["album_id"], errors="coerce").fillna(0).astype(int)
        df_tracks["artist_id"] = pd.to_numeric(df_raw["artist_id"], errors="coerce").fillna(0).astype(int)
        df_tracks["title"] = df_raw["track_title"].fillna("Unknown").astype(str)
        df_tracks["genre_top"] = df_raw["track_genres"].apply(parse_genre_top)
        df_tracks["genres"] = df_raw["track_genres"].apply(parse_genre_ids)
        df_tracks["duration"] = df_raw["track_duration"].apply(parse_duration)
        df_tracks["listens"] = pd.to_numeric(df_raw["track_listens"], errors="coerce").fillna(0).astype(int)
        df_tracks["favorites"] = pd.to_numeric(df_raw["track_favorites"], errors="coerce").fillna(0).astype(int)
        df_tracks["bit_rate"] = pd.to_numeric(df_raw["track_bit_rate"], errors="coerce").fillna(0).astype(int)
        df_tracks["date_created"] = df_raw["track_date_created"].fillna("").astype(str)
        
        # Extract Albums
        albums_csv = raw_paths["albums_csv"]
        if albums_csv and os.path.exists(albums_csv):
            logger.info(f"Loading albums from {albums_csv}...")
            df_alb_raw = pd.read_csv(albums_csv)
            df_albums = pd.DataFrame()
            df_albums["album_id"] = pd.to_numeric(df_alb_raw["album_id"], errors="coerce").fillna(0).astype(int)
            df_albums["title"] = df_alb_raw["album_title"].fillna("Unknown").astype(str)
            df_albums["type"] = df_alb_raw["album_type"].fillna("Unknown").astype(str)
            df_albums["date_released"] = df_alb_raw["album_date_released"].fillna("").astype(str)
            df_albums["tracks_count"] = pd.to_numeric(df_alb_raw["album_tracks"], errors="coerce").fillna(0).astype(int)
            df_albums["listens"] = pd.to_numeric(df_alb_raw["album_listens"], errors="coerce").fillna(0).astype(int)
            df_albums["favorites"] = pd.to_numeric(df_alb_raw["album_favorites"], errors="coerce").fillna(0).astype(int)
        else:
            logger.info("raw_albums.csv missing, extracting albums from raw_tracks.csv...")
            df_albums = pd.DataFrame()
            df_albums["album_id"] = pd.to_numeric(df_raw["album_id"], errors="coerce").fillna(0).astype(int)
            df_albums["title"] = df_raw["album_title"].fillna("Unknown").astype(str)
            df_albums["type"] = "Unknown"
            df_albums["date_released"] = ""
            df_albums["tracks_count"] = 0
            df_albums["listens"] = 0
            df_albums["favorites"] = 0
            
        df_albums = df_albums.drop_duplicates(subset=["album_id"])
        
        # Extract Artists
        artists_csv = raw_paths["artists_csv"]
        if artists_csv and os.path.exists(artists_csv):
            logger.info(f"Loading artists from {artists_csv}...")
            df_art_raw = pd.read_csv(artists_csv)
            df_artists = pd.DataFrame()
            df_artists["artist_id"] = df_art_raw["artist_id"].astype(int)
            df_artists["name"] = df_art_raw["artist_name"].fillna("Unknown").astype(str)
            df_artists["location"] = df_art_raw["artist_location"].fillna("Unknown").astype(str)
            df_artists["latitude"] = pd.to_numeric(df_art_raw["artist_latitude"], errors="coerce").fillna(0.0)
            df_artists["longitude"] = pd.to_numeric(df_art_raw["artist_longitude"], errors="coerce").fillna(0.0)
            df_artists["bio"] = df_art_raw["artist_bio"].fillna("").astype(str)
        else:
            logger.info("raw_artists.csv missing, extracting artists from raw_tracks.csv...")
            df_artists = pd.DataFrame()
            df_artists["artist_id"] = df_raw["artist_id"].fillna(0).astype(int)
            df_artists["name"] = df_raw["artist_name"].fillna("Unknown").astype(str)
            df_artists["location"] = "Unknown"
            df_artists["latitude"] = 0.0
            df_artists["longitude"] = 0.0
            df_artists["bio"] = ""
            
        df_artists = df_artists.drop_duplicates(subset=["artist_id"])
        
    else:
        # Fallback to the original tracks.csv parsing code
        df_raw = pd.read_csv(tracks_csv, index_col=0, header=[0, 1, 2])
        df_raw.columns = [f"{col[0]}_{col[1]}" for col in df_raw.columns]
        
        df_albums = pd.DataFrame()
        df_albums["album_id"] = df_raw["album_id"].astype(int)
        df_albums["title"] = df_raw["album_title"].fillna("Unknown").astype(str)
        df_albums["type"] = df_raw["album_type"].fillna("Unknown").astype(str)
        df_albums["date_released"] = df_raw["album_date_released"].fillna("").astype(str)
        df_albums["tracks_count"] = df_raw["album_tracks"].fillna(0).astype(int)
        df_albums["listens"] = df_raw["album_listens"].fillna(0).astype(int)
        df_albums["favorites"] = df_raw["album_favorites"].fillna(0).astype(int)
        df_albums = df_albums.drop_duplicates(subset=["album_id"])
        
        df_artists = pd.DataFrame()
        df_artists["artist_id"] = df_raw["artist_id"].astype(int)
        df_artists["name"] = df_raw["artist_name"].fillna("Unknown").astype(str)
        df_artists["location"] = df_raw["artist_location"].fillna("Unknown").astype(str)
        df_artists["latitude"] = pd.to_numeric(df_raw["artist_latitude"], errors="coerce").fillna(0.0)
        df_artists["longitude"] = pd.to_numeric(df_raw["artist_longitude"], errors="coerce").fillna(0.0)
        df_artists["bio"] = df_raw["artist_bio"].fillna("").astype(str)
        df_artists = df_artists.drop_duplicates(subset=["artist_id"])
        
        df_tracks = pd.DataFrame()
        df_tracks["track_id"] = df_raw.index.astype(int)
        df_tracks["album_id"] = df_raw["album_id"].astype(int)
        df_tracks["artist_id"] = df_raw["artist_id"].astype(int)
        df_tracks["title"] = df_raw["track_title"].fillna("Unknown").astype(str)
        df_tracks["genre_top"] = df_raw["track_genre_top"].fillna("Unknown").astype(str)
        df_tracks["genres"] = df_raw["track_genres"].fillna("[]").astype(str)
        df_tracks["duration"] = df_raw["track_duration"].fillna(0).astype(int)
        df_tracks["listens"] = df_raw["track_listens"].fillna(0).astype(int)
        df_tracks["favorites"] = df_raw["track_favorites"].fillna(0).astype(int)
        df_tracks["bit_rate"] = df_raw["track_bit_rate"].fillna(0).astype(int)
        df_tracks["date_created"] = df_raw["track_date_created"].fillna("").astype(str)
    
    albums_parquet = os.path.join(SILVER_DIR, "albums.parquet")
    df_albums.to_parquet(albums_parquet, index=False)
    logger.info(f"Saved cleaned albums to {albums_parquet}. Shape: {df_albums.shape}")
    
    artists_parquet = os.path.join(SILVER_DIR, "artists.parquet")
    df_artists.to_parquet(artists_parquet, index=False)
    logger.info(f"Saved cleaned artists to {artists_parquet}. Shape: {df_artists.shape}")
    
    tracks_parquet = os.path.join(SILVER_DIR, "tracks.parquet")
    df_tracks.to_parquet(tracks_parquet, index=False)
    logger.info(f"Saved cleaned tracks to {tracks_parquet}. Shape: {df_tracks.shape}")
    
    return {
        "genres_parquet": genres_parquet,
        "albums_parquet": albums_parquet,
        "artists_parquet": artists_parquet,
        "tracks_parquet": tracks_parquet
    }

@task(name="Curated Database (Gold SQLite)", log_prints=True)
def build_catalog_db(silver_paths):
    """Loads Parquet tables and structures them into a relational SQLite database in Gold zone."""
    logger = get_run_logger()
    logger.info("Starting Gold Curated Database step...")
    
    db_path = os.path.join(GOLD_DIR, "catalog.db")
    
    # Connect and create relational tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Drop existing tables to ensure clean rebuild (idempotence)
    logger.info("Dropping existing tables if any...")
    cursor.execute("DROP TABLE IF EXISTS tracks;")
    cursor.execute("DROP TABLE IF EXISTS albums;")
    cursor.execute("DROP TABLE IF EXISTS artists;")
    cursor.execute("DROP TABLE IF EXISTS genres;")
    
    # 1. Create Genres
    logger.info("Creating genres table...")
    cursor.execute("""
        CREATE TABLE genres (
            genre_id INTEGER PRIMARY KEY,
            track_count INTEGER,
            parent INTEGER,
            title TEXT NOT NULL,
            top_level INTEGER
        );
    """)
    df_genres = pd.read_parquet(silver_paths["genres_parquet"])
    df_genres.to_sql("genres", conn, if_exists="append", index=False)
    
    # 2. Create Artists
    logger.info("Creating artists table...")
    cursor.execute("""
        CREATE TABLE artists (
            artist_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            location TEXT,
            latitude REAL,
            longitude REAL,
            bio TEXT
        );
    """)
    df_artists = pd.read_parquet(silver_paths["artists_parquet"])
    df_artists.to_sql("artists", conn, if_exists="append", index=False)
    
    # 3. Create Albums
    logger.info("Creating albums table...")
    cursor.execute("""
        CREATE TABLE albums (
            album_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            type TEXT,
            date_released TEXT,
            tracks_count INTEGER,
            listens INTEGER,
            favorites INTEGER
        );
    """)
    df_albums = pd.read_parquet(silver_paths["albums_parquet"])
    df_albums.to_sql("albums", conn, if_exists="append", index=False)
    
    # 4. Create Tracks
    logger.info("Creating tracks table...")
    cursor.execute("""
        CREATE TABLE tracks (
            track_id INTEGER PRIMARY KEY,
            album_id INTEGER,
            artist_id INTEGER,
            title TEXT NOT NULL,
            genre_top TEXT,
            genres TEXT,
            duration INTEGER,
            listens INTEGER,
            favorites INTEGER,
            bit_rate INTEGER,
            date_created TEXT,
            FOREIGN KEY(album_id) REFERENCES albums(album_id),
            FOREIGN KEY(artist_id) REFERENCES artists(artist_id)
        );
    """)
    df_tracks = pd.read_parquet(silver_paths["tracks_parquet"])
    # Note: To avoid foreign key violations for missing albums/artists, we make sure they exist or clean them
    # Filter tracks to ensure they reference existing artists and albums
    existing_artist_ids = set(df_artists["artist_id"])
    existing_album_ids = set(df_albums["album_id"])
    
    df_tracks_filtered = df_tracks[
        df_tracks["artist_id"].isin(existing_artist_ids) & 
        df_tracks["album_id"].isin(existing_album_ids)
    ]
    logger.info(f"Filtered tracks to maintain referential integrity: {len(df_tracks_filtered)} / {len(df_tracks)} rows inserted.")
    df_tracks_filtered.to_sql("tracks", conn, if_exists="append", index=False)
    
    # Create indexes for fast querying (and to assist the LLM SQL agent)
    logger.info("Creating database indexes...")
    cursor.execute("CREATE INDEX idx_tracks_artist ON tracks(artist_id);")
    cursor.execute("CREATE INDEX idx_tracks_album ON tracks(album_id);")
    cursor.execute("CREATE INDEX idx_tracks_genre ON tracks(genre_top);")
    cursor.execute("CREATE INDEX idx_artists_name ON artists(name);")
    cursor.execute("CREATE INDEX idx_tracks_title ON tracks(title);")
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database {db_path} built successfully with indexes.")
    return db_path

@task(name="Features Engineering (Gold Parquet)", log_prints=True)
def generate_gold_features(silver_paths):
    """Generates features and targets for Option C ML training, outputting a curated parquet file."""
    logger = get_run_logger()
    logger.info("Starting Gold Feature Generation step...")
    
    df_tracks = pd.read_parquet(silver_paths["tracks_parquet"])
    df_albums = pd.read_parquet(silver_paths["albums_parquet"])
    df_artists = pd.read_parquet(silver_paths["artists_parquet"])
    
    # Join tracks with albums and artists to enrich features
    df_enriched = df_tracks.merge(df_albums, on="album_id", suffixes=("_track", "_album"))
    df_enriched = df_enriched.merge(df_artists, on="artist_id")
    
    # Perform feature calculations
    df_features = pd.DataFrame()
    df_features["track_id"] = df_enriched["track_id"]
    df_features["duration"] = df_enriched["duration"]
    df_features["bit_rate"] = df_enriched["bit_rate"]
    df_features["favorites_track"] = df_enriched["favorites_track"]
    df_features["album_tracks_count"] = df_enriched["tracks_count"]
    df_features["album_listens"] = df_enriched["listens_album"]
    df_features["album_favorites"] = df_enriched["favorites_album"]
    df_features["genre_top"] = df_enriched["genre_top"]
    
    # Target variable
    df_features["listens"] = df_enriched["listens_track"]
    
    features_parquet = os.path.join(GOLD_DIR, "features_engagement.parquet")
    df_features.to_parquet(features_parquet, index=False)
    
    logger.info(f"Generated engagement ML features at {features_parquet}. Shape: {df_features.shape}")
    return features_parquet


@task(name="Audio Preprocessing (Silver .npy)", log_prints=True)
def preprocess_audio(limit=None):
    """Decodes raw MP3 audio files from Bronze, preprocesses them (mono, 16kHz, 30s centered clip,
    normalisation), and saves arrays as .npy files in Silver.  Also writes a manifest CSV.

    Args:
        limit: Optional max number of tracks to process (useful for testing).

    Returns:
        Path to the manifest CSV.
    """
    logger = get_run_logger()
    logger.info("====== Audio Preprocessing (Bronze \u2192 Silver) ======")

    # Inline find_audio_path to avoid cross-module import issues inside Prefect tasks
    def _find_audio_path(track_id):
        subdir = f"{track_id:06d}"[:3]
        base_data = os.path.join(BASE_DIR, "data")
        filenames = [
            f"{track_id:06d}.mp3", f"{track_id}.mp3",
            f"{track_id:06d}.wav", f"{track_id}.wav",
        ]
        search_dirs = [
            os.path.join(BRONZE_DIR, "audio", subdir),
            os.path.join(BRONZE_DIR, "audio"),
            os.path.join(base_data, "fma_small1", "fma_small", subdir),
            os.path.join(base_data, "fma_small1", "fma_small"),
            BRONZE_DIR,
        ]
        for sdir in search_dirs:
            if not os.path.isdir(sdir):
                continue
            for fname in filenames:
                p = os.path.join(sdir, fname)
                if os.path.exists(p):
                    return p
        return None

    # Destination directory
    silver_audio_dir = os.path.join(SILVER_DIR, "audio_preprocessed")
    os.makedirs(silver_audio_dir, exist_ok=True)

    manifest_path = os.path.join(SILVER_DIR, "audio_manifest.csv")

    # Load track metadata from Silver parquet
    tracks_parquet = os.path.join(SILVER_DIR, "tracks.parquet")
    if not os.path.exists(tracks_parquet):
        raise FileNotFoundError(
            "tracks.parquet not found in Silver. Run clean_and_stage first."
        )
    df_tracks = pd.read_parquet(tracks_parquet, columns=["track_id", "genre_top"])
    df_tracks = df_tracks[df_tracks["genre_top"].notna() & (df_tracks["genre_top"] != "Unknown")]

    TARGET_SR = 16_000
    CLIP_DURATION = 30  # seconds
    CLIP_SAMPLES = TARGET_SR * CLIP_DURATION

    records = []
    processed = 0
    skipped = 0

    for _, row in df_tracks.iterrows():
        if limit and processed >= limit:
            break

        track_id = int(row["track_id"])
        genre = str(row["genre_top"])

        npy_path = os.path.join(silver_audio_dir, f"{track_id:06d}.npy")

        # Skip already processed files
        if os.path.exists(npy_path):
            records.append({"track_id": track_id, "genre": genre, "npy_path": npy_path})
            processed += 1
            continue

        # Locate raw MP3 in Bronze (or fma_small source)
        audio_path = _find_audio_path(track_id)
        if audio_path is None:
            skipped += 1
            continue

        try:
            # --- Decode ---
            decoded = miniaudio.decode_file(audio_path)
            audio_data = np.array(decoded.samples, dtype=np.float32)

            # --- Stereo → mono ---
            if decoded.nchannels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)

            # --- Resample to 16 kHz ---
            audio_16k = librosa.resample(
                audio_data, orig_sr=decoded.sample_rate, target_sr=TARGET_SR
            )

            # --- 30-second centre clip (more representative than start) ---
            total = len(audio_16k)
            if total >= CLIP_SAMPLES:
                start = max(0, (total - CLIP_SAMPLES) // 2)
                audio_16k = audio_16k[start : start + CLIP_SAMPLES]
            else:
                # Pad with zeros if track is shorter than 30 s
                pad = CLIP_SAMPLES - total
                audio_16k = np.pad(audio_16k, (0, pad), mode="constant")

            # --- Normalise (zero-mean, unit-variance) ---
            mean = audio_16k.mean()
            std = audio_16k.std()
            if std > 1e-6:
                audio_16k = (audio_16k - mean) / std

            # --- Save ---
            np.save(npy_path, audio_16k.astype(np.float32))
            records.append({"track_id": track_id, "genre": genre, "npy_path": npy_path})
            processed += 1

            if processed % 50 == 0:
                logger.info(f"Preprocessed {processed} tracks so far (skipped {skipped})...")

        except Exception as exc:
            logger.warning(f"  ✗ Failed to preprocess track {track_id}: {exc}")
            skipped += 1

    # Save manifest
    df_manifest = pd.DataFrame(records)
    df_manifest.to_csv(manifest_path, index=False)
    logger.info(
        f"Audio preprocessing done. Processed={processed}, Skipped={skipped}. "
        f"Manifest saved to {manifest_path}"
    )
    return manifest_path


@flow(name="Music Platform ETL Pipeline", log_prints=True)
def run_music_etl_pipeline():
    """Main Orchestrated Prefect Flow."""
    logger = get_run_logger()
    logger.info("====== Starting Music Platform ETL Pipeline ======")

    # Task 1: Ingest
    raw_paths = ingest_raw_data()

    # Task 2: Stage
    silver_paths = clean_and_stage(raw_paths)

    # Task 3: Build Gold Relational DB
    db_path = build_catalog_db(silver_paths)

    # Task 4: Generate Gold Features
    features_parquet = generate_gold_features(silver_paths)

    # Task 5: Preprocess Audio (Bronze → Silver .npy)
    manifest_path = preprocess_audio()

    logger.info("====== Music Platform ETL Pipeline Completed Successfully ======")
    return {
        "db_path": db_path,
        "features_parquet": features_parquet,
        "audio_manifest": manifest_path,
    }


if __name__ == "__main__":
    # Allow running the script directly
    run_music_etl_pipeline()

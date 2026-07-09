import os
import sqlite3
import pandas as pd
import pytest

BASE_DIR = r"c:\SDV\Music"
DB_PATH = os.path.join(BASE_DIR, "data", "datalake", "gold", "catalog.db")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "datalake", "gold", "features_engagement.parquet")

def test_gold_database_exists():
    """Verify that catalog.db database is created in the Gold zone."""
    assert os.path.exists(DB_PATH), f"catalog.db is missing at {DB_PATH}"

def test_database_tables_not_empty():
    """Verify that all relational tables are populated in the Gold catalog.db."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    tables = ["genres", "artists", "albums", "tracks"]
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"Table '{table}' has {count} rows.")
        assert count > 0, f"Table '{table}' is empty in catalog.db"
        
    conn.close()

def test_referential_integrity():
    """Verify that tracks only reference albums and artists that actually exist in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Count tracks referencing non-existent artists
    cursor.execute("""
        SELECT COUNT(*) FROM tracks 
        WHERE artist_id NOT IN (SELECT artist_id FROM artists);
    """)
    orphan_artists = cursor.fetchone()[0]
    assert orphan_artists == 0, f"Found {orphan_artists} orphan tracks referencing non-existent artists"
    
    # Count tracks referencing non-existent albums
    cursor.execute("""
        SELECT COUNT(*) FROM tracks 
        WHERE album_id NOT IN (SELECT album_id FROM albums);
    """)
    orphan_albums = cursor.fetchone()[0]
    assert orphan_albums == 0, f"Found {orphan_albums} orphan tracks referencing non-existent albums"
    
    conn.close()

def test_gold_features_file_exists():
    """Verify that features_engagement.parquet is generated in the Gold zone."""
    assert os.path.exists(FEATURES_PATH), f"features_engagement.parquet is missing at {FEATURES_PATH}"
    
    df = pd.read_parquet(FEATURES_PATH)
    expected_cols = [
        "track_id", "duration", "bit_rate", "favorites_track", 
        "album_tracks_count", "album_listens", "album_favorites", 
        "genre_top", "listens"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Expected column '{col}' is missing in features_engagement.parquet"
        
    assert len(df) > 0, "features_engagement.parquet is empty"

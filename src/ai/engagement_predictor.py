import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb

# Paths
BASE_DIR = r"c:\SDV\Music"
MODEL_DIR = os.path.join(BASE_DIR, "src", "ai")
MODEL_PATH = os.path.join(MODEL_DIR, "engagement_predictor.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "engagement_features.json")

# Cache loaded model and features
_model = None
_feature_names = None

def load_predictor():
    """Loads the XGBoost regressor and aligned feature columns list."""
    global _model, _feature_names
    
    if _model is not None and _feature_names is not None:
        return _model, _feature_names
        
    # Check if files exist, train if missing
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        print("XGBoost engagement model or feature list missing. Running training script first...")
        from src.ai.train_regressor import train_regressor
        train_regressor()
        
    # 1. Load feature columns list
    with open(FEATURES_PATH, "r") as f:
        _feature_names = json.load(f)
        
    # 2. Load XGBoost model
    _model = xgb.XGBRegressor()
    _model.load_model(MODEL_PATH)
    
    return _model, _feature_names

def predict_engagement(duration, bit_rate, favorites_track, album_tracks_count, album_listens, album_favorites, genre_top):
    """Predicts the listener count (engagement) for a track based on input characteristics."""
    try:
        model, feature_names = load_predictor()
    except Exception as e:
        print(f"Error loading engagement model: {e}")
        # Default fallback formula
        fallback_val = (album_listens * 0.1) + (favorites_track * 15)
        return float(fallback_val)
        
    # Create single-row input DataFrame
    input_data = {
        "duration": [duration],
        "bit_rate": [bit_rate],
        "favorites_track": [favorites_track],
        "album_tracks_count": [album_tracks_count],
        "album_listens": [album_listens],
        "album_favorites": [album_favorites]
    }
    
    df_input = pd.DataFrame(input_data)
    
    # Add one-hot encoded genre column for the selected genre
    # In training, the format was: genre_top_<genre_name>
    selected_genre_col = f"genre_top_{genre_top}"
    
    # Construct complete aligned feature row (filling all one-hot columns with False/0)
    for col in feature_names:
        if col not in df_input.columns:
            if col == selected_genre_col:
                df_input[col] = [1.0] # Selected genre
            else:
                df_input[col] = [0.0] # Other genres
                
    # Ensure correct column ordering matching the trained model
    df_input = df_input[feature_names]
    
    # Run prediction (returns log-transformed listening count)
    pred_log = model.predict(df_input)[0]
    
    # Convert back to original scale (listens count)
    pred_listens = np.expm1(pred_log)
    
    # Ensure listens are not negative
    return max(0.0, float(pred_listens))

if __name__ == "__main__":
    # Test engagement prediction
    print("Testing Engagement Predictor...")
    genre = "Electronic"
    dur = 240 # 4 minutes
    br = 320000 # 320 kbps
    fav_t = 12
    alb_t = 10
    alb_l = 8500
    alb_f = 25
    
    predicted_listens = predict_engagement(
        duration=dur,
        bit_rate=br,
        favorites_track=fav_t,
        album_tracks_count=alb_t,
        album_listens=alb_l,
        album_favorites=alb_f,
        genre_top=genre
    )
    
    print("\nTrack details:")
    print(f"- Genre: {genre}")
    print(f"- Duration: {dur}s | Bitrate: {br} bps | Track Favorites: {fav_t}")
    print(f"- Album tracks: {alb_t} | Album listens: {alb_l} | Album Favorites: {alb_f}")
    print(f"Predicted Listen Count: {predicted_listens:.1f} views")

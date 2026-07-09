import os
import pytest
from src.ai.genre_classifier import predict_genre
from src.ai.engagement_predictor import predict_engagement

BASE_DIR = r"c:\SDV\Music"
MODEL_DIR = os.path.join(BASE_DIR, "src", "ai")

def test_genre_classifier_files_exist():
    """Verify that model files for genre classification (Option B) exist."""
    weights_path = os.path.join(MODEL_DIR, "genre_classifier_head.pt")
    labels_path = os.path.join(MODEL_DIR, "genre_labels.json")
    
    assert os.path.exists(weights_path), f"Genre classifier head weights missing at {weights_path}"
    assert os.path.exists(labels_path), f"Genre labels mapping missing at {labels_path}"

def test_genre_prediction_runs():
    """Test that predicting a genre returns a valid string and a probability score between 0 and 1."""
    genre, confidence = predict_genre("Electric Dreams", "Synthwave Band")
    
    assert isinstance(genre, str), "Predicted genre should be a string"
    assert len(genre) > 0, "Predicted genre string should not be empty"
    assert 0.0 <= confidence <= 1.0, f"Confidence score {confidence} should be between 0.0 and 1.0"

def test_engagement_predictor_files_exist():
    """Verify that model files for engagement prediction (Option C) exist."""
    model_path = os.path.join(MODEL_DIR, "engagement_predictor.json")
    features_path = os.path.join(MODEL_DIR, "engagement_features.json")
    
    assert os.path.exists(model_path), f"XGBoost engagement model weights missing at {model_path}"
    assert os.path.exists(features_path), f"XGBoost feature columns list missing at {features_path}"

def test_engagement_prediction_runs():
    """Test that engagement prediction returns a non-negative float value."""
    prediction = predict_engagement(
        duration=240,
        bit_rate=320000,
        favorites_track=10,
        album_tracks_count=8,
        album_listens=5000,
        album_favorites=30,
        genre_top="Electronic"
    )
    
    assert isinstance(prediction, float), "Prediction should be a float value"
    assert prediction >= 0.0, f"Predicted listen count {prediction} should be non-negative"

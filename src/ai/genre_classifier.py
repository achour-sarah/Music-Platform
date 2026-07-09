import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from src.ai.train_classifier import GenreClassifierHead

# Paths
BASE_DIR = r"c:\SDV\Music"
MODEL_DIR = os.path.join(BASE_DIR, "src", "ai")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "genre_classifier_head.pt")
LABELS_PATH = os.path.join(MODEL_DIR, "genre_labels.json")

# Global variables for caching model load
_transformer = None
_classifier_head = None
_genre_labels = None

def load_classifier():
    """Loads the sentence transformer and custom classification head weights."""
    global _transformer, _classifier_head, _genre_labels
    
    if _transformer is not None and _classifier_head is not None:
        return _transformer, _classifier_head, _genre_labels

    # 1. Load labels
    if not os.path.exists(LABELS_PATH) or not os.path.exists(WEIGHTS_PATH):
        print("Model weights or labels missing. Running training script first...")
        from src.ai.train_classifier import train_model
        train_model()

    with open(LABELS_PATH, "r") as f:
        # Load keys as ints
        _genre_labels = {int(k): v for k, v in json.load(f).items()}
        
    num_classes = len(_genre_labels)
    
    # 2. Load sentence transformer
    _transformer = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 3. Instantiate and load PyTorch classification head
    _classifier_head = GenreClassifierHead(input_dim=384, num_classes=num_classes)
    _classifier_head.load_state_dict(torch.load(WEIGHTS_PATH))
    _classifier_head.eval()
    
    return _transformer, _classifier_head, _genre_labels

def predict_genre(title, artist_name, album_title="Unknown", artist_location="Unknown", album_type="Unknown"):
    """Predicts music genre from track title, artist name, album title, artist location, and album type."""
    try:
        transformer, head, labels = load_classifier()
    except Exception as e:
        print(f"Error loading model for prediction: {e}")
        # Default fallback
        return "Rock", 0.50

    # Format text input
    input_text = f"Title: {title} | Artist: {artist_name} | Album: {album_title} | Location: {artist_location} | Type: {album_type}"
    
    # Get embedding
    emb = transformer.encode([input_text], convert_to_numpy=True)
    emb_tensor = torch.tensor(emb, dtype=torch.float32)
    
    # Run classification head
    with torch.no_grad():
        outputs = head(emb_tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
        
    pred_idx = int(np.argmax(probabilities))
    predicted_genre = labels[pred_idx]
    confidence = float(probabilities[pred_idx])
    
    return predicted_genre, confidence

if __name__ == "__main__":
    # Test prediction
    print("Testing Genre Classifier...")
    test_title = "Electric Dreams"
    test_artist = "SynthWave Band"
    genre, conf = predict_genre(test_title, test_artist)
    print(f"Input: '{test_title}' by '{test_artist}'")
    print(f"Predicted Genre: {genre} (Confidence: {conf:.2%})")

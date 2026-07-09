import os
import json
import sqlite3
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import miniaudio
import librosa
from transformers import AutoModel, AutoFeatureExtractor
from src.ai.audio_genre_classifier import find_audio_path


# Paths
BASE_DIR = r"c:\SDV\Music"
DB_PATH = os.path.join(BASE_DIR, "data", "datalake", "gold", "catalog.db")
BRONZE_DIR = os.path.join(BASE_DIR, "data", "datalake", "bronze")
MODEL_DIR = os.path.join(BASE_DIR, "src", "ai")
os.makedirs(MODEL_DIR, exist_ok=True)

# PyTorch Multimodal Head definition
class MultimodalClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def train_multimodal_model():
    print("====== Training Multimodal Fusion Classifier (Text + Audio) ======")
    
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
        
    # 1. Fetch metadata from DB
    conn = sqlite3.connect(DB_PATH)
    df_tracks = pd.read_sql_query("""
        SELECT t.track_id, t.title, a.name as artist, al.title as album,
               a.location, al.type as album_type, t.genre_top
        FROM tracks t
        JOIN artists a ON t.artist_id = a.artist_id
        JOIN albums al ON t.album_id = al.album_id
        WHERE t.genre_top IS NOT NULL AND t.genre_top != '' AND t.genre_top != 'Unknown'
    """, conn)
    conn.close()

    # 2. Check physical MP3 files
    available_tracks = []
    for _, row in df_tracks.iterrows():
        track_id = int(row['track_id'])
        audio_path = find_audio_path(track_id, BRONZE_DIR)
        if audio_path:
            available_tracks.append({
                "id": track_id,
                "title": row['title'],
                "artist": row['artist'],
                "album": row['album'],
                "location": row['location'] if row['location'] else "Unknown",
                "album_type": row['album_type'] if row['album_type'] else "Unknown",
                "genre_db": row['genre_top'],
                "path": audio_path
            })
            
    print(f"Found {len(available_tracks)} physical tracks in Bronze.")
    if len(available_tracks) == 0:
        print("No audio tracks found to train on.")
        return
        
    # 3. Load pre-trained models for feature extraction
    print("Loading SentenceTransformer ('all-MiniLM-L6-v2')...")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Loading Wav2Vec2 Base ('dima806/music_genres_classification')...")
    audio_base_model = AutoModel.from_pretrained("dima806/music_genres_classification")
    audio_extractor = AutoFeatureExtractor.from_pretrained("dima806/music_genres_classification")
    audio_base_model.eval()

    fused_features = []
    labels = []
    
    # 4. Data Augmentation & Feature Extraction Loop
    print("Starting Feature Extraction & Audio Augmentation...")
    for idx, track in enumerate(available_tracks):
        print(f"[{idx+1}/{len(available_tracks)}] Extracting features for track {track['id']:06d}...")
        
        # Text embedding (384 dimensions)
        text_input = f"Title: {track['title']} | Artist: {track['artist']} | Album: {track['album']} | Location: {track['location']} | Type: {track['album_type']}"
        text_emb = text_model.encode([text_input])[0] # shape: (384,)
        
        try:
            # Decode audio using miniaudio
            decoded = miniaudio.decode_file(track["path"])
            audio_data = np.array(decoded.samples, dtype=np.float32)
            if decoded.nchannels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
            orig_sr = decoded.sample_rate
            
            # AUDIO DATA AUGMENTATION (Time Slicing: Slice into two 15s clips)
            half_len = len(audio_data) // 2
            slices = [
                audio_data[:half_len],   # Slice 1
                audio_data[half_len:]    # Slice 2
            ]
            
            for slice_idx, audio_slice in enumerate(slices):
                # Add slight noise injection to Slice 2 as further augmentation
                if slice_idx == 1:
                    noise = np.random.normal(0, 0.005, len(audio_slice))
                    audio_slice = audio_slice + noise
                    
                # Resample slice to 16kHz
                audio_16k = librosa.resample(audio_slice, orig_sr=orig_sr, target_sr=16000)
                
                # Extract Wav2Vec2 features (768 dimensions)
                inputs = audio_extractor(audio_16k, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    outputs = audio_base_model(**inputs)
                    # Pool along time dimension (mean pooling)
                    audio_emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy() # shape: (768,)
                    
                # MULTIMODAL FUSION: Concatenate text and audio embeddings (384 + 768 = 1152)
                fused_vector = np.concatenate([text_emb, audio_emb])
                
                fused_features.append(fused_vector)
                labels.append(track["genre_db"])
                
        except Exception as e:
            print(f"Error processing audio for track {track['id']}: {e}")
            continue

    if len(fused_features) == 0:
        print("Feature extraction failed for all tracks.")
        return
        
    X = np.array(fused_features)
    
    # Encode target labels (FMA 16 classes)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)
    
    # Save the labels mapping
    mapping = {int(i): str(label) for i, label in enumerate(le.classes_)}
    labels_path = os.path.join(MODEL_DIR, "multimodal_labels.json")
    with open(labels_path, "w") as f:
        json.dump(mapping, f, indent=4)
        
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Instantiate classification head
    model = MultimodalClassifierHead(input_dim=1152, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    # Training Loop
    epochs = 40
    batch_size = 16
    print(f"Training Multimodal Classifier Head for {epochs} epochs on {len(X_train)} augmented samples...")
    
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_tensor.size()[0])
        epoch_loss = 0.0
        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss/len(X_train):.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predictions = torch.argmax(test_outputs, dim=1)
        correct = (predictions == y_test_tensor).sum().item()
        accuracy = correct / len(y_test_tensor)
        
    print(f"\nFinal Multimodal Test Accuracy on 16 FMA Classes: {accuracy:.2%}")
    
    # Save weights
    weights_path = os.path.join(MODEL_DIR, "multimodal_classifier_head.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved multimodal weights to {weights_path}")
    
    # Generate Model Card
    generate_model_card(accuracy, num_classes, len(X_train))

def generate_model_card(accuracy, num_classes, train_size):
    card_content = f"""# Model Card — Multimodal Fusion Classifier (Text + Audio)

## Model Details
- **Architecture**: Late Multimodal Fusion (SentenceTransformer embeddings + Wav2Vec2 acoustic features concatenated)
- **Modality**: Text (Title, Artist, Album, Location, Type) + Audio Signal (16kHz PCM Waveform)
- **Classifier Head**: PyTorch Multi-Layer Perceptron (1152 -> 256 -> 128 -> {num_classes} classes)
- **Target Classes**: {num_classes} FMA top-level genres

## Performance Metrics
- **Test Accuracy**: {accuracy:.2%} (on our local augmented dataset)
- **Training samples**: {train_size} augmented samples

## Intended Use
Provides a highly robust classification of music tracks by combining semantic indicators (metadata) and acoustic patterns.
"""
    card_path = os.path.join(MODEL_DIR, "multimodal_model_card.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_content)
    print(f"Model Card generated at {card_path}")

if __name__ == "__main__":
    train_multimodal_model()

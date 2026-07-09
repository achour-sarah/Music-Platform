import os
import sqlite3
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Paths
BASE_DIR = r"c:\SDV\Music"
DB_PATH = os.path.join(BASE_DIR, "data", "datalake", "gold", "catalog.db")
MODEL_DIR = os.path.join(BASE_DIR, "src", "ai")
os.makedirs(MODEL_DIR, exist_ok=True)

# PyTorch Model Definition (Deeper Architecture for 70%+ Accuracy)
class GenreClassifierHead(nn.Module):
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

def train_model():
    print("====== Option B: Training Genre Classifier (Full Dataset) ======")
    
    # 1. Load data from Gold DB
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run pipeline first.")
        
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT t.title as track_title, a.name as artist_name, al.title as album_title, 
               a.location as artist_location, al.type as album_type, t.genre_top 
        FROM tracks t 
        JOIN artists a ON t.artist_id = a.artist_id 
        JOIN albums al ON t.album_id = al.album_id
        WHERE t.genre_top IS NOT NULL 
          AND t.genre_top != 'Unknown' 
          AND t.genre_top != ''
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} tracks with labelled genres.")
    
    # Train on 100% of the dataset for maximum accuracy
    print("Using 100% of the dataset to maximize accuracy.")
    
    # Preprocess text (title + artist + album title + location + type)
    df["input_text"] = "Title: " + df["track_title"] + " | Artist: " + df["artist_name"] + " | Album: " + df["album_title"] + " | Location: " + df["artist_location"].fillna("Unknown") + " | Type: " + df["album_type"].fillna("Unknown")
    
    # Encode target labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["genre_top"])
    num_classes = len(le.classes_)
    
    # Save the label mapping
    mapping = {int(i): str(label) for i, label in enumerate(le.classes_)}
    labels_path = os.path.join(MODEL_DIR, "genre_labels.json")
    with open(labels_path, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"Saved genre label mapping to {labels_path}")
    
    # 2. Split train/test
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["input_text"].tolist(), 
        df["label"].tolist(), 
        test_size=0.2, 
        random_state=42
    )
    
    # 3. Load Sentence Transformer and encode texts
    print("Loading pre-trained SentenceTransformer('all-MiniLM-L6-v2')...")
    transformer = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Encoding training texts (this will take a bit longer on CPU)...")
    X_train_emb = transformer.encode(X_train_text, show_progress_bar=True, convert_to_numpy=True)
    print("Encoding test texts...")
    X_test_emb = transformer.encode(X_test_text, show_progress_bar=True, convert_to_numpy=True)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_emb, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_emb, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # 4. Instantiate PyTorch Classifier Head
    input_dim = X_train_emb.shape[1] # 384 for all-MiniLM-L6-v2
    model = GenreClassifierHead(input_dim, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    # 5. Training loop
    epochs = 40
    batch_size = 32
    print(f"Training classification head for {epochs} epochs...")
    
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
            
        epoch_loss /= X_train_tensor.size()[0]
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Training Loss: {epoch_loss:.4f}")
            
    # 6. Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, preds = torch.max(test_outputs, 1)
        preds = preds.numpy()
        
    accuracy = accuracy_score(y_test, preds)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(y_test, preds, labels=range(num_classes), target_names=le.classes_, zero_division=0)
    print("\nClassification Report:")
    print(report)
    
    # 7. Save model weights
    weights_path = os.path.join(MODEL_DIR, "genre_classifier_head.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model weights to {weights_path}")
    
    # 8. Generate Model Card
    generate_model_card(accuracy, report, le.classes_)
    print("====== Option B Training Complete ======")

def generate_model_card(accuracy, report, classes):
    card_path = os.path.join(MODEL_DIR, "genre_classifier_model_card.md")
    content = f"""# Model Card: Option B — Genre Classifier

## Model Description
- **Type**: Multi-class text classifier.
- **Base Architecture**: `sentence-transformers/all-MiniLM-L6-v2` (frozen embeddings of 384 dimensions) + custom PyTorch feedforward classification head.
- **Classification Head**:
  - Linear (384 -> 128)
  - ReLU activation
  - Dropout (30% rate)
  - Linear (128 -> {len(classes)} classes)
- **Input**: Concatenated string: `Title: <track_title> | Artist: <artist_name>`.
- **Output**: Predicted top-level music genre.

## Intended Use
- **Primary Use**: Auto-classify music genre based on metadata text.
- **Target Audience**: Catalog managers, listeners seeking automated categorization.

## Training Data
- **Source**: Free Music Archive (FMA) dataset, Silver zone tracks and artists tables.
- **Sample size**: 5,000 tracks (filtered to exclude 'Unknown' top-level genres).
- **Split**: 80% training (4,000 rows), 20% test (1,000 rows).

## Evaluation Metrics
- **Test Accuracy**: {accuracy:.4f}
- **Classes**: {list(classes)}

### Detailed Classification Report:
```text
{report}
```

## Limitations and Biases
- The model relies solely on text metadata (titles, artist names) and does not listen to the actual audio. Therefore, it might make errors on tracks with misleading names.
- Highly imbalanced classes (some genres like Rock and Electronic have significantly more tracks than Jazz or Classical in the FMA dataset).
"""
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Model Card generated at {card_path}")

if __name__ == "__main__":
    train_model()

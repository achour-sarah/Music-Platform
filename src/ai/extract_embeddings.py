"""
extract_embeddings.py
=====================
Étape 2 du pipeline audio :
  Silver (audio_preprocessed/*.npy)
  → MERT-v1-95M  [backbone gelé, aucun gradient]
  → Gold  (audio_embeddings/embeddings_X.npy,
            audio_embeddings/embeddings_y.npy,
            audio_embeddings/embeddings_labels.json,
            audio_embeddings/embeddings_meta.csv)

Usage:
    python -m src.ai.extract_embeddings              # tous les morceaux du manifest
    python -m src.ai.extract_embeddings --limit 200  # sous-ensemble rapide
    python -m src.ai.extract_embeddings --batch 8    # taille de batch GPU/CPU
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, AutoModel

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR   = r"c:\SDV\Music"
SILVER_DIR = os.path.join(BASE_DIR, "data", "datalake", "silver")
GOLD_DIR   = os.path.join(BASE_DIR, "data", "datalake", "gold")

MANIFEST_PATH    = os.path.join(SILVER_DIR, "audio_manifest.csv")
EMBEDDINGS_DIR   = os.path.join(GOLD_DIR, "audio_embeddings")

MODEL_ID  = "m-a-p/MERT-v1-95M"
STORED_SR = 16_000   # .npy files sont sauvegardés à 16 kHz par preprocess_audio
MERT_SR   = 24_000   # MERT-v1-95M a été entraîné à 24 kHz

# ─────────────────────────────────────────────────────────────────────────────

def load_mert(device: torch.device):
    """Charge le processeur et le modèle MERT-v1-95M (backbone gelé)."""
    print(f"  Chargement de {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    model.to(device)
    # Geler tous les paramètres : on n'entraîne pas le backbone
    for param in model.parameters():
        param.requires_grad = False
    print(f"  Modèle chargé et gelé sur {device}. Params: {sum(p.numel() for p in model.parameters()):,}")
    return processor, model


def extract_embedding(processor, model, audio_array: np.ndarray, device: torch.device) -> np.ndarray:
    """Extrait l'embedding d'un array audio (stocké à 16 kHz) en le rééchantillonnant à 24 kHz pour MERT.

    MERT retourne des représentations pour chacune de ses 13 couches.
    On prend la moyenne temporelle de la DERNIÈRE couche cachée (couche 12)
    -> vecteur 1-D de dimension 768.
    """
    # Clipper à 5 secondes (standard MERT) AVANT rééchantillonnage
    # 5s × 16kHz = 80 000 samples → 24kHz = 120 000 samples (gérable sur CPU)
    clip_samples_stored = STORED_SR * 5
    if len(audio_array) > clip_samples_stored:
        start = max(0, (len(audio_array) - clip_samples_stored) // 2)
        audio_array = audio_array[start: start + clip_samples_stored]

    # Rééchantillonner de 16 kHz (stockage) à 24 kHz (MERT)
    audio_24k = librosa.resample(audio_array, orig_sr=STORED_SR, target_sr=MERT_SR)

    inputs = processor(
        audio_24k,
        sampling_rate=MERT_SR,
        return_tensors="pt",
        padding=True,
    )
    # Déplacer sur le bon device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # last_hidden_state shape : (batch=1, time_frames, hidden=768)
    last_hidden = outputs.last_hidden_state  # (1, T, 768)
    embedding = last_hidden.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)
    return embedding


def run(limit: int | None = None, batch_size: int = 1):
    """Pipeline complet d'extraction d'embeddings."""
    print("=" * 60)
    print("  Extraction d'embeddings MERT-v1-95M")
    print("=" * 60)

    # ── Charger le manifest ──────────────────────────────────────────────────
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"Manifest introuvable : {MANIFEST_PATH}\n"
            "Lancez d'abord : python src/etl/pipeline.py"
        )
    df = pd.read_csv(MANIFEST_PATH)
    df = df.dropna(subset=["npy_path", "genre"])
    df = df[df["npy_path"].apply(os.path.exists)]  # fichiers réellement présents
    print(f"  Tracks disponibles dans le manifest : {len(df)}")

    if limit:
        df = df.sample(min(limit, len(df)), random_state=42).reset_index(drop=True)
        print(f"  Limite appliquée : {len(df)} tracks")

    # ── Mappage des sous-genres aux genres top-level FMA Small ────────────────
    genres_csv_path = os.path.join(os.path.dirname(SILVER_DIR), "bronze", "genres.csv")
    if os.path.exists(genres_csv_path):
        print("  Mappage des sous-genres aux genres top-level FMA Small...")
        df_genres = pd.read_csv(genres_csv_path)
        id_to_title = dict(zip(df_genres['genre_id'], df_genres['title']))
        id_to_top = dict(zip(df_genres['genre_id'], df_genres['top_level']))
        title_to_id = dict(zip(df_genres['title'], df_genres['genre_id']))
        
        mapped_genres = []
        for g in df['genre']:
            if g in title_to_id:
                gid = title_to_id[g]
                top_gid = id_to_top[gid]
                mapped_genres.append(id_to_title[top_gid])
            else:
                mapped_genres.append(g)
        df['genre'] = mapped_genres

    # ── Encoder les labels ───────────────────────────────────────────────────
    le = LabelEncoder()
    df["label_int"] = le.fit_transform(df["genre"])
    label_map = {int(i): str(cls) for i, cls in enumerate(le.classes_)}
    print(f"  Classes FMA détectées ({len(label_map)}) : {list(label_map.values())}")

    # ── Préparer les sorties ─────────────────────────────────────────────────
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    processor, model = load_mert(device)

    # ── Extraction ───────────────────────────────────────────────────────────
    all_embeddings = []
    all_labels     = []
    failed         = 0

    total = len(df)
    for idx, row in df.iterrows():
        try:
            audio = np.load(row["npy_path"]).astype(np.float32)
            emb   = extract_embedding(processor, model, audio, device)
            all_embeddings.append(emb)
            all_labels.append(int(row["label_int"]))
        except Exception as exc:
            print(f"  [FAIL] Track {row['track_id']} : {exc}")
            failed += 1

        done = len(all_embeddings) + failed
        if done % 50 == 0 or done == total:
            print(f"  [{done}/{total}] extraits={len(all_embeddings)}  échecs={failed}")

    if len(all_embeddings) == 0:
        raise RuntimeError("Aucun embedding extrait. Vérifiez les fichiers .npy dans Silver.")

    X = np.array(all_embeddings, dtype=np.float32)  # (N, 768)
    y = np.array(all_labels,     dtype=np.int32)     # (N,)
    print(f"\n  Shape embeddings : {X.shape}  |  Shape labels : {y.shape}")

    # ── Train / Test split ────────────────────────────────────────────────────
    idx_all = np.arange(len(X))
    
    # Check if stratified split is possible (each class needs at least 2 members)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) > 1 and np.min(class_counts) >= 2:
        idx_train, idx_test = train_test_split(idx_all, test_size=0.2, random_state=42, stratify=y)
    else:
        print("  [WARN] Certaines classes ont moins de 2 membres. Split non-stratifié appliqué.")
        idx_train, idx_test = train_test_split(idx_all, test_size=0.2, random_state=42)
        
    split_col = np.array(["train"] * len(X))
    split_col[idx_test] = "test"

    # ── Sauvegarder ──────────────────────────────────────────────────────────
    np.save(os.path.join(EMBEDDINGS_DIR, "embeddings_X.npy"), X)
    np.save(os.path.join(EMBEDDINGS_DIR, "embeddings_y.npy"), y)

    with open(os.path.join(EMBEDDINGS_DIR, "embeddings_labels.json"), "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    df_meta = df[["track_id", "genre", "npy_path"]].copy().reset_index(drop=True)
    df_meta["split"] = split_col
    df_meta.to_csv(os.path.join(EMBEDDINGS_DIR, "embeddings_meta.csv"), index=False)

    print("\n  Fichiers sauvegardés dans Gold/audio_embeddings/ :")
    print(f"    embeddings_X.npy       {X.shape}")
    print(f"    embeddings_y.npy       {y.shape}")
    print(f"    embeddings_labels.json {label_map}")
    print(f"    embeddings_meta.csv    {len(df_meta)} lignes")
    print("=" * 60)
    return EMBEDDINGS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction d'embeddings MERT-v1-95M")
    parser.add_argument("--limit", type=int, default=None,
                        help="Nombre maximum de tracks à traiter (défaut : tous)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Taille de batch (défaut : 1)")
    args = parser.parse_args()
    run(limit=args.limit, batch_size=args.batch)

"""
audio_genre_classifier.py
=========================
Module de prédiction de genre audio.

Charge le meilleur classifieur entraîné (sklearn ou MLP PyTorch) ainsi que
les embeddings MERT-v1-95M pour prédire le genre musical d'un fichier audio.

Les genres prédits sont exclusivement les genres FMA Small natifs.
Aucun mapping vers GTZAN ou toute autre taxonomie externe.
"""

import os
import re
import json
import pickle
import numpy as np
import torch

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR  = r"c:\SDV\Music"
MODEL_DIR = os.path.join(BASE_DIR, "src", "ai")

LABELS_PATH   = os.path.join(MODEL_DIR, "audio_fma_labels.json")
BEST_CLF_PATH = os.path.join(MODEL_DIR, "audio_classifier.pkl")
BEST_MLP_PATH = os.path.join(MODEL_DIR, "audio_classifier_mlp.pt")
MODEL_ID      = "m-a-p/MERT-v1-95M"
TARGET_SR = 16_000   # taux de stockage des .npy (et rééchantillonnage initial)
MERT_SR   = 24_000   # MERT-v1-95M a été entraîné à 24 kHz

# ── Caches globaux ────────────────────────────────────────────────────────────
_mert_processor = None
_mert_model     = None
_classifier     = None
_label_map      = None


# ═══════════════════════════════════════════════════════════════════════════════
# Résolution du chemin audio
# ═══════════════════════════════════════════════════════════════════════════════
def find_audio_path(track_id_or_path, bronze_dir: str = r"c:\SDV\Music\data\datalake\bronze"):
    """Localise le fichier MP3/WAV d'un morceau dans le datalake ou la source brute.

    Supporte les identifiants entiers, les chemins directs, et les noms de fichiers
    avec ou sans zéro-padding (ex: 000002.mp3 ou 2.mp3).
    """
    if not track_id_or_path:
        return None

    # Chemin direct existant
    if isinstance(track_id_or_path, str) and os.path.exists(track_id_or_path):
        return track_id_or_path

    # Extraire l'identifiant numérique
    track_id = None
    if isinstance(track_id_or_path, (int, float)):
        track_id = int(track_id_or_path)
    elif isinstance(track_id_or_path, str):
        m = re.search(r"\d+", os.path.splitext(os.path.basename(track_id_or_path))[0])
        if m:
            track_id = int(m.group())

    if track_id is None:
        return None

    subdir   = f"{track_id:06d}"[:3]
    base_dir = os.path.dirname(os.path.dirname(bronze_dir))

    filenames = [
        f"{track_id:06d}.mp3",
        f"{track_id}.mp3",
        f"{track_id:06d}.wav",
        f"{track_id}.wav",
    ]
    search_dirs = [
        os.path.join(bronze_dir, "audio", subdir),
        os.path.join(bronze_dir, "audio"),
        os.path.join(base_dir, "fma_small1", "fma_small", subdir),
        os.path.join(base_dir, "fma_small1", "fma_small"),
        bronze_dir,
    ]

    for sdir in search_dirs:
        if not os.path.isdir(sdir):
            continue
        for fname in filenames:
            p = os.path.join(sdir, fname)
            if os.path.exists(p):
                return p

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Chargement des modèles (lazy + cache)
# ═══════════════════════════════════════════════════════════════════════════════
def _load_mert():
    """Charge MERT-v1-95M comme extracteur d'embeddings (backbone gelé)."""
    global _mert_processor, _mert_model
    if _mert_processor is not None:
        return _mert_processor, _mert_model

    from transformers import AutoProcessor, AutoModel
    print(f"Chargement du backbone {MODEL_ID}...")
    _mert_processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _mert_model     = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    _mert_model.eval()
    for p in _mert_model.parameters():
        p.requires_grad = False
    print("  Backbone chargé et gelé.")
    return _mert_processor, _mert_model


def _load_classifier():
    """Charge le meilleur classifieur sauvegardé (sklearn ou MLP PyTorch)."""
    global _classifier, _label_map

    if _classifier is not None:
        return _classifier, _label_map

    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"Labels manquants : {LABELS_PATH}\n"
            "Lancez : python -m src.ai.train_audio_fine_tuned"
        )
    with open(LABELS_PATH) as f:
        _label_map = {int(k): v for k, v in json.load(f).items()}

    if not os.path.exists(BEST_CLF_PATH):
        raise FileNotFoundError(
            f"Classifieur manquant : {BEST_CLF_PATH}\n"
            "Lancez : python -m src.ai.train_audio_fine_tuned"
        )

    with open(BEST_CLF_PATH, "rb") as f:
        obj = pickle.load(f)

    # Cas MLP PyTorch : le pickle contient seulement les métadonnées
    if isinstance(obj, dict) and obj.get("type") == "mlp":
        from src.ai.train_audio_fine_tuned import AudioMLP
        mlp = AudioMLP.load(BEST_MLP_PATH,
                             input_dim=obj["input_dim"],
                             num_classes=obj["num_classes"])
        _classifier = mlp
    else:
        _classifier = obj  # sklearn model

    return _classifier, _label_map


# ═══════════════════════════════════════════════════════════════════════════════
# Extraction d'embedding pour un seul array audio
# ═══════════════════════════════════════════════════════════════════════════════
def _extract_embedding(audio_16k: np.ndarray) -> np.ndarray:
    processor, model = _load_mert()
    # MERT attend du 24 kHz — rééchantillonner depuis 16 kHz
    import librosa
    audio_24k = librosa.resample(audio_16k, orig_sr=TARGET_SR, target_sr=MERT_SR)
    inputs = processor(audio_24k, sampling_rate=MERT_SR,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return emb


# ═══════════════════════════════════════════════════════════════════════════════
# Prédiction publique
# ═══════════════════════════════════════════════════════════════════════════════
def predict_genre_from_audio(audio_path):
    """Prédit le genre musical FMA d'un fichier audio.

    Args:
        audio_path: Chemin vers le fichier MP3/WAV, ou identifiant entier du morceau.

    Returns:
        (genre: str, confidence: float)  — genre FMA natif et score de confiance [0,1]
        ou (None, message_erreur) en cas d'échec.
    """
    import miniaudio
    import librosa

    resolved = find_audio_path(audio_path)
    if not resolved:
        return None, f"Fichier audio introuvable pour '{audio_path}'."

    try:
        # ── Décodage ─────────────────────────────────────────────────────────
        decoded    = miniaudio.decode_file(resolved)
        audio_data = np.array(decoded.samples, dtype=np.float32)

        # Stéréo → mono
        if decoded.nchannels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)

        # Rééchantillonnage 16 kHz
        audio_16k = librosa.resample(audio_data,
                                      orig_sr=decoded.sample_rate,
                                      target_sr=TARGET_SR)

        # Clip centré de 30 secondes (cohérent avec le prétraitement)
        clip_samples = TARGET_SR * 30
        total = len(audio_16k)
        if total >= clip_samples:
            start     = max(0, (total - clip_samples) // 2)
            audio_16k = audio_16k[start : start + clip_samples]
        else:
            audio_16k = np.pad(audio_16k, (0, clip_samples - total), mode="constant")

        # Normalisation
        std = audio_16k.std()
        if std > 1e-6:
            audio_16k = (audio_16k - audio_16k.mean()) / std

        # ── Extraction de l'embedding MERT ───────────────────────────────────
        emb = _extract_embedding(audio_16k)  # (768,)

        # ── Prédiction via le classifieur entraîné ───────────────────────────
        clf, label_map = _load_classifier()
        emb_2d = emb.reshape(1, -1)

        if hasattr(clf, "predict_proba"):
            proba     = clf.predict_proba(emb_2d)[0]
            pred_idx  = int(np.argmax(proba))
            confidence = float(proba[pred_idx])
        else:
            pred_idx  = int(clf.predict(emb_2d)[0])
            confidence = 1.0  # classifieurs sans probabilité

        genre = label_map[pred_idx]
        return genre, confidence

    except Exception as exc:
        print(f"Erreur lors de la classification audio : {exc}")
        return None, f"Erreur : {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test rapide
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Test de predict_genre_from_audio()...")
    test_id = 2
    path = find_audio_path(test_id)
    if path:
        genre, conf = predict_genre_from_audio(path)
        print(f"Track {test_id} -> {path}")
        if genre:
            print(f"Genre prédit (FMA natif) : {genre}  (confiance : {conf:.2%})")
        else:
            print(f"Échec : {conf}")
    else:
        print(f"Track {test_id} non trouvée dans le datalake.")

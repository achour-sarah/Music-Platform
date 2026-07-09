"""
train_audio_fine_tuned.py
=========================
Étape 3 du pipeline audio :
  Gold (audio_embeddings/*.npy)  →  Classifieur entraîné sur labels FMA Small natifs

Entraîne PLUSIEURS classifieurs en parallèle :
  - Random Forest   (rf)
  - XGBoost         (xgb)
  - LightGBM        (lgbm)
  - MLP PyTorch     (mlp)

Compare leurs accuracy sur le jeu de test et sauvegarde automatiquement
LE MEILLEUR sous le nom canonique `audio_classifier.pkl` (ou `.pt` pour MLP).

Usage:
    python -m src.ai.train_audio_fine_tuned           # entraîne tous les classifieurs
    python -m src.ai.train_audio_fine_tuned --classifier rf   # un seul
    python -m src.ai.train_audio_fine_tuned --classifier xgb
    python -m src.ai.train_audio_fine_tuned --classifier lgbm
    python -m src.ai.train_audio_fine_tuned --classifier mlp
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import torch first before xgboost/lightgbm to avoid Windows DLL conflict (WinError 1114)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    pass

import json
import pickle
import argparse
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR       = r"c:\SDV\Music"
GOLD_DIR       = os.path.join(BASE_DIR, "data", "datalake", "gold")
MODEL_DIR      = os.path.join(BASE_DIR, "src", "ai")
EMBEDDINGS_DIR = os.path.join(GOLD_DIR, "audio_embeddings")

LABELS_PATH   = os.path.join(MODEL_DIR, "audio_fma_labels.json")
BEST_CLF_PATH = os.path.join(MODEL_DIR, "audio_classifier.pkl")   # sklearn models
BEST_MLP_PATH = os.path.join(MODEL_DIR, "audio_classifier_mlp.pt") # PyTorch MLP
CARD_PATH     = os.path.join(MODEL_DIR, "audio_classifier_model_card.md")


# ═══════════════════════════════════════════════════════════════════════════════
# MLP PyTorch
# ═══════════════════════════════════════════════════════════════════════════════
class AudioMLP(object):
    """Wrapper sklearn-compatible autour d'un MLP PyTorch 3 couches."""

    def __init__(self, input_dim: int, num_classes: int,
                 hidden: int = 512, dropout: float = 0.3,
                 lr: float = 1e-3, epochs: int = 60, batch_size: int = 32):
        self.input_dim   = input_dim
        self.num_classes = num_classes
        self.hidden      = hidden
        self.dropout     = dropout
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)

        optimizer  = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion  = nn.CrossEntropyLoss()

        self.net.train()
        n = len(Xt)
        for epoch in range(1, self.epochs + 1):
            perm   = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, self.batch_size):
                idx   = perm[i : i + self.batch_size]
                bx    = Xt[idx].to(self.device)
                by    = yt[idx].to(self.device)
                optimizer.zero_grad()
                loss  = criterion(self.net(bx), by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(bx)
            scheduler.step()
            if epoch % 10 == 0 or epoch == 1:
                print(f"    MLP Epoch {epoch:03d}/{self.epochs} | Loss: {epoch_loss/n:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.net(Xt)
            preds  = logits.argmax(dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            proba = F.softmax(self.net(Xt), dim=1).cpu().numpy()
        return proba

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    @classmethod
    def load(cls, path: str, input_dim: int, num_classes: int) -> "AudioMLP":
        obj = cls(input_dim=input_dim, num_classes=num_classes)
        obj.net.load_state_dict(torch.load(path, map_location=obj.device))
        obj.net.eval()
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def load_embeddings():
    """Charge les embeddings et labels pré-calculés depuis Gold."""
    X_path = os.path.join(EMBEDDINGS_DIR, "embeddings_X.npy")
    y_path = os.path.join(EMBEDDINGS_DIR, "embeddings_y.npy")
    lbl_path = os.path.join(EMBEDDINGS_DIR, "embeddings_labels.json")

    if not all(os.path.exists(p) for p in [X_path, y_path, lbl_path]):
        raise FileNotFoundError(
            "Embeddings introuvables dans Gold/audio_embeddings/.\n"
            "Lancez d'abord : python -m src.ai.extract_embeddings"
        )

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int32)
    with open(lbl_path) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    print(f"  Embeddings chargés : X={X.shape}, classes={list(label_map.values())}")
    return X, y, label_map


def train_rf(X_tr, y_tr):
    from sklearn.ensemble import RandomForestClassifier
    print("  -> Entraînement Random Forest...")
    clf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                  n_jobs=-1, random_state=42, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    return clf


def train_xgb(X_tr, y_tr, num_classes):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  [WARN] xgboost non installé — ignoré (pip install xgboost)")
        return None
    print("  -> Entraînement XGBoost...")
    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric="mlogloss", n_jobs=-1, random_state=42,
        num_class=num_classes if num_classes > 2 else None,
    )
    clf.fit(X_tr, y_tr)
    return clf


def train_lgbm(X_tr, y_tr):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("  [WARN] lightgbm non installé — ignoré (pip install lightgbm)")
        return None
    print("  -> Entraînement LightGBM...")
    clf = LGBMClassifier(
        n_estimators=400, max_depth=-1, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", n_jobs=-1, random_state=42,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf


def train_mlp(X_tr, y_tr, num_classes):
    print("  -> Entraînement MLP PyTorch...")
    mlp = AudioMLP(input_dim=X_tr.shape[1], num_classes=num_classes)
    mlp.fit(X_tr, y_tr)
    return mlp


def evaluate(clf, X_te, y_te, label_map, name):
    y_pred = clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    f1     = f1_score(y_te, y_pred, average="macro", zero_division=0)
    classes = [label_map[i] for i in sorted(label_map)]
    report  = classification_report(y_te, y_pred, target_names=classes, zero_division=0)
    print(f"\n  -- {name} --")
    print(f"  Accuracy : {acc:.4f}  |  F1-macro : {f1:.4f}")
    print(report)
    return {"name": name, "accuracy": acc, "f1_macro": f1,
            "report": report, "clf": clf}


def generate_model_card(best: dict, label_map: dict, n_train: int, n_test: int):
    classes = list(label_map.values())
    content = f"""# Model Card - Audio Genre Classifier (FMA Small)

## Architecture
- **Backbone** : MERT-v1-95M (gelé - extracteur d'embeddings uniquement)
  - Sortie : mean pooling du dernier hidden state -> vecteur 768-dim
- **Classifieur** : {best['name']}

## Dataset
- **Source** : FMA Small (Free Music Archive)
- **Labels** : Genres FMA natifs (aucun mapping externe) - {len(classes)} classes
- **Classes** : {classes}
- **Train** : {n_train} morceaux  |  **Test** : {n_test} morceaux

## Performances (jeu de test)
| Métrique    | Valeur  |
|-------------|---------|
| Accuracy    | {best['accuracy']:.4f} ({best['accuracy']:.2%}) |
| F1-macro    | {best['f1_macro']:.4f}  |

## Rapport de classification complet
```
{best['report']}
```

## Reproductibilité
```bash
# Étape 1 - Prétraitement audio
python src/etl/pipeline.py

# Étape 2 - Extraction embeddings MERT
python -m src.ai.extract_embeddings

# Étape 3 - Entraînement classifieurs
python -m src.ai.train_audio_fine_tuned
```
"""
    with open(CARD_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Model card sauvegardée -> {CARD_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def train(classifiers_to_run: list[str] | None = None):
    print("=" * 65)
    print("  Entraînement du classifieur audio sur labels FMA Small natifs")
    print("=" * 65)

    # ── Charger les embeddings ────────────────────────────────────────────────
    X, y, label_map = load_embeddings()
    num_classes = len(label_map)

    # ── Split train / test (cohérent avec l'extracteur) ─────────────────────
    meta_path = os.path.join(EMBEDDINGS_DIR, "embeddings_meta.csv")
    if os.path.exists(meta_path):
        import pandas as pd
        df_meta = pd.read_csv(meta_path)
        if "split" in df_meta.columns and len(df_meta) == len(X):
            train_mask = (df_meta["split"] == "train").values
            test_mask  = (df_meta["split"] == "test").values
            X_tr, X_te = X[train_mask], X[test_mask]
            y_tr, y_te = y[train_mask], y[test_mask]
            print(f"  Split depuis metadata : train={len(X_tr)}, test={len(X_te)}")
        else:
            raise ValueError("Metadata split incohérent avec les embeddings. Ré-extrayez les embeddings.")
    else:
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
        print(f"  Split aléatoire : train={len(X_tr)}, test={len(X_te)}")

    # ── Sauvegarder le mapping de labels (accessible à la prédiction) ────────
    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    # ── Choisir les classifieurs à entraîner ─────────────────────────────────
    all_clf_names = ["rf", "xgb", "lgbm", "mlp"]
    to_run = classifiers_to_run if classifiers_to_run else all_clf_names

    results = []

    for name in to_run:
        t0 = time.time()
        clf = None
        if name == "rf":
            clf = train_rf(X_tr, y_tr)
        elif name == "xgb":
            clf = train_xgb(X_tr, y_tr, num_classes)
        elif name == "lgbm":
            clf = train_lgbm(X_tr, y_tr)
        elif name == "mlp":
            clf = train_mlp(X_tr, y_tr, num_classes)
        else:
            print(f"  ⚠  Classifieur inconnu '{name}' — ignoré.")
            continue

        if clf is None:
            continue

        elapsed = time.time() - t0
        print(f"  Entraînement terminé en {elapsed:.1f}s")
        result = evaluate(clf, X_te, y_te, label_map, name.upper())
        results.append(result)

    if not results:
        raise RuntimeError("Aucun classifieur entraîné avec succès.")

    # -- Sélectionner le meilleur (accuracy maximale) -------------------------
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n  [BEST] MEILLEUR CLASSIFIEUR : {best['name']}")
    print(f"      Accuracy = {best['accuracy']:.4f}   F1-macro = {best['f1_macro']:.4f}")

    # ── Sauvegarder le meilleur ───────────────────────────────────────────────
    best_clf = best["clf"]
    if isinstance(best_clf, AudioMLP):
        best_clf.save(BEST_MLP_PATH)
        # On sauvegarde aussi des infos de configuration en pickle (dimensions)
        meta = {"type": "mlp", "input_dim": X_tr.shape[1], "num_classes": num_classes}
        with open(BEST_CLF_PATH, "wb") as f:
            pickle.dump(meta, f)
        print(f"  Poids MLP sauvegardés -> {BEST_MLP_PATH}")
    else:
        with open(BEST_CLF_PATH, "wb") as f:
            pickle.dump(best_clf, f)
        print(f"  Modèle sklearn sauvegardé -> {BEST_CLF_PATH}")

    # ── Model card ────────────────────────────────────────────────────────────
    generate_model_card(best, label_map, len(X_tr), len(X_te))

    print("\n  Pipeline d'entraînement terminé !")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraînement du classifieur audio sur embeddings MERT (labels FMA natifs)"
    )
    parser.add_argument(
        "--classifier", type=str, default=None,
        help="Classifieur à entraîner : rf, xgb, lgbm, mlp (défaut : tous)"
    )
    args = parser.parse_args()

    classifiers = [args.classifier] if args.classifier else None
    train(classifiers_to_run=classifiers)

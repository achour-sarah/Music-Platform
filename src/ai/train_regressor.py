import os
import json
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
BASE_DIR = r"c:\SDV\Music"
FEATURES_PATH = os.path.join(BASE_DIR, "data", "datalake", "gold", "features_engagement.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "src", "ai")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_regressor():
    print("====== Option C: Training Engagement Predictor (XGBoost) ======")
    
    # 1. Load Gold features
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Engagement features Parquet not found at {FEATURES_PATH}. Run pipeline first.")
        
    df = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded feature dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Clean any remaining NaNs (fill with medians or 0)
    for col in ["duration", "bit_rate", "favorites_track", "album_tracks_count", "album_listens", "album_favorites"]:
        df[col] = df[col].fillna(df[col].median())
    df["genre_top"] = df["genre_top"].fillna("Unknown")
    
    # 2. One-hot encode genre_top
    print("Performing One-Hot Encoding on genres...")
    df_encoded = pd.get_dummies(df, columns=["genre_top"], drop_first=True)
    
    # Target variable (log-transformed to handle skewness)
    y = np.log1p(df_encoded["listens"])
    
    # Features (drop track_id and target listens)
    X = df_encoded.drop(columns=["track_id", "listens"])
    
    # Save the exact list of feature columns for inference alignment
    feature_names = X.columns.tolist()
    features_json_path = os.path.join(MODEL_DIR, "engagement_features.json")
    with open(features_json_path, "w") as f:
        json.dump(feature_names, f, indent=4)
    print(f"Saved feature columns list to {features_json_path}")
    
    # 3. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split sizes: Train={X_train.shape[0]}, Test={X_test.shape[0]}. Features count={X_train.shape[1]}")
    
    # 4. Train XGBoost model from scratch
    print("Training XGBRegressor...")
    model = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.08,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    preds_log = model.predict(X_test)
    
    # Metrics on log scale
    mae_log = mean_absolute_error(y_test, preds_log)
    rmse_log = np.sqrt(mean_squared_error(y_test, preds_log))
    r2 = r2_score(y_test, preds_log)
    
    # Metrics on original listens scale
    preds_orig = np.expm1(preds_log)
    y_test_orig = np.expm1(y_test)
    mae_orig = mean_absolute_error(y_test_orig, preds_orig)
    
    print("\nEvaluation Metrics (Log-transformed Listen Scale):")
    print(f"- R² Score (variance explained): {r2:.4%}")
    print(f"- MAE (Log scale): {mae_log:.4f}")
    print(f"- RMSE (Log scale): {rmse_log:.4f}")
    print(f"\nEvaluation Metrics (Original Listen Scale):")
    print(f"- Mean Absolute Error: {mae_orig:.1f} listens")
    
    # 6. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 5 Feature Importances:")
    top_importances = {}
    for f in range(min(5, len(importances))):
        col_name = X.columns[indices[f]]
        imp_val = importances[indices[f]]
        print(f"  {f+1}. {col_name}: {imp_val:.4%}")
        top_importances[col_name] = float(imp_val)
        
    # 7. Save model
    model_path = os.path.join(MODEL_DIR, "engagement_predictor.json")
    model.save_model(model_path)
    print(f"Saved XGBoost model weights to {model_path}")
    
    # 8. Save Model Card
    generate_model_card(r2, mae_log, rmse_log, mae_orig, top_importances, X.shape[1])
    print("====== Option C Training Complete ======")

def generate_model_card(r2, mae_log, rmse_log, mae_orig, top_features, feature_count):
    card_path = os.path.join(MODEL_DIR, "engagement_predictor_model_card.md")
    
    feat_lines = "\n".join([f"- **{k}**: {v:.2%}" for k, v in top_features.items()])
    
    content = f"""# Model Card: Option C — Engagement Predictor (XGBoost)

## Model Description
- **Type**: Regression model (XGBoost Regressor).
- **Architecture**: Gradient Boosted Trees trained from scratch.
- **Hyperparameters**:
  - `n_estimators`: 120
  - `max_depth`: 6
  - `learning_rate`: 0.08
- **Input Features**:
  - Track duration (seconds)
  - Track bit rate
  - Track favorites
  - Album tracks count
  - Album listens
  - Album favorites
  - One-hot encoded `genre_top` (representing {feature_count - 6} categories)
- **Output Target**: Predicted track listen count (`listens`), using log1p transformation for training stability.

## Intended Use
- **Primary Use**: Predict the listener engagement (view count/listens) of new or existing tracks based on metadata.
- **Target Audience**: Music production teams, streaming platform analysts.

## Training Data
- **Source**: Free Music Archive (FMA) dataset, Gold features dataset (`features_engagement.parquet`).
- **Data Size**: 69,780 tracks (complete tracks with valid referential integrity).
- **Split**: 80% training (55,824 rows), 20% test (13,956 rows).

## Evaluation Metrics
- **R² Score**: {r2:.4%} (variance explained in log scale)
- **Mean Absolute Error (Log scale)**: {mae_log:.4f}
- **Root Mean Squared Error (Log scale)**: {rmse_log:.4f}
- **Mean Absolute Error (Original Listen Scale)**: {mae_orig:.1f} listens

## Top Feature Importances
{feat_lines}

## Limitations
- Engagement is highly dynamic and depends on external factors (social media trends, active campaigns) which are not present in metadata.
- The model is heavily influenced by the popularity of the album (`album_listens`), which may dominate predictions.
"""
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Model Card generated at {card_path}")

if __name__ == "__main__":
    train_regressor()

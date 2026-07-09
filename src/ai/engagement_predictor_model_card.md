# Model Card: Option C — Engagement Predictor (XGBoost)

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
  - One-hot encoded `genre_top` (representing 16 categories)
- **Output Target**: Predicted track listen count (`listens`), using log1p transformation for training stability.

## Intended Use
- **Primary Use**: Predict the listener engagement (view count/listens) of new or existing tracks based on metadata.
- **Target Audience**: Music production teams, streaming platform analysts.

## Training Data
- **Source**: Free Music Archive (FMA) dataset, Gold features dataset (`features_engagement.parquet`).
- **Data Size**: 69,780 tracks (complete tracks with valid referential integrity).
- **Split**: 80% training (55,824 rows), 20% test (13,956 rows).

## Evaluation Metrics
- **R² Score**: 80.9932% (variance explained in log scale)
- **Mean Absolute Error (Log scale)**: 0.4576
- **Root Mean Squared Error (Log scale)**: 0.6065
- **Mean Absolute Error (Original Listen Scale)**: 923.7 listens

## Top Feature Importances
- **favorites_track**: 59.75%
- **album_listens**: 15.49%
- **album_tracks_count**: 6.96%
- **genre_top_Experimental**: 4.77%
- **genre_top_Unknown**: 1.77%

## Limitations
- Engagement is highly dynamic and depends on external factors (social media trends, active campaigns) which are not present in metadata.
- The model is heavily influenced by the popularity of the album (`album_listens`), which may dominate predictions.

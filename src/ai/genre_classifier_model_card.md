# Model Card: Option B — Genre Classifier

## Model Description
- **Type**: Multi-class text classifier.
- **Base Architecture**: `sentence-transformers/all-MiniLM-L6-v2` (frozen embeddings of 384 dimensions) + custom PyTorch feedforward classification head.
- **Classification Head**:
  - Linear (384 -> 128)
  - ReLU activation
  - Dropout (30% rate)
  - Linear (128 -> 16 classes)
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
- **Test Accuracy**: 0.6520
- **Classes**: ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']

### Detailed Classification Report:
```text
                     precision    recall  f1-score   support

              Blues       0.00      0.00      0.00        12
          Classical       0.92      0.84      0.88        80
            Country       0.67      0.25      0.36         8
     Easy Listening       0.00      0.00      0.00         3
         Electronic       0.57      0.64      0.60       572
       Experimental       0.61      0.67      0.64       635
               Folk       0.60      0.50      0.55       146
            Hip-Hop       0.63      0.60      0.61       205
       Instrumental       0.51      0.35      0.42        74
      International       0.70      0.66      0.68        93
               Jazz       0.77      0.21      0.33        47
Old-Time / Historic       0.93      0.66      0.77        38
                Pop       0.65      0.32      0.43       126
               Rock       0.72      0.78      0.75       916
           Soul-RnB       1.00      0.35      0.52        17
             Spoken       0.81      0.61      0.69        28

           accuracy                           0.65      3000
          macro avg       0.63      0.46      0.51      3000
       weighted avg       0.65      0.65      0.64      3000

```

## Limitations and Biases
- The model relies solely on text metadata (titles, artist names) and does not listen to the actual audio. Therefore, it might make errors on tracks with misleading names.
- Highly imbalanced classes (some genres like Rock and Electronic have significantly more tracks than Jazz or Classical in the FMA dataset).

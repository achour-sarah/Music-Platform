# Model Card - Audio Genre Classifier (FMA Small)

## Architecture
- **Backbone** : MERT-v1-95M (gelé - extracteur d'embeddings uniquement)
  - Sortie : mean pooling du dernier hidden state -> vecteur 768-dim
- **Classifieur** : XGB

## Dataset
- **Source** : FMA Small (Free Music Archive)
- **Labels** : Genres FMA natifs (aucun mapping externe) - 8 classes
- **Classes** : ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
- **Train** : 800 morceaux  |  **Test** : 200 morceaux

## Performances (jeu de test)
| Métrique    | Valeur  |
|-------------|---------|
| Accuracy    | 0.4950 (49.50%) |
| F1-macro    | 0.4818  |

## Rapport de classification complet
```
               precision    recall  f1-score   support

   Electronic       0.64      0.61      0.62        23
 Experimental       0.59      0.40      0.48        25
         Folk       0.31      0.36      0.33        22
      Hip-Hop       0.54      0.69      0.61        29
 Instrumental       0.39      0.48      0.43        23
International       0.50      0.53      0.52        30
          Pop       0.45      0.23      0.30        22
         Rock       0.56      0.58      0.57        26

     accuracy                           0.49       200
    macro avg       0.50      0.48      0.48       200
 weighted avg       0.50      0.49      0.49       200

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

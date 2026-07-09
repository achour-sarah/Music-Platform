# Model Card: Fine-Tuned Audio Classifier (Option B Fine-Tuning)
- **Architecture**: Wav2Vec2 base features + Custom PyTorch Head (768 -> 256 -> 128 -> 6 classes)
- **Dataset**: FMA Small Audio files fine-tuned on true database genres
- **Classes**: ['Classical', 'Country', 'Disco', 'Hiphop', 'Pop', 'Rock']
- **Test Accuracy**: 63.75%
- **Training Samples**: 320

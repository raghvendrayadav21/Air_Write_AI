# Model Directory

Place your trained model file here:

```
model/airwrite_model.h5
```

## Training the Model

Run the training script from the project root:

```bash
python train_model.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 25 | Number of training epochs |
| `--batch-size` | 128 | Batch size |
| `--letters-only` | True | Train on A-Z only (26 classes) |

### What the training script does

1. **Tries EMNIST Letters** – downloads ~500 MB dataset via `tensorflow_datasets`
2. **Falls back to synthetic data** – renders characters using system fonts (Pillow) if EMNIST isn't available. Works fully offline.
3. Trains a 3-block CNN with BatchNorm, GlobalAveragePooling, and Dropout
4. Saves the best checkpoint here as `airwrite_model.h5`
5. Generates a training history plot at `model/training_history.png`

## Expected Accuracy

| Dataset | Expected Test Accuracy |
|---------|----------------------|
| EMNIST Letters | ~90–95% |
| Synthetic (fonts) | ~85–90% |

## Model Architecture

```
Input  (28, 28, 1)
  ↓
Conv2D(32, 3×3) + BatchNorm + ReLU
  ↓
MaxPool(2×2)
  ↓
Conv2D(64, 3×3) + BatchNorm + ReLU
  ↓
MaxPool(2×2)
  ↓
Conv2D(128, 3×3) + BatchNorm + ReLU
  ↓
GlobalAveragePooling2D
  ↓
Dense(256) + ReLU + Dropout(0.5)
  ↓
Dense(26) + Softmax
```

## Using a Pre-trained Model

You can also use any compatible Keras `.h5` model trained on EMNIST or MNIST-style data.
Just rename it to `airwrite_model.h5` and place it here.

The model must accept input of shape `(batch, 28, 28, 1)` and output `(batch, 26)` or `(batch, 36)`.

"""
train_model.py – AirWrite AI CNN Training Script

Trains a CNN on handwritten character images (A-Z).

Strategy:
  1. Try to load EMNIST Letters dataset via tensorflow_datasets.
  2. If unavailable, fall back to generating synthetic training data
     from system fonts using Pillow – works fully offline.

Saves the trained model to: model/airwrite_model.h5

Usage:
    python train_model.py
    python train_model.py --epochs 30 --batch-size 64 --letters-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# ── Suppress TF verbose logging ───────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import config


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_emnist_letters() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Attempt to load EMNIST Letters via tensorflow_datasets.

    Returns:
        (x_train, y_train, x_test, y_test) – normalised float32, labels 0-based.

    Raises:
        ImportError / Exception if tensorflow_datasets is not installed or
        the dataset cannot be downloaded.
    """
    import tensorflow_datasets as tfds
    import tensorflow as tf

    print("  Loading EMNIST Letters dataset (this may download ~500 MB)…")

    (ds_train, ds_test), info = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
    )

    def extract(ds):
        images, labels = [], []
        for img, lbl in tfds.as_numpy(ds):
            # EMNIST images are (28,28), labels 1-26 → we remap to 0-25
            img = img.astype(np.float32) / 255.0
            # EMNIST letters are rotated 90° and mirrored – correct them
            img = np.rot90(img, k=-1, axes=(0, 1))
            img = np.fliplr(img)
            images.append(img)
            labels.append(int(lbl) - 1)   # 1-indexed → 0-indexed
        return np.array(images), np.array(labels)

    x_train, y_train = extract(ds_train)
    x_test,  y_test  = extract(ds_test)

    print(f"  EMNIST: {len(x_train)} train / {len(x_test)} test samples (26 classes)")
    return x_train, y_train, x_test, y_test


def generate_synthetic_dataset(
    img_size: int = 28,
    samples_per_class: int = 400,
    letters_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic dataset by rendering characters with Pillow fonts.

    Works fully offline. Generates variations using rotation, scale, and
    pixel-level noise to improve model generalisation.

    Args:
        img_size:          Size of each output image (square).
        samples_per_class: Number of augmented samples per character.
        letters_only:      If True, only A-Z; otherwise A-Z + 0-9.

    Returns:
        (x_train, y_train, x_test, y_test)
    """
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    import random
    from recognition.labels import LETTER_LABELS, ALL_LABELS

    labels = LETTER_LABELS if letters_only else ALL_LABELS
    num_classes = len(labels)

    print(f"  Generating synthetic dataset: {num_classes} classes × {samples_per_class} samples…")

    # Attempt to load a variety of fonts for diversity
    font_paths = _find_system_fonts()
    fonts_loaded = []
    for fp in font_paths:
        for size in [18, 20, 22, 24]:
            try:
                fonts_loaded.append(ImageFont.truetype(fp, size))
            except Exception:
                pass

    if not fonts_loaded:
        print("  No TTF fonts found – using PIL default font (lower quality).")
        default_font = ImageFont.load_default()
        fonts_loaded = [default_font]

    all_images, all_labels = [], []

    for cls_idx, char in enumerate(labels):
        for _ in range(samples_per_class):
            img = _render_char(char, img_size, fonts_loaded)
            all_images.append(img)
            all_labels.append(cls_idx)

        if (cls_idx + 1) % 10 == 0:
            print(f"    {cls_idx + 1}/{num_classes} classes done…")

    all_images = np.array(all_images, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(all_images))
    all_images = all_images[idx]
    all_labels = all_labels[idx]

    # 90/10 train/test split
    split = int(0.9 * len(all_images))
    x_train, y_train = all_images[:split], all_labels[:split]
    x_test,  y_test  = all_images[split:], all_labels[split:]

    print(f"  Synthetic dataset: {len(x_train)} train / {len(x_test)} test")
    return x_train, y_train, x_test, y_test


def _render_char(
    char: str,
    img_size: int,
    fonts: list,
) -> np.ndarray:
    """Render a single character with random augmentation.

    Returns a float32 array of shape (img_size, img_size) normalised to [0,1].
    """
    import random
    from PIL import Image, ImageDraw, ImageFilter
    import math

    font = random.choice(fonts)

    # Create a larger canvas to allow rotation without clipping
    canvas_size = int(img_size * 2.5)
    bg = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(bg)

    # Get character bounding box
    try:
        bbox = font.getbbox(char)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        tw, th = img_size // 2, img_size // 2

    # Center the character
    x = (canvas_size - tw) // 2 - (bbox[0] if hasattr(font, 'getbbox') else 0)
    y = (canvas_size - th) // 2 - (bbox[1] if hasattr(font, 'getbbox') else 0)

    # Random brightness (simulate pen pressure)
    brightness = random.randint(200, 255)
    draw.text((x, y), char, fill=brightness, font=font)

    # ── Augmentations ─────────────────────────────────────────────────────────
    # 1. Random rotation (±25°)
    angle = random.uniform(-25, 25)
    bg = bg.rotate(angle, expand=False, fillcolor=0)

    # 2. Random scale (80 – 120 %)
    scale  = random.uniform(0.80, 1.20)
    new_sz = int(canvas_size * scale)
    bg = bg.resize((new_sz, new_sz), Image.LANCZOS)

    # Re-crop/pad to canvas_size
    final = Image.new("L", (canvas_size, canvas_size), 0)
    ox = (canvas_size - new_sz) // 2
    oy = (canvas_size - new_sz) // 2
    final.paste(bg, (ox, oy))

    # 3. Slight blur (simulate air-writing smoothness)
    if random.random() < 0.5:
        final = final.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.9)))

    # 4. Crop to img_size centered
    left = (canvas_size - img_size) // 2
    top  = (canvas_size - img_size) // 2
    final = final.crop((left, top, left + img_size, top + img_size))

    # 5. Pixel noise
    arr = np.array(final, dtype=np.float32) / 255.0
    arr += np.random.normal(0, 0.02, arr.shape)
    arr = np.clip(arr, 0.0, 1.0)

    return arr


def _find_system_fonts() -> list[str]:
    """Return a list of available TTF font paths on the system."""
    import platform
    candidate_dirs = []

    system = platform.system()
    if system == "Windows":
        candidate_dirs = [
            r"C:\Windows\Fonts",
        ]
    elif system == "Darwin":
        candidate_dirs = [
            "/Library/Fonts",
            "/System/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ]
    else:  # Linux
        candidate_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
        ]

    fonts = []
    for d in candidate_dirs:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.lower().endswith((".ttf", ".otf")):
                    full = os.path.join(d, fname)
                    # Prefer common readable fonts (skip icon fonts)
                    if any(k in fname.lower() for k in [
                        "arial", "times", "courier", "calibri", "georgia",
                        "verdana", "trebuchet", "comic", "impact", "consolas",
                        "tahoma", "segoe", "dejavu", "liberation", "ubuntu",
                        "freemono", "freesans", "freeserif",
                    ]):
                        fonts.append(full)

    # If no preferred fonts found, take any TTF
    if not fonts:
        for d in candidate_dirs:
            if os.path.isdir(d):
                for fname in os.listdir(d):
                    if fname.lower().endswith(".ttf"):
                        fonts.append(os.path.join(d, fname))
                        if len(fonts) >= 5:
                            break

    return fonts[:10]  # Cap to avoid too many fonts


# ─────────────────────────────────────────────────────────────────────────────
#  CNN Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn(num_classes: int, img_size: int = 28) -> "keras.Model":
    """Build and compile the CNN model.

    Architecture:
      Input (28, 28, 1)
      Conv2D(32, 3) → BatchNorm → ReLU
      MaxPool(2)
      Conv2D(64, 3) → BatchNorm → ReLU
      MaxPool(2)
      Conv2D(128, 3) → BatchNorm → ReLU
      GlobalAveragePooling
      Dense(256) → ReLU → Dropout(0.5)
      Dense(num_classes) → Softmax
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        # ── Block 1 ───────────────────────────────────────────────────────────
        layers.Input(shape=(img_size, img_size, 1)),
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # ── Block 2 ───────────────────────────────────────────────────────────
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # ── Block 3 ───────────────────────────────────────────────────────────
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        # ── Head ──────────────────────────────────────────────────────────────
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.50),
        layers.Dense(num_classes, activation="softmax"),
    ], name="AirWriteCNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    epochs: int         = config.TRAIN_EPOCHS,
    batch_size: int     = config.TRAIN_BATCH_SIZE,
    letters_only: bool  = True,
) -> None:
    """Main training entry point.

    1. Load dataset (EMNIST or synthetic fallback).
    2. Build CNN.
    3. Train with callbacks.
    4. Evaluate.
    5. Save model.
    """
    import tensorflow as tf
    from tensorflow import keras

    print()
    print("=" * 60)
    print("  AirWrite AI – CNN Training")
    print("=" * 60)
    print(f"  TensorFlow version : {tf.__version__}")
    print(f"  Epochs             : {epochs}")
    print(f"  Batch size         : {batch_size}")
    print(f"  Mode               : {'Letters only (A-Z)' if letters_only else 'A-Z + 0-9'}")
    print()

    # ── Dataset ───────────────────────────────────────────────────────────────
    x_train, y_train, x_test, y_test = None, None, None, None

    try:
        x_train, y_train, x_test, y_test = load_emnist_letters()
        num_classes_actual = 26
        letters_only = True
    except Exception as e:
        print(f"  EMNIST not available ({e})")
        print("  Falling back to synthetic dataset generation…")
        print()
        x_train, y_train, x_test, y_test = generate_synthetic_dataset(
            img_size=config.TRAIN_IMG_SIZE,
            samples_per_class=500,
            letters_only=letters_only,
        )
        num_classes_actual = 26 if letters_only else 36

    # Add channel dimension (H, W) → (H, W, 1)
    x_train = x_train[..., np.newaxis]
    x_test  = x_test[..., np.newaxis]

    print(f"  x_train shape: {x_train.shape}  y_train shape: {y_train.shape}")
    print(f"  x_test  shape: {x_test.shape}   y_test  shape: {y_test.shape}")
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_cnn(num_classes=num_classes_actual, img_size=config.TRAIN_IMG_SIZE)
    model.summary()
    print()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_PATH,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # ── Training ──────────────────────────────────────────────────────────────
    print("  Starting training…")
    t0 = time.time()

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=config.TRAIN_VAL_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed = time.time() - t0

    # ── Evaluation ────────────────────────────────────────────────────────────
    print()
    print("  Evaluating on test set…")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Training time  : {elapsed/60:.1f} minutes")
    print(f"  Test accuracy  : {acc*100:.2f}%")
    print(f"  Test loss      : {loss:.4f}")
    print(f"  Model saved to : {config.MODEL_PATH}")
    print()

    # ── Plot training curves (optional) ───────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("AirWrite AI – Training History", fontsize=14)

        axes[0].plot(history.history["accuracy"],     label="Train Acc")
        axes[0].plot(history.history["val_accuracy"], label="Val Acc")
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history.history["loss"],     label="Train Loss")
        axes[1].plot(history.history["val_loss"], label="Val Loss")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True)

        plot_path = os.path.join(config.MODEL_DIR, "training_history.png")
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        print(f"  Training plot saved to: {plot_path}")
        plt.show()
    except Exception:
        pass  # matplotlib not available – skip plot


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the AirWrite AI CNN character recognition model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",       type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--batch-size",   type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--letters-only", action="store_true", default=True,
                        help="Train on A-Z only (26 classes) instead of A-Z + 0-9.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    train(
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        letters_only= args.letters_only,
    )

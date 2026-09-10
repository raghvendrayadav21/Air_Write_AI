"""
recognition/predictor.py

Loads the trained CNN model and performs character prediction.

Returns:
  - Predicted character label
  - Confidence percentage
  - Top-K (default 3) predictions as [(label, confidence), ...]
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

import config
from recognition.labels import index_to_label


class Predictor:
    """Wraps the Keras/TF model and provides a clean prediction API."""

    def __init__(self, model_path: str = config.MODEL_PATH) -> None:
        self._model      = None
        self._model_path = model_path
        self._loaded     = False
        self._error: Optional[str] = None

        self._try_load()

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True when the model is loaded and ready."""
        return self._loaded

    @property
    def error_message(self) -> Optional[str]:
        """Return the error string if loading failed, else None."""
        return self._error

    def predict(
        self,
        tensor: np.ndarray,
        top_k: int = config.TOP_K,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Run inference on a preprocessed image tensor.

        Args:
            tensor: Float32 array of shape (1, IMG_SIZE, IMG_SIZE, 1).
            top_k:  Number of top predictions to return.

        Returns:
            (best_label, best_confidence, top_k_list)
            where top_k_list is [(label, confidence), ...] sorted descending.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. " + (self._error or ""))

        # Model inference
        probs = self._model.predict(tensor, verbose=0)[0]  # shape: (num_classes,)

        # Determine if model outputs 26 or 36 classes
        letters_only = len(probs) == 26

        # Sort descending
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_results = [
            (index_to_label(int(i), letters_only), float(probs[i]) * 100)
            for i in top_indices
        ]

        best_label, best_conf = top_results[0]
        return best_label, best_conf, top_results

    def predict_word(
        self,
        character_tensors: List[np.ndarray],
    ) -> Tuple[str, float, List[Tuple[str, float]], Optional[str]]:
        """Run inference on multiple character tensors and assemble into a word.

        Args:
            character_tensors: List of float32 tensors, each (1, IMG_SIZE, IMG_SIZE, 1).

        Returns:
            (word_string, avg_confidence, per_character_results, suggested_word)
            where per_character_results is [(char_label, confidence), ...]
            and suggested_word is an optional dictionary-matched word.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. " + (self._error or ""))

        if not character_tensors:
            return "", 0.0, [], None

        # Batch predict for instant real-time response
        batch = np.concatenate(character_tensors, axis=0)
        all_probs = self._model.predict(batch, verbose=0)

        letters_only = all_probs.shape[1] == 26
        word_chars: List[str] = []
        confidences: List[float] = []
        char_results: List[Tuple[str, float]] = []

        for probs in all_probs:
            best_idx = int(np.argmax(probs))
            char = index_to_label(best_idx, letters_only)
            conf = float(probs[best_idx]) * 100.0
            word_chars.append(char)
            confidences.append(conf)
            char_results.append((char, conf))

        word = "".join(word_chars)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        # Dictionary lookup / spellcheck for English words
        suggested = self._match_dictionary(word, all_probs, letters_only)
        return word, avg_conf, char_results, suggested

    def _match_dictionary(
        self,
        word: str,
        all_probs: np.ndarray,
        letters_only: bool,
    ) -> Optional[str]:
        """Find the best dictionary word matching the predicted characters."""
        import difflib
        from recognition.vocabulary import COMMON_ENGLISH_WORDS

        word_upper = word.upper()
        if word_upper in COMMON_ENGLISH_WORDS:
            return word_upper

        # Close matches with high similarity
        matches = difflib.get_close_matches(word_upper, COMMON_ENGLISH_WORDS, n=3, cutoff=0.65)
        if matches:
            return matches[0]

        return None

    def reload(self) -> bool:
        """Attempt to reload the model from disk.

        Returns:
            True on success, False on failure.
        """
        self._loaded = False
        self._model  = None
        self._error  = None
        self._try_load()
        return self._loaded

    # ──────────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _try_load(self) -> None:
        """Attempt to load the Keras model; set _error on failure."""
        if not os.path.isfile(self._model_path):
            self._error = (
                f"Model file not found:\n{self._model_path}\n\n"
                "Run 'python train_model.py' to train and save the model."
            )
            return

        try:
            # Lazy import so the app starts even without TF installed
            import tensorflow as tf  # noqa: F401
            from tensorflow import keras

            self._model  = keras.models.load_model(self._model_path)
            self._loaded = True
        except Exception as exc:
            self._error = f"Failed to load model:\n{exc}"

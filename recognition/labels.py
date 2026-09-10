"""
recognition/labels.py
Maps model class index → human-readable character label.

Class layout (configurable in config.py):
  0-25  →  A-Z
  26-35 →  0-9
"""

# ─── Letters A-Z (indices 0–25) ───────────────────────────────────────────────
LETTER_LABELS = [chr(ord('A') + i) for i in range(26)]

# ─── Digits 0-9 (indices 26–35) ───────────────────────────────────────────────
DIGIT_LABELS = [str(i) for i in range(10)]

# ─── Combined A-Z + 0-9 ───────────────────────────────────────────────────────
ALL_LABELS = LETTER_LABELS + DIGIT_LABELS    # length 36

# ─── Quick lookup ─────────────────────────────────────────────────────────────

def index_to_label(index: int, letters_only: bool = False) -> str:
    """Return the character label for a given class index.

    Args:
        index: Model output class index.
        letters_only: If True, treat labels as A-Z only (26 classes).

    Returns:
        Character string.
    """
    labels = LETTER_LABELS if letters_only else ALL_LABELS
    if 0 <= index < len(labels):
        return labels[index]
    return "?"


def label_to_index(char: str, letters_only: bool = False) -> int:
    """Return the class index for a given character label.

    Args:
        char: Character string (e.g. 'A', '5').
        letters_only: If True, treat labels as A-Z only.

    Returns:
        Class index, or -1 if not found.
    """
    labels = LETTER_LABELS if letters_only else ALL_LABELS
    char = char.upper()
    try:
        return labels.index(char)
    except ValueError:
        return -1


def num_classes(letters_only: bool = False) -> int:
    """Return total number of classes."""
    return 26 if letters_only else 36

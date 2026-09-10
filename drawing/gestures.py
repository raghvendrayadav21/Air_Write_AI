"""
drawing/gestures.py

Classifies the current hand pose into one of four gestures:
  DRAW    – only index finger raised  ☝️
  PAUSE   – index + middle raised     ✌️
  CLEAR   – closed fist               ✊
  PREDICT – thumb up only             👍
  NONE    – unrecognised pose
"""

from __future__ import annotations

from typing import List

import config


def classify(fingers_up: List[int]) -> str:
    """Map a 5-element finger-state list to a gesture name.

    Args:
        fingers_up: [Thumb, Index, Middle, Ring, Pinky] – 1=raised, 0=folded.

    Returns:
        One of the GESTURE_* constants defined in config.
    """
    if len(fingers_up) != 5:
        return config.GESTURE_NONE

    thumb, index, middle, ring, pinky = fingers_up

    # ── PREDICT: only thumb raised (thumbs up) ───────────────────────────────
    if thumb == 1 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
        return config.GESTURE_PREDICT

    # ── DRAW: index finger raised, other 3 fingers folded (thumb relaxed/any) ──
    if index == 1 and middle == 0 and ring == 0 and pinky == 0:
        return config.GESTURE_DRAW

    # ── PAUSE: index + middle raised (peace sign, other 2 folded) ────────────
    if index == 1 and middle == 1 and ring == 0 and pinky == 0:
        return config.GESTURE_PAUSE

    # ── CLEAR: closed fist (all fingers folded) ──────────────────────────────
    if thumb == 0 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
        return config.GESTURE_CLEAR

    return config.GESTURE_NONE


# Emoji mapping for UI display
GESTURE_EMOJI = {
    config.GESTURE_DRAW:    "☝️  DRAW",
    config.GESTURE_PAUSE:   "✌️  PAUSE",
    config.GESTURE_CLEAR:   "✊  CLEAR",
    config.GESTURE_PREDICT: "👍  PREDICT",
    config.GESTURE_NONE:    "🤚  NONE",
}


def gesture_label(gesture: str) -> str:
    """Return a human-readable emoji label for a gesture."""
    return GESTURE_EMOJI.get(gesture, "—")

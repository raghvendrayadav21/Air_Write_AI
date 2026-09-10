"""
utils/helpers.py

Shared utility functions used across the project.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk


# ─────────────────────────────────────────────────────────────────────────────
#  OpenCV frame utilities
# ─────────────────────────────────────────────────────────────────────────────

def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a frame to (width, height)."""
    return cv2.resize(frame, (width, height))


def draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.7,
    font_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    padding: int = 6,
) -> None:
    """Draw text with a filled background rectangle for readability.

    Args:
        frame:      BGR image to draw on (modified in-place).
        text:       String to render.
        pos:        (x, y) top-left of text.
        font_scale: OpenCV font scale factor.
        font_color: BGR text color.
        bg_color:   BGR background rectangle color.
        thickness:  Font thickness.
        padding:    Pixel padding around text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = pos
    # Background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color,
        cv2.FILLED,
    )
    # Text
    cv2.putText(frame, text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)


def overlay_canvas_on_frame(
    frame: np.ndarray,
    canvas: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Alpha-blend white canvas strokes onto the webcam frame.

    Only non-black pixels from the canvas are blended so the background
    remains transparent.

    Args:
        frame:  BGR webcam frame (H×W×3).
        canvas: BGR canvas image – same size or resized to frame.
        alpha:  Blend weight for canvas strokes (0 = invisible, 1 = opaque).

    Returns:
        Blended BGR image.
    """
    if canvas.shape[:2] != frame.shape[:2]:
        canvas = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))

    # Create mask of non-zero canvas pixels
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    blended = frame.copy()
    blended[mask > 0] = cv2.addWeighted(
        frame, 1 - alpha, canvas, alpha, 0
    )[mask > 0]
    return blended


# ─────────────────────────────────────────────────────────────────────────────
#  Tkinter / PIL conversion
# ─────────────────────────────────────────────────────────────────────────────

def bgr_to_photoimage(bgr: np.ndarray) -> "ImageTk.PhotoImage":
    """Convert a BGR NumPy array to a Tkinter-compatible PhotoImage."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(image=pil_img)


def gray_to_photoimage(gray: np.ndarray) -> "ImageTk.PhotoImage":
    """Convert a grayscale NumPy array to a Tkinter-compatible PhotoImage."""
    pil_img = Image.fromarray(gray)
    return ImageTk.PhotoImage(image=pil_img)


# ─────────────────────────────────────────────────────────────────────────────
#  Status / colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def confidence_color(confidence: float) -> str:
    """Return a hex colour string based on prediction confidence.

    ≥80 % → green, 50–80 % → orange, <50 % → red
    """
    if confidence >= 80:
        return "#00E676"
    elif confidence >= 50:
        return "#FFA726"
    else:
        return "#EF5350"

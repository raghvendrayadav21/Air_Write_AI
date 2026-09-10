"""
drawing/air_canvas.py

Manages the virtual black canvas on which the user's air-writing is drawn.

Responsibilities:
  - Create and maintain a black canvas (NumPy array)
  - Draw lines between consecutive finger positions
  - Clear the canvas
  - Return a copy of the current canvas
  - Save the canvas to disk
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np

import config


class AirCanvas:
    """A virtual drawing surface for air-writing."""

    def __init__(
        self,
        width:  int = config.CANVAS_WIDTH,
        height: int = config.CANVAS_HEIGHT,
    ) -> None:
        self.width  = width
        self.height = height

        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._prev_point: Optional[Tuple[int, int]] = None

        # Smoothing buffer for jitter reduction
        self._smooth_window: list[Tuple[int,int]] = []
        self._smooth_size = 3

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def draw(self, point: Tuple[int, int]) -> None:
        """Draw from the previous point to the current point.

        Coordinates are expected to be in webcam-frame space and are
        automatically scaled to canvas space.

        Args:
            point: (x, y) in webcam pixel coordinates.
        """
        # ── Scale webcam coords → canvas coords ───────────────────────────────
        cx = int(point[0] * self.width  / config.WEBCAM_WIDTH)
        cy = int(point[1] * self.height / config.WEBCAM_HEIGHT)
        canvas_point = (cx, cy)

        # ── Smooth jitter ─────────────────────────────────────────────────────
        self._smooth_window.append(canvas_point)
        if len(self._smooth_window) > self._smooth_size:
            self._smooth_window.pop(0)

        smoothed = (
            int(sum(p[0] for p in self._smooth_window) / len(self._smooth_window)),
            int(sum(p[1] for p in self._smooth_window) / len(self._smooth_window)),
        )

        # ── Draw ──────────────────────────────────────────────────────────────
        if self._prev_point is not None:
            cv2.line(
                self._canvas,
                self._prev_point,
                smoothed,
                config.DRAW_COLOR,
                config.DRAW_THICKNESS,
            )
            # Rounded line caps
            cv2.circle(
                self._canvas,
                smoothed,
                config.DRAW_THICKNESS // 2,
                config.DRAW_COLOR,
                cv2.FILLED,
            )

        self._prev_point = smoothed

    def lift_pen(self) -> None:
        """Call when the user pauses drawing (pen lifted)."""
        self._prev_point = None
        self._smooth_window.clear()

    def clear(self) -> None:
        """Clear the entire canvas."""
        self._canvas[:] = 0
        self.lift_pen()

    def get_image(self) -> np.ndarray:
        """Return a copy of the current canvas (BGR)."""
        return self._canvas.copy()

    def is_empty(self) -> bool:
        """Return True if the canvas has no strokes."""
        return np.sum(self._canvas) == 0

    def save(self, path: Optional[str] = None) -> str:
        """Save the canvas to an image file.

        Args:
            path: Optional file path. Defaults to screenshots/<timestamp>.png.

        Returns:
            Absolute path to the saved file.
        """
        if path is None:
            os.makedirs(config.SCREENSHOTS, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(config.SCREENSHOTS, f"airwrite_{timestamp}.png")

        cv2.imwrite(path, self._canvas)
        return path

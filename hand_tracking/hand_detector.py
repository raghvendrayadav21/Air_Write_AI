"""
hand_tracking/hand_detector.py

Wraps MediaPipe Hands to provide:
  - Hand landmark detection
  - Index-finger tip position
  - Per-finger "up/down" state
  - Annotated frame drawing
"""

from __future__ import annotations

# Protobuf 6.x compatibility shim for MediaPipe 0.10.x
try:
    from google.protobuf import message_factory as _mf, symbol_database as _sd
    _proto_fn = lambda self_or_cls, desc: _mf.GetMessageClass(desc)
    if not hasattr(_mf.MessageFactory, "GetPrototype"):
        _mf.MessageFactory.GetPrototype = _proto_fn
    if not hasattr(_mf, "GetPrototype"):
        _mf.GetPrototype = _mf.GetMessageClass
    if not hasattr(_sd.SymbolDatabase, "GetPrototype"):
        _sd.SymbolDatabase.GetPrototype = _proto_fn
    if not hasattr(_sd, "GetPrototype"):
        _sd.GetPrototype = _mf.GetMessageClass
except Exception:
    pass


import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List

import config


class HandDetector:
    """Detects a single hand and exposes landmark data + finger states."""

    # MediaPipe hand-landmark IDs for each fingertip and pip joint
    FINGER_TIPS = [
        config.THUMB_TIP,    # 4
        config.INDEX_TIP,    # 8
        config.MIDDLE_TIP,   # 12
        config.RING_TIP,     # 16
        config.PINKY_TIP,    # 20
    ]
    FINGER_PIPS = [3, 6, 10, 14, 18]  # one joint below each tip

    def __init__(
        self,
        max_hands: int           = config.MAX_HANDS,
        detection_conf: float    = config.MIN_DETECTION_CONF,
        tracking_conf: float     = config.MIN_TRACKING_CONF,
    ) -> None:
        self._mp_hands  = mp.solutions.hands
        self._mp_draw   = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self.hands = self._mp_hands.Hands(
            static_image_mode       = False,
            max_num_hands           = max_hands,
            min_detection_confidence= detection_conf,
            min_tracking_confidence = tracking_conf,
        )

        # Internal state populated on each call to find_hands()
        self.landmarks: List[Tuple[int,int]] = []   # (x, y) pixel coords
        self.results   = None
        self.handedness: str = "Right"

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def find_hands(
        self,
        frame: np.ndarray,
        draw: bool = True,
    ) -> np.ndarray:
        """Process a BGR frame, detect hand, optionally draw landmarks.

        Args:
            frame: BGR image from OpenCV capture.
            draw:  Whether to annotate the frame with landmarks.

        Returns:
            Annotated (or original) frame.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        self.landmarks = []

        if self.results.multi_hand_landmarks:
            # Use the first detected hand only
            hand_lms = self.results.multi_hand_landmarks[0]
            if self.results.multi_handedness:
                self.handedness = self.results.multi_handedness[0].classification[0].label

            h, w = frame.shape[:2]
            for lm in hand_lms.landmark:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                self.landmarks.append((cx, cy))

            if draw:
                self._mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )
                self._highlight_index_tip(frame)

        return frame

    def get_index_tip(self) -> Optional[Tuple[int, int]]:
        """Return the (x, y) pixel position of the index-finger tip, or None."""
        if len(self.landmarks) > config.INDEX_TIP:
            return self.landmarks[config.INDEX_TIP]
        return None

    def fingers_up(self) -> List[int]:
        """Return a 5-element list of 1/0 indicating which fingers are raised.

        Order: [Thumb, Index, Middle, Ring, Pinky]
        """
        if len(self.landmarks) < 21:
            return [0, 0, 0, 0, 0]

        fingers = []

        # ── Thumb ─────────────────────────────────────────────────────────────
        # Check both horizontal extension and vertical thumbs-up
        thumb_tip = self.landmarks[config.THUMB_TIP]
        thumb_ip  = self.landmarks[3]
        thumb_mcp = self.landmarks[2]

        # In mirrored feed:
        if self.handedness == "Right":
            thumb_extended = thumb_tip[0] < thumb_ip[0]
        else:
            thumb_extended = thumb_tip[0] > thumb_ip[0]

        # Also consider thumbs-up pose (tip is significantly higher than MCP)
        is_thumbs_up = thumb_tip[1] < thumb_mcp[1] and abs(thumb_tip[0] - thumb_mcp[0]) < 60

        fingers.append(1 if (thumb_extended or is_thumbs_up) else 0)

        # ── Other four fingers (tip.y vs pip.y) ───────────────────────────────
        for tip, pip in zip(self.FINGER_TIPS[1:], self.FINGER_PIPS[1:]):
            tip_y = self.landmarks[tip][1]
            pip_y = self.landmarks[pip][1]
            fingers.append(1 if tip_y < pip_y else 0)

        return fingers

    def hand_detected(self) -> bool:
        """Return True if a hand is currently detected."""
        return bool(self.landmarks)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()

    # ──────────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _highlight_index_tip(self, frame: np.ndarray) -> None:
        """Draw a prominent circle around the index-finger tip."""
        tip = self.get_index_tip()
        if tip:
            # Outer glow ring
            cv2.circle(frame, tip, 18, (0, 255, 255), 3)
            # Inner filled dot
            cv2.circle(frame, tip, 8, (0, 200, 255), cv2.FILLED)

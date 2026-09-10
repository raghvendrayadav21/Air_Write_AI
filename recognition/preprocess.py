"""
recognition/preprocess.py

Preprocessing and Character Segmentation pipeline:
  - preprocess: Single-character bounding box crop, pad, resize, normalize.
  - segment_characters: Multi-character segmentation for whole-word recognition.
    Finds and clusters strokes left-to-right into separate letters.
  - annotate_word_canvas: Draws bounding boxes and letter tags on canvas preview.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


def preprocess(canvas_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Convert a single-character canvas image into a model-ready tensor.

    Args:
        canvas_bgr: The raw canvas image in BGR format (H×W×3 uint8).

    Returns:
        Normalised float32 array of shape (1, IMG_SIZE, IMG_SIZE, 1),
        or None if the canvas appears empty.
    """
    gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    return _crop_and_normalize(binary, x, y, w, h)


def segment_characters(
    canvas_bgr: np.ndarray,
    merge_gap: int = getattr(config, "WORD_MERGE_GAP", 28),
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Segment a canvas containing one or more drawn characters.

    Finds all strokes, splits conjoined wide letters, clusters multi-stroke
    letters (like crossbars in 'A'/'H' or dots in 'i'/'j') while preserving
    independent letters, sorts left-to-right, and returns model-ready tensors.

    Args:
        canvas_bgr: The raw canvas image in BGR format (H×W×3 uint8).
        merge_gap:  Unused legacy parameter kept for API compatibility.

    Returns:
        List of (tensor, (x, y, w, h)) sorted from left to right.
    """
    gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(binary) < 40:
        return []

    # 1. Find external contours of drawn strokes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes: List[List[int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= 4 and h >= 6 and cv2.contourArea(cnt) >= 15:
            raw_boxes.append([x, y, w, h])

    if not raw_boxes:
        return []

    # 2. VPP Split for any touching/conjoined wide characters (w > 1.25 * h)
    refined_boxes: List[List[int]] = []
    for x, y, w, h in raw_boxes:
        if w > 1.25 * h and w >= 24:
            crop = binary[y:y+h, x:x+w]
            vpp = np.sum(crop > 0, axis=0).astype(float)
            ksize = max(3, int(w * 0.08))
            if ksize % 2 == 0:
                ksize += 1
            kernel = np.ones(ksize) / ksize
            smoothed = np.convolve(vpp, kernel, mode="same")

            margin = max(6, int(h * 0.25))
            best_col = None
            min_val = float("inf")
            for c in range(margin, w - margin):
                left_max = np.max(smoothed[:c])
                right_max = np.max(smoothed[c:])
                if smoothed[c] < 0.6 * min(left_max, right_max) and smoothed[c] < min_val:
                    min_val = smoothed[c]
                    best_col = c

            if best_col is not None and best_col >= 6 and (w - best_col) >= 6:
                refined_boxes.append([x, y, best_col, h])
                refined_boxes.append([x + best_col, y, w - best_col, h])
            else:
                refined_boxes.append([x, y, w, h])
        else:
            refined_boxes.append([x, y, w, h])

    # 3. Sort strokes by X coordinate (left to right)
    refined_boxes.sort(key=lambda b: b[0])

    # 4. Smart Stroke Clustering
    # In uppercase English handwriting, strokes belonging to the same letter
    # either overlap horizontally (e.g. crossbars in A, H, E, F, T, X) or are
    # vertically stacked (dot of 'i'/'j' directly above stem).
    # Strokes with a positive horizontal gap are separate letters.
    merged_boxes: List[List[int]] = []
    for b in refined_boxes:
        if not merged_boxes:
            merged_boxes.append(list(b))
            continue

        prev = merged_boxes[-1]
        px, py, pw, ph = prev
        cx, cy, cw, ch = b
        pr = px + pw
        cr = cx + cw

        # Horizontal overlap amount
        overlap_x = min(pr, cr) - max(px, cx)

        # Vertical dot alignment (e.g., dot of 'i' or 'j' above/below stem)
        center_p = px + pw / 2.0
        center_c = cx + cw / 2.0
        vert_aligned = abs(center_p - center_c) <= max(pw, cw) * 0.8
        vert_stacked = (cy >= py + ph) or (py >= cy + ch)

        comb_w = max(pr, cr) - min(px, cx)
        comb_h = max(py + ph, cy + ch) - min(py, cy)
        comb_ar = comb_w / max(1, comb_h)

        should_merge = False
        if vert_aligned and vert_stacked:
            should_merge = True
        elif overlap_x > 0:
            # Overlapping strokes merge if combined aspect ratio is reasonable for a letter
            if comb_ar <= 1.25:
                should_merge = True

        if should_merge:
            nx = min(px, cx)
            ny = min(py, cy)
            nw = max(pr, cr) - nx
            nh = max(py + ph, cy + ch) - ny
            merged_boxes[-1] = [nx, ny, nw, nh]
        else:
            merged_boxes.append(list(b))

    # 5. Convert each merged letter box into model-ready tensor
    results: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
    for b in merged_boxes:
        x, y, w, h = b
        if w < 4 or h < 4:
            continue
        tensor = _crop_and_normalize(binary, x, y, w, h)
        if tensor is not None:
            results.append((tensor, (x, y, w, h)))

    return results


def _crop_and_normalize(
    binary_img: np.ndarray,
    x: int, y: int, w: int, h: int,
) -> Optional[np.ndarray]:
    """Helper: Crop box, pad to square, resize to 28×28, and normalize to [0, 1]."""
    cropped = binary_img[y:y+h, x:x+w]
    if cropped.size == 0:
        return None

    size    = max(w, h)
    pad_x   = (size - w) // 2
    pad_y   = (size - h) // 2
    padding = max(int(size * 0.20), 4)

    square = cv2.copyMakeBorder(
        cropped,
        pad_y + padding,
        pad_y + padding,
        pad_x + padding,
        pad_x + padding,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    resized = cv2.resize(
        square,
        (config.IMG_SIZE, config.IMG_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    normalised = resized.astype(np.float32) / 255.0
    return normalised.reshape(1, config.IMG_SIZE, config.IMG_SIZE, 1)


def annotate_word_canvas(
    canvas_bgr: np.ndarray,
    predictions: List[Tuple[str, float, Tuple[int, int, int, int]]],
) -> np.ndarray:
    """Draw bounding boxes and recognized letter badges over the canvas.

    Args:
        canvas_bgr:  Original canvas BGR image.
        predictions: List of (letter, confidence, (x, y, w, h)).

    Returns:
        Annotated copy of the canvas BGR image.
    """
    annotated = canvas_bgr.copy()
    for letter, conf, (x, y, w, h) in predictions:
        # Bounding box
        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(annotated.shape[1] - 1, x + w + pad)
        y2 = min(annotated.shape[0] - 1, y + h + pad)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 128), 2)

        # Badge tag on top
        label_text = f"{letter} ({conf:.0f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, scale, thick)

        badge_y1 = max(0, y1 - th - 8)
        badge_y2 = y1
        badge_x2 = min(annotated.shape[1] - 1, x1 + tw + 8)

        cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), (0, 200, 100), cv2.FILLED)
        cv2.putText(
            annotated,
            label_text,
            (x1 + 4, badge_y2 - 4),
            font,
            scale,
            (0, 0, 0),
            thick,
            cv2.LINE_AA,
        )

    return annotated


def get_preview(canvas_bgr: np.ndarray, size: int = 128) -> np.ndarray:
    """Return a resized preview of the preprocessed image for UI."""
    tensor = preprocess(canvas_bgr)
    if tensor is None:
        return np.zeros((size, size), dtype=np.uint8)

    img28 = (tensor.squeeze() * 255).astype(np.uint8)
    return cv2.resize(img28, (size, size), interpolation=cv2.INTER_NEAREST)

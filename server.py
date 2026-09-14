"""
server.py – AirWrite AI Web Backend

Flask server that:
  - Serves the frontend static files
  - Exposes /api/predict  → CNN character/word recognition
  - Exposes /api/health   → model status check

Exports a top-level `app` variable for Vercel / Gunicorn compatibility.

Usage (local dev):
    python server.py

Usage (production via Gunicorn):
    gunicorn server:app
"""

from __future__ import annotations

import base64
import io
import os
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Ensure project root is on the path ───────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Protobuf 6.x compatibility shim (needed by mediapipe) ───────────────────
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

import config
from recognition.predictor import Predictor
from recognition.preprocess import preprocess, segment_characters

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(PROJECT_ROOT, "static"),
    static_url_path="/static",
)
CORS(app)  # Allow cross-origin requests (needed for local dev)

# ── Load model once at startup ────────────────────────────────────────────────
_predictor = Predictor(model_path=config.MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────

def _decode_image(b64_string: str) -> np.ndarray:
    """Decode a base64 PNG/JPEG string into a BGR numpy array."""
    # Strip optional data-URI prefix
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image from base64 data.")
    return bgr


# ─────────────────────────────────────────────────────────────────────────────
#  Routes – Frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page frontend."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    """Catch-all for any static asset the frontend requests."""
    return send_from_directory(app.static_folder, filename)


# ─────────────────────────────────────────────────────────────────────────────
#  Routes – API
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Return model readiness status."""
    return jsonify({
        "model_loaded": _predictor.is_ready,
        "error": _predictor.error_message,
        "model_path": config.MODEL_PATH,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON body: { "image": "<base64 PNG of the air-canvas>" }

    Returns:
    {
        "success": true,
        "mode": "word" | "single",
        "word": "HELLO",
        "suggested": "HELLO",
        "confidence": 91.3,
        "characters": [
            {"char": "H", "confidence": 95.1},
            ...
        ],
        "top3": [
            {"label": "HELLO", "confidence": 91.3},
            ...
        ]
    }
    """
    if not _predictor.is_ready:
        return jsonify({
            "success": False,
            "error": "Model not loaded: " + (_predictor.error_message or "unknown error"),
        }), 503

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"success": False, "error": "Missing 'image' field in JSON body."}), 400

    try:
        canvas_bgr = _decode_image(data["image"])
    except Exception as exc:
        return jsonify({"success": False, "error": f"Image decode error: {exc}"}), 400

    # ── Try multi-character word recognition first ────────────────────────────
    segments = segment_characters(canvas_bgr)

    if len(segments) > 1:
        # Word mode
        tensors = [t for t, _ in segments]
        try:
            word, avg_conf, char_results, suggested = _predictor.predict_word(tensors)
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500

        return jsonify({
            "success": True,
            "mode": "word",
            "word": word,
            "suggested": suggested,
            "confidence": round(avg_conf, 1),
            "characters": [
                {"char": ch, "confidence": round(cf, 1)}
                for ch, cf in char_results
            ],
            "top3": [],
        })

    else:
        # Single-character mode
        tensor = preprocess(canvas_bgr)
        if tensor is None:
            return jsonify({"success": False, "error": "Canvas appears empty."}), 400

        try:
            best_label, best_conf, top_k = _predictor.predict(tensor)
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500

        return jsonify({
            "success": True,
            "mode": "single",
            "word": best_label,
            "suggested": None,
            "confidence": round(best_conf, 1),
            "characters": [{"char": best_label, "confidence": round(best_conf, 1)}],
            "top3": [
                {"label": lbl, "confidence": round(cf, 1)}
                for lbl, cf in top_k
            ],
        })


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"\n  🚀  AirWrite AI Web Server")
    print(f"  📡  http://localhost:{port}")
    print(f"  🧠  Model loaded: {_predictor.is_ready}")
    if not _predictor.is_ready:
        print(f"  ⚠️   {_predictor.error_message}")
    print()
    app.run(host="0.0.0.0", port=port, debug=debug)

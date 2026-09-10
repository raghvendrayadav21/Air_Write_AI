"""
AirWrite AI - Central Configuration
All project-wide settings live here.
"""

import os

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "model")
ASSETS_DIR  = os.path.join(BASE_DIR, "assets")
SCREENSHOTS = os.path.join(BASE_DIR, "screenshots")

MODEL_PATH  = os.path.join(MODEL_DIR, "airwrite_model.h5")

# ─────────────────────────────────────────────
#  Canvas
# ─────────────────────────────────────────────
CANVAS_WIDTH  = 640
CANVAS_HEIGHT = 480
DRAW_COLOR    = (255, 255, 255)   # White strokes on black canvas
DRAW_THICKNESS = 10
ERASER_THICKNESS = 30
WORD_MERGE_GAP = 28               # Max horizontal pixel gap to merge strokes into one letter

# ─────────────────────────────────────────────
#  Webcam
# ─────────────────────────────────────────────
WEBCAM_INDEX  = 0
WEBCAM_WIDTH  = 640
WEBCAM_HEIGHT = 480
MIRROR_FEED   = True              # Flip horizontal for natural interaction

# ─────────────────────────────────────────────
#  MediaPipe Hand Tracking
# ─────────────────────────────────────────────
MAX_HANDS           = 1
MIN_DETECTION_CONF  = 0.5
MIN_TRACKING_CONF   = 0.5

# Landmark indices
THUMB_TIP   = 4
INDEX_TIP   = 8
MIDDLE_TIP  = 12
RING_TIP    = 16
PINKY_TIP   = 20

# ─────────────────────────────────────────────
#  Gesture Names
# ─────────────────────────────────────────────
GESTURE_DRAW    = "DRAW"
GESTURE_PAUSE   = "PAUSE"
GESTURE_CLEAR   = "CLEAR"
GESTURE_PREDICT = "PREDICT"
GESTURE_NONE    = "NONE"

# ─────────────────────────────────────────────
#  Model / Prediction
# ─────────────────────────────────────────────
IMG_SIZE      = 28          # CNN input size (28×28)
NUM_CLASSES   = 36          # 26 letters + 10 digits  (or 26 for letters only)
TOP_K         = 3           # Top-K predictions to show

# Minimum confidence threshold to accept a prediction
CONFIDENCE_THRESHOLD = 0.30

# ─────────────────────────────────────────────
#  Voice
# ─────────────────────────────────────────────
VOICE_ENABLED = True
VOICE_RATE    = 150         # Words per minute

# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
APP_TITLE      = "AirWrite AI"
APP_SUBTITLE   = "Write in the Air. Recognize with AI."
WINDOW_WIDTH   = 1280
WINDOW_HEIGHT  = 780
THEME          = "dark"     # CustomTkinter theme

# Overlay text
FONT_SCALE   = 0.7
FONT_THICK   = 2

# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
TRAIN_EPOCHS      = 25
TRAIN_BATCH_SIZE  = 128
TRAIN_VAL_SPLIT   = 0.10
TRAIN_IMG_SIZE    = 28

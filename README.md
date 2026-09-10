# ✍️ AirWrite AI

> **Write in the Air. Recognize with AI.**

A real-time gesture-based air writing recognition system that uses your webcam, MediaPipe hand tracking, and a Convolutional Neural Network to recognize characters you draw in the air with your index finger.

---

## 🎯 Features

- **Real-time Hand Tracking** – MediaPipe detects your hand and highlights the index fingertip
- **Air Drawing** – Trace characters in the air with just your index finger
- **Gesture Controls** – Natural hand gestures for Draw, Pause, Clear, and Predict
- **CNN Recognition** – Deep learning model predicts A-Z characters
- **Top-3 Predictions** – See the top 3 candidates with confidence scores
- **Voice Output** – Optional TTS announces your predicted character
- **Modern GUI** – Dark-themed CustomTkinter interface with live camera and canvas preview
- **Save Drawings** – Export your air-written characters as PNG images

---

## 🖥️ Screenshots

*(Run the app and try it yourself!)*

---

## 🛠️ Technologies

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core language |
| OpenCV | Webcam capture, image processing |
| MediaPipe | Real-time hand tracking |
| TensorFlow / Keras | CNN model training & inference |
| NumPy | Numerical operations |
| Pillow | Image manipulation, synthetic data |
| CustomTkinter | Modern GUI framework |
| pyttsx3 | Text-to-speech voice output |

---

## 📁 Project Structure

```
AirWrite-AI/
├── main.py                 # Application entry point
├── app.py                  # CustomTkinter GUI application
├── config.py               # Central configuration
├── train_model.py          # CNN training script
├── requirements.txt        # Dependencies
│
├── hand_tracking/
│   ├── __init__.py
│   └── hand_detector.py    # MediaPipe hand detection wrapper
│
├── drawing/
│   ├── __init__.py
│   ├── air_canvas.py       # Virtual drawing canvas
│   └── gestures.py         # Gesture classification
│
├── recognition/
│   ├── __init__.py
│   ├── preprocess.py       # Image preprocessing pipeline
│   ├── predictor.py        # Model inference
│   └── labels.py           # Character label mapping
│
├── model/
│   ├── README.md           # Model training instructions
│   └── airwrite_model.h5   # Trained model (generate with train_model.py)
│
├── utils/
│   ├── __init__.py
│   └── helpers.py          # Shared utility functions
│
├── assets/                 # Icons and images
└── screenshots/            # Saved air drawings
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/AirWrite-AI.git
cd AirWrite-AI
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

> The training script will automatically download the EMNIST dataset, or generate synthetic training data from system fonts if the dataset is unavailable.

### 5. Run the application

```bash
python main.py
```

---

## 🎮 Controls

### Gesture Controls

| Gesture | Action | Description |
|---------|--------|-------------|
| ☝️ Index finger up | **DRAW** | Draw on the virtual canvas |
| ✌️ Two fingers up | **PAUSE** | Pause drawing (lift pen) |
| ✊ Closed fist | **CLEAR** | Clear the entire canvas |
| 👍 Thumb up | **PREDICT** | Run recognition on canvas |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `C` | Clear canvas |
| `P` | Predict character |
| `Q` | Quit application |

---

## 🤖 How It Works

```
Webcam Feed
     ↓
MediaPipe Hand Detection
     ↓
Index Finger Tip (Landmark #8)
     ↓
Coordinate Tracking
     ↓
Draw Lines on Virtual Canvas
     ↓
Gesture → PREDICT triggered
     ↓
Grayscale → Bounding Box Crop → Pad → Resize 28×28 → Normalize
     ↓
CNN Model Inference
     ↓
Top-3 Predictions + Confidence
     ↓
Display + Voice Output
```

---

## 🧠 CNN Architecture

```
Input (28×28×1)
  → Conv2D(32) + BatchNorm + ReLU → MaxPool
  → Conv2D(64) + BatchNorm + ReLU → MaxPool
  → Conv2D(128) + BatchNorm + ReLU
  → GlobalAveragePooling
  → Dense(256) + Dropout(0.5)
  → Dense(26) + Softmax
```

---

## 📊 Training

```bash
# Default (25 epochs, A-Z only)
python train_model.py

# Custom settings
python train_model.py --epochs 40 --batch-size 64
```

| Dataset | Expected Accuracy |
|---------|------------------|
| EMNIST Letters (real handwriting) | ~90–95% |
| Synthetic (system fonts) | ~85–90% |

---

## 🔧 Configuration

Edit [`config.py`](config.py) to customise:

- Canvas size and drawing thickness
- Webcam resolution
- MediaPipe confidence thresholds
- Voice speech rate
- CNN input size and number of classes
- Training hyperparameters

---

## 🔮 Future Improvements

- [ ] Word recognition (sequence of characters)
- [ ] Sentence recognition
- [ ] Multi-language support
- [ ] Digit recognition (0–9) and mixed mode
- [ ] Real-time drawing smoothing improvements
- [ ] Web version (Flask + WebSockets)
- [ ] Mobile app (Android/iOS)
- [ ] Custom gesture training
- [ ] Export to GIF

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

Built with ❤️ using Python, OpenCV, MediaPipe, and TensorFlow.

*Suitable as a final year engineering project and portfolio showcase.*

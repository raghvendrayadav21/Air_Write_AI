"""
main.py – AirWrite AI Entry Point

Usage:
    python main.py

Keyboard shortcuts (active when camera window is focused):
    C  →  Clear canvas
    P  →  Predict
    Q  →  Quit
"""

import os
import sys

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



def check_dependencies() -> bool:
    """Verify that critical packages are importable."""
    missing = []
    required = {
        "cv2":          "opencv-python",
        "mediapipe":    "mediapipe",
        "numpy":        "numpy",
        "customtkinter":"customtkinter",
        "PIL":          "Pillow",
    }
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("=" * 60)
        print("  AirWrite AI – Missing Dependencies")
        print("=" * 60)
        print("  Please install the following packages:\n")
        for pkg in missing:
            print(f"    pip install {pkg}")
        print()
        print("  Or install all at once:")
        print("    pip install -r requirements.txt")
        print("=" * 60)
        return False
    return True


def main() -> None:
    # ── Ensure we're running from the project root ────────────────────────────
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # ── Dependency check ──────────────────────────────────────────────────────
    if not check_dependencies():
        sys.exit(1)

    # ── Create required directories ───────────────────────────────────────────
    import config
    os.makedirs(config.MODEL_DIR,   exist_ok=True)
    os.makedirs(config.ASSETS_DIR,  exist_ok=True)
    os.makedirs(config.SCREENSHOTS, exist_ok=True)

    # ── Model check (non-fatal) ────────────────────────────────────────────────
    if not os.path.isfile(config.MODEL_PATH):
        print()
        print("  ⚠  WARNING: Trained model not found at:")
        print(f"     {config.MODEL_PATH}")
        print()
        print("  The application will start, but prediction will be unavailable.")
        print("  To train the model, run:")
        print("     python train_model.py")
        print()

    # ── Launch GUI ─────────────────────────────────────────────────────────────
    from app import AirWriteApp
    app = AirWriteApp()
    app.mainloop()


if __name__ == "__main__":
    main()

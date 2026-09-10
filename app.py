"""
app.py

Main AirWrite AI Application – built with CustomTkinter.

Layout:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Header: AIRWRITE AI                                             │
  ├─────────────────────────────┬────────────────────────────────────┤
  │  Left: Live Camera Feed     │  Right: Virtual Canvas + Results   │
  │                             │                                    │
  │  (hand landmarks, gesture,  │  (air drawing preview, prediction  │
  │   finger tip highlight)     │   card, top-3 list, controls)      │
  ├─────────────────────────────┴────────────────────────────────────┤
  │  Bottom: Status bar                                              │
  └──────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import messagebox
from typing import Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

import config
from drawing.air_canvas import AirCanvas
from drawing.gestures import classify as classify_gesture, gesture_label
from hand_tracking.hand_detector import HandDetector
from recognition.predictor import Predictor
from recognition.preprocess import preprocess, get_preview, segment_characters, annotate_word_canvas
from utils.helpers import (
    resize_frame,
    draw_text_with_bg,
    overlay_canvas_on_frame,
    confidence_color,
)


# ─── CustomTkinter theme ──────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AirWriteApp(ctk.CTk):
    """Main application window."""

    # Poll interval for the webcam update loop (milliseconds)
    _POLL_MS = 15

    def __init__(self) -> None:
        super().__init__()

        # ── Window setup ──────────────────────────────────────────────────────
        self.title(f"{config.APP_TITLE}  –  {config.APP_SUBTITLE}")
        self.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.minsize(900, 600)
        self.configure(fg_color="#0D1117")

        # ── Core components ───────────────────────────────────────────────────
        self._cap:      Optional[cv2.VideoCapture] = None
        self._detector  = HandDetector()
        self._canvas    = AirCanvas()
        self._predictor = Predictor()

        # ── Application state ─────────────────────────────────────────────────
        self._running       = False
        self._current_gesture = config.GESTURE_NONE
        self._last_gesture    = config.GESTURE_NONE
        self._clear_cooldown  = 0         # frames to ignore CLEAR after trigger
        self._predict_cooldown= 0
        self._voice_enabled   = config.VOICE_ENABLED
        self._voice_engine    = None
        self._last_prediction: Optional[str]   = None
        self._last_confidence: Optional[float] = None
        self._top_k_results:   list            = []
        self._annotated_canvas: Optional[np.ndarray] = None

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_header()
        self._build_main_area()
        self._build_status_bar()

        # ── Key bindings ──────────────────────────────────────────────────────
        self.bind("<KeyPress-c>", lambda _: self._clear_canvas())
        self.bind("<KeyPress-C>", lambda _: self._clear_canvas())
        self.bind("<KeyPress-p>", lambda _: self._trigger_predict())
        self.bind("<KeyPress-P>", lambda _: self._trigger_predict())
        self.bind("<KeyPress-q>", lambda _: self._quit())
        self.bind("<KeyPress-Q>", lambda _: self._quit())
        self.protocol("WM_DELETE_WINDOW", self._quit)

        # ── Init voice engine in background ───────────────────────────────────
        threading.Thread(target=self._init_voice, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────────
    #  UI Construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_header(self) -> None:
        """Top banner with title and subtitle."""
        header = ctk.CTkFrame(self, fg_color="#161B22", corner_radius=0, height=70)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="✍️  AIRWRITE AI",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color="#58A6FF",
        ).pack(side="left", padx=24, pady=10)

        ctk.CTkLabel(
            header,
            text="Write in the Air. Recognize with AI.",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#8B949E",
        ).pack(side="left", padx=0, pady=10)

        # Keyboard shortcuts hint
        ctk.CTkLabel(
            header,
            text="  C = Clear   P = Predict   Q = Quit",
            font=ctk.CTkFont(family="Segoe UI Mono", size=11),
            text_color="#484F58",
        ).pack(side="right", padx=20)

    def _build_main_area(self) -> None:
        """Two-column main content area."""
        main = ctk.CTkFrame(self, fg_color="#0D1117", corner_radius=0)
        main.pack(fill="both", expand=True, padx=0, pady=0)

        main.columnconfigure(0, weight=1)   # Left column takes dynamic space
        main.columnconfigure(1, weight=0, minsize=400)   # Right column dedicated width
        main.rowconfigure(0, weight=1)

        self._build_left_panel(main)
        self._build_right_panel(main)

    # ── Left panel: Live camera ────────────────────────────────────────────────
    def _build_left_panel(self, parent: ctk.CTkFrame) -> None:
        left = ctk.CTkFrame(parent, fg_color="#161B22", corner_radius=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)

        # Section label
        ctk.CTkLabel(
            left,
            text="📷  LIVE CAMERA",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#8B949E",
        ).pack(anchor="w", padx=14, pady=(10, 4))

        # Camera container - fixed propagation prevents positive feedback loop
        self._cam_container = ctk.CTkFrame(left, fg_color="#0D1117", corner_radius=8)
        self._cam_container.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self._cam_container.pack_propagate(False)

        self._cam_label = ctk.CTkLabel(self._cam_container, text="")
        self._cam_label.place(relx=0.5, rely=0.5, anchor="center")

        # Gesture + mode row
        info_row = ctk.CTkFrame(left, fg_color="#0D1117", corner_radius=8, height=42)
        info_row.pack(fill="x", padx=10, pady=(0, 6))
        info_row.pack_propagate(False)

        self._gesture_label = ctk.CTkLabel(
            info_row,
            text="Gesture: —",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#E6EDF3",
        )
        self._gesture_label.pack(side="left", padx=14)

        self._mode_label = ctk.CTkLabel(
            info_row,
            text="Mode: IDLE",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#F0883E",
        )
        self._mode_label.pack(side="right", padx=14)

        # Status indicators
        status_row = ctk.CTkFrame(left, fg_color="#0D1117", corner_radius=8, height=34)
        status_row.pack(fill="x", padx=10, pady=(0, 6))
        status_row.pack_propagate(False)

        self._cam_status = ctk.CTkLabel(
            status_row,
            text="⬤  Camera: OFF",
            font=ctk.CTkFont(size=12),
            text_color="#EF5350",
        )
        self._cam_status.pack(side="left", padx=14)

        self._hand_status = ctk.CTkLabel(
            status_row,
            text="⬤  Hand: Not Detected",
            font=ctk.CTkFont(size=12),
            text_color="#8B949E",
        )
        self._hand_status.pack(side="left", padx=10)

        # Start camera button
        self._start_btn = ctk.CTkButton(
            left,
            text="▶  Start Camera",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#238636",
            hover_color="#2EA043",
            height=40,
            corner_radius=8,
            command=self._toggle_camera,
        )
        self._start_btn.pack(fill="x", padx=10, pady=(0, 10))

    # ── Right panel: Canvas + Results + Controls ────────────────────────────────
    def _build_right_panel(self, parent: ctk.CTkFrame) -> None:
        right = ctk.CTkScrollableFrame(parent, fg_color="#161B22", corner_radius=12, width=390)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)

        # ── Canvas preview ─────────────────────────────────────────────────────
        ctk.CTkLabel(
            right,
            text="✏️  AIR CANVAS",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#8B949E",
        ).pack(anchor="w", padx=14, pady=(10, 4))

        canvas_frame = ctk.CTkFrame(right, fg_color="#000000", corner_radius=8)
        canvas_frame.pack(fill="x", padx=10, pady=(0, 4))

        self._canvas_label = ctk.CTkLabel(canvas_frame, text="", corner_radius=0)
        self._canvas_label.pack(padx=4, pady=4)

        # ── Word Writing Guidance Tip ──────────────────────────────────────────
        ctk.CTkLabel(
            right,
            text="💡 Tip: Write words in CAPITAL letters (e.g. LION, CAT)\nLeave small space between letters.",
            font=ctk.CTkFont(size=11),
            text_color="#58A6FF",
            justify="center",
        ).pack(fill="x", padx=10, pady=(0, 8))

        # ── Prediction card ────────────────────────────────────────────────────
        pred_card = ctk.CTkFrame(right, fg_color="#21262D", corner_radius=10)
        pred_card.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkLabel(
            pred_card,
            text="PREDICTION",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#484F58",
        ).pack(pady=(10, 0))

        self._pred_char_label = ctk.CTkLabel(
            pred_card,
            text="—",
            font=ctk.CTkFont(family="Segoe UI", size=72, weight="bold"),
            text_color="#58A6FF",
        )
        self._pred_char_label.pack(pady=0)

        self._pred_conf_label = ctk.CTkLabel(
            pred_card,
            text="Confidence: —",
            font=ctk.CTkFont(size=14),
            text_color="#8B949E",
        )
        self._pred_conf_label.pack(pady=(0, 6))

        # ── Top-3 list ─────────────────────────────────────────────────────────
        self._topk_frame = ctk.CTkFrame(pred_card, fg_color="#161B22", corner_radius=6)
        self._topk_frame.pack(fill="x", padx=10, pady=(0, 10))

        self._topk_title = ctk.CTkLabel(
            self._topk_frame,
            text="Breakdown / Top Predictions",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#484F58",
        )
        self._topk_title.pack(pady=(6, 2))

        self._topk_labels = []
        for i in range(6):
            lbl = ctk.CTkLabel(
                self._topk_frame,
                text=f"{i+1}.  —",
                font=ctk.CTkFont(family="Segoe UI Mono", size=12),
                text_color="#8B949E",
            )
            lbl.pack(anchor="w", padx=14, pady=1)
            self._topk_labels.append(lbl)

        ctk.CTkLabel(self._topk_frame, text="").pack(pady=2)  # spacer

        # ── Control buttons ────────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(right, fg_color="#0D1117", corner_radius=8)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))

        btn_cfg = dict(height=36, corner_radius=6, font=ctk.CTkFont(size=12))

        ctk.CTkButton(
            btn_frame, text="🗑  Clear Canvas", fg_color="#21262D",
            hover_color="#30363D", **btn_cfg,
            command=self._clear_canvas,
        ).pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkButton(
            btn_frame, text="🔮  Predict  (P)", fg_color="#1F6FEB",
            hover_color="#388BFD", **btn_cfg,
            command=self._trigger_predict,
        ).pack(fill="x", padx=8, pady=4)

        ctk.CTkButton(
            btn_frame, text="💾  Save Drawing", fg_color="#21262D",
            hover_color="#30363D", **btn_cfg,
            command=self._save_drawing,
        ).pack(fill="x", padx=8, pady=4)

        self._voice_btn = ctk.CTkButton(
            btn_frame,
            text=f"🔊  Voice: {'ON' if self._voice_enabled else 'OFF'}",
            fg_color="#21262D" if not self._voice_enabled else "#0D419D",
            hover_color="#30363D",
            **btn_cfg,
            command=self._toggle_voice,
        )
        self._voice_btn.pack(fill="x", padx=8, pady=(4, 8))

    def _build_status_bar(self) -> None:
        """Bottom status bar."""
        bar = ctk.CTkFrame(self, fg_color="#161B22", corner_radius=0, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._status_label = ctk.CTkLabel(
            bar,
            text="Ready.  Open webcam to begin.",
            font=ctk.CTkFont(size=11),
            text_color="#484F58",
        )
        self._status_label.pack(side="left", padx=14)

        if not self._predictor.is_ready:
            ctk.CTkLabel(
                bar,
                text="⚠  Model not found – run train_model.py first",
                font=ctk.CTkFont(size=11),
                text_color="#F0883E",
            ).pack(side="right", padx=14)
        else:
            ctk.CTkLabel(
                bar,
                text="✓  Model loaded",
                font=ctk.CTkFont(size=11),
                text_color="#3FB950",
            ).pack(side="right", padx=14)

    # ──────────────────────────────────────────────────────────────────────────
    #  Camera + Main Loop
    # ──────────────────────────────────────────────────────────────────────────

    def _toggle_camera(self) -> None:
        """Toggle webcam on/off."""
        if self._running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self) -> None:
        """Open webcam and start the update loop."""
        self._cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        if not self._cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Camera not detected.\nPlease connect a webcam and try again.",
            )
            self._cap = None
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.WEBCAM_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)

        self._running = True
        self._start_btn.configure(text="⏹  Stop Camera", fg_color="#B91C1C", hover_color="#DC2626")
        self._cam_status.configure(text="⬤  Camera: ON", text_color="#3FB950")
        self._set_status("Camera started.  Raise your index finger to draw.")
        self._update_loop()

    def _stop_camera(self) -> None:
        """Release webcam resources."""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None

        self._start_btn.configure(text="▶  Start Camera", fg_color="#238636", hover_color="#2EA043")
        self._cam_status.configure(text="⬤  Camera: OFF", text_color="#EF5350")
        self._hand_status.configure(text="⬤  Hand: Not Detected", text_color="#8B949E")
        self._gesture_label.configure(text="Gesture: —")
        self._mode_label.configure(text="Mode: IDLE")
        self._set_status("Camera stopped.")

        # Clear camera display
        blank = np.zeros((config.WEBCAM_HEIGHT, config.WEBCAM_WIDTH, 3), dtype=np.uint8)
        self._update_cam_display(blank)

    def _update_loop(self) -> None:
        """Main per-frame update. Scheduled with after() for Tkinter compatibility."""
        if not self._running or self._cap is None:
            return

        try:
            ret, frame = self._cap.read()
            if not ret:
                self._set_status("⚠  Could not read frame from camera.")
                if self._running:
                    self.after(self._POLL_MS, self._update_loop)
                return

            # Mirror
            if config.MIRROR_FEED:
                frame = cv2.flip(frame, 1)

            # ── Hand detection ────────────────────────────────────────────────────
            frame = self._detector.find_hands(frame, draw=True)
            hand_found = self._detector.hand_detected()

            # ── Update hand status ────────────────────────────────────────────────
            if hand_found:
                self._hand_status.configure(text="⬤  Hand: Detected", text_color="#3FB950")
            else:
                self._hand_status.configure(text="⬤  Hand: Not Detected", text_color="#8B949E")
                self._canvas.lift_pen()
                self._gesture_label.configure(text="Gesture: —")
                self._mode_label.configure(text="Mode: IDLE", text_color="#8B949E")
                self._update_cam_display(frame)
                self._update_canvas_display()
                if self._running:
                    self.after(self._POLL_MS, self._update_loop)
                return

            # ── Gesture classification ────────────────────────────────────────────
            fingers = self._detector.fingers_up()
            gesture = classify_gesture(fingers)
            self._current_gesture = gesture
            self._gesture_label.configure(text=f"Gesture: {gesture_label(gesture)}")

            # ── Cooldown timers ───────────────────────────────────────────────────
            if self._clear_cooldown > 0:
                self._clear_cooldown -= 1
            if self._predict_cooldown > 0:
                self._predict_cooldown -= 1

            # ── Act on gesture ────────────────────────────────────────────────────
            tip = self._detector.get_index_tip()

            if gesture == config.GESTURE_DRAW and tip:
                self._annotated_canvas = None
                self._mode_label.configure(text="Mode: DRAW ✏️", text_color="#58A6FF")
                self._canvas.draw(tip)
                self._set_status("Drawing…")

            elif gesture == config.GESTURE_PAUSE:
                self._mode_label.configure(text="Mode: PAUSE ✌️", text_color="#F0883E")
                self._canvas.lift_pen()
                self._set_status("Paused.  Lower your middle finger to resume drawing.")

            elif gesture == config.GESTURE_CLEAR and self._clear_cooldown == 0:
                self._annotated_canvas = None
                self._mode_label.configure(text="Mode: CLEAR ✊", text_color="#EF5350")
                self._canvas.clear()
                self._clear_cooldown = 30   # ~0.5 s at 60 fps
                self._set_status("Canvas cleared.")

            elif gesture == config.GESTURE_PREDICT and self._predict_cooldown == 0:
                self._mode_label.configure(text="Mode: PREDICT 👍", text_color="#3FB950")
                self._canvas.lift_pen()
                self._run_prediction(show_dialog=False)
                self._predict_cooldown = 60  # ~1 s

            else:
                self._canvas.lift_pen()

            # ── Overlay canvas on camera frame ────────────────────────────────────
            canvas_img = self._canvas.get_image()
            blended = overlay_canvas_on_frame(frame, canvas_img, alpha=0.55)

            # ── Draw coordinate HUD ───────────────────────────────────────────────
            if tip:
                draw_text_with_bg(
                    blended,
                    f"Finger: ({tip[0]}, {tip[1]})",
                    (10, 24),
                    font_color=(0, 255, 255),
                    bg_color=(0, 0, 0),
                )

            self._update_cam_display(blended)
            self._update_canvas_display()

        except Exception as exc:
            self._set_status(f"Tracking error: {exc}")

        if self._running:
            self.after(self._POLL_MS, self._update_loop)

    # ──────────────────────────────────────────────────────────────────────────
    #  Display helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _update_cam_display(self, frame: np.ndarray) -> None:
        """Push a BGR frame to the camera label widget fitted to container."""
        cw = self._cam_container.winfo_width()
        ch = self._cam_container.winfo_height()

        if cw < 50 or ch < 50:
            cw, ch = 640, 480

        target_w = cw - 8
        target_h = ch - 8

        aspect = config.WEBCAM_WIDTH / config.WEBCAM_HEIGHT  # 640 / 480 = 1.3333
        if (target_w / target_h) > aspect:
            h = max(int(target_h), 120)
            w = max(int(h * aspect), 160)
        else:
            w = max(int(target_w), 160)
            h = max(int(w / aspect), 120)

        resized  = cv2.resize(frame, (w, h))
        rgb      = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(rgb)
        ctk_img  = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))

        self._cam_label.configure(image=ctk_img, text="")
        self._cam_label.image = ctk_img  # prevent GC

    def _update_canvas_display(self) -> None:
        """Push the current air canvas or persistent annotation to the canvas label widget."""
        canvas_img = self._annotated_canvas if self._annotated_canvas is not None else self._canvas.get_image()
        size = 200
        resized  = cv2.resize(canvas_img, (size, size))
        rgb      = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(rgb)
        ctk_img  = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(size, size))

        self._canvas_label.configure(image=ctk_img, text="")
        self._canvas_label.image = ctk_img

    # ──────────────────────────────────────────────────────────────────────────
    #  Prediction
    # ──────────────────────────────────────────────────────────────────────────

    def _trigger_predict(self, *_) -> None:
        """Manual prediction trigger (button or key)."""
        self._canvas.lift_pen()
        self._run_prediction(show_dialog=True)

    def _run_prediction(self, show_dialog: bool = False) -> None:
        """Preprocess canvas, segment into letters, and run model inference."""
        if self._canvas.is_empty():
            self._set_status("⚠  Canvas is empty.  Please write something first.")
            if show_dialog:
                messagebox.showinfo("AirWrite AI", "Canvas is empty.\nPlease write a character first.")
            return

        if not self._predictor.is_ready:
            err = self._predictor.error_message or "Model not loaded."
            self._set_status(f"⚠  {err}")
            if show_dialog:
                messagebox.showwarning("Model Not Found", err)
            return

        canvas_bgr = self._canvas.get_image()

        # Segment characters left-to-right
        segments = segment_characters(canvas_bgr)
        if not segments:
            # Fallback to single character crop
            t = preprocess(canvas_bgr)
            if t is not None:
                segments = [(t, (0, 0, canvas_bgr.shape[1], canvas_bgr.shape[0]))]

        if not segments:
            self._set_status("⚠  Could not detect any characters on canvas.")
            return

        tensors = [s[0] for s in segments]
        boxes   = [s[1] for s in segments]

        try:
            if len(segments) == 1:
                label, conf, top_k = self._predictor.predict(tensors[0])
                word = label
                avg_conf = conf
                char_boxes = [(label, conf, boxes[0])]
                suggested = None
                self._topk_title.configure(text="Top Predictions")
            else:
                word, avg_conf, char_results, suggested = self._predictor.predict_word(tensors)
                char_boxes = [(cr[0], cr[1], box) for cr, box in zip(char_results, boxes)]
                self._topk_title.configure(text=f"Letters Breakdown ({len(word)})")
        except Exception as exc:
            self._set_status(f"⚠  Prediction failed: {exc}")
            return

        display_word = suggested if (suggested and suggested != word and avg_conf < 95) else word
        self._last_prediction = display_word
        self._last_confidence = avg_conf

        # ── Update prediction UI ──────────────────────────────────────────────
        col = confidence_color(avg_conf)
        font_size = 64 if len(display_word) <= 2 else (46 if len(display_word) <= 4 else 34)
        self._pred_char_label.configure(
            text=display_word,
            text_color=col,
            font=ctk.CTkFont(family="Segoe UI", size=font_size, weight="bold"),
        )
        conf_text = f"Confidence: {avg_conf:.1f}% ({len(segments)} letter{'s' if len(segments) > 1 else ''})"
        if suggested and suggested != word:
            conf_text += f" • Suggestion: {suggested}"
        self._pred_conf_label.configure(
            text=conf_text,
            text_color=col,
        )

        if len(segments) > 1:
            for i in range(len(self._topk_labels)):
                if i < len(char_boxes):
                    lbl, c, _ = char_boxes[i]
                    bar = "█" * max(1, int(c / 10))
                    self._topk_labels[i].configure(
                        text=f"Letter {i+1}:  {lbl}  –  {c:.1f}%  {bar}",
                        text_color=confidence_color(c),
                    )
                else:
                    self._topk_labels[i].configure(text="", text_color="#8B949E")
        else:
            for i in range(len(self._topk_labels)):
                if i < len(top_k):
                    lbl, c = top_k[i]
                    bar = "█" * max(1, int(c / 10))
                    self._topk_labels[i].configure(
                        text=f"{i+1}.  {lbl}  –  {c:.1f}%  {bar}",
                        text_color=confidence_color(c),
                    )
                else:
                    self._topk_labels[i].configure(text="", text_color="#8B949E")

        self._set_status(f"Recognized Word: '{display_word}'  ({avg_conf:.1f}% confidence)")

        # ── Show bounding boxes on canvas preview ──────────────────────────────
        self._annotated_canvas = annotate_word_canvas(canvas_bgr, char_boxes)
        self._update_canvas_display()

        # ── Voice output ──────────────────────────────────────────────────────
        if self._voice_enabled and self._voice_engine:
            speech = f"You wrote {display_word}" if len(display_word) == 1 else f"Word recognized: {display_word}"
            threading.Thread(
                target=self._speak,
                args=(speech,),
                daemon=True,
            ).start()

    # ──────────────────────────────────────────────────────────────────────────
    #  Controls
    # ──────────────────────────────────────────────────────────────────────────

    def _clear_canvas(self, *_) -> None:
        self._annotated_canvas = None
        self._canvas.clear()
        self._pred_char_label.configure(
            text="—",
            text_color="#58A6FF",
            font=ctk.CTkFont(family="Segoe UI", size=72, weight="bold"),
        )
        self._pred_conf_label.configure(text="Confidence: —", text_color="#8B949E")
        self._topk_title.configure(text="Breakdown / Top Predictions")
        for lbl in self._topk_labels:
            lbl.configure(text="—", text_color="#8B949E")
        self._set_status("Canvas cleared.")
        self._update_canvas_display()

    def _save_drawing(self) -> None:
        if self._canvas.is_empty():
            messagebox.showinfo("AirWrite AI", "Nothing to save – canvas is empty.")
            return
        path = self._canvas.save()
        self._set_status(f"Drawing saved → {path}")
        messagebox.showinfo("Saved", f"Drawing saved to:\n{path}")

    def _toggle_voice(self) -> None:
        self._voice_enabled = not self._voice_enabled
        state = "ON" if self._voice_enabled else "OFF"
        self._voice_btn.configure(
            text=f"🔊  Voice: {state}",
            fg_color="#0D419D" if self._voice_enabled else "#21262D",
        )


    # ──────────────────────────────────────────────────────────────────────────
    #  Voice
    # ──────────────────────────────────────────────────────────────────────────

    def _init_voice(self) -> None:
        """Initialise pyttsx3 in a background thread (avoids blocking the UI)."""
        try:
            import pyttsx3
            self._voice_engine = pyttsx3.init()
            self._voice_engine.setProperty("rate", config.VOICE_RATE)
        except Exception:
            self._voice_engine = None  # Voice not available

    def _speak(self, text: str) -> None:
        """Run TTS synchronously in the calling thread."""
        try:
            if self._voice_engine:
                self._voice_engine.say(text)
                self._voice_engine.runAndWait()
        except Exception:
            pass  # Silently ignore TTS errors

    # ──────────────────────────────────────────────────────────────────────────
    #  Status bar
    # ──────────────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self._status_label.configure(text=msg)

    # ──────────────────────────────────────────────────────────────────────────
    #  Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def _quit(self, *_) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._cap:
            self._cap.release()
        self._detector.close()
        self.destroy()

/**
 * app.js – AirWrite AI Web Edition
 *
 * Responsibilities:
 *  1. Initialise MediaPipe Hands (JS) via CDN
 *  2. Stream webcam to a canvas with hand landmarks drawn
 *  3. Classify gesture from landmark positions (mirrors Python gestures.py logic)
 *  4. Drive the air-writing canvas:
 *       DRAW    → draw strokes
 *       PAUSE   → lift pen (no draw)
 *       CLEAR   → wipe canvas
 *       PREDICT → export canvas as base64 → POST /api/predict → show results
 *  5. Update UI (gesture badge, confidence bar, history, status bar)
 */

'use strict';

// ── Constants (mirrors config.py) ────────────────────────────────────────────
const GESTURE = { DRAW: 'draw', PAUSE: 'pause', PREDICT: 'predict', CLEAR: 'clear', NONE: 'none' };
const GESTURE_EMOJI = {
  draw:    '☝️  DRAW',
  pause:   '✌️  PAUSE',
  predict: '👍  PREDICT',
  clear:   '✊  CLEAR',
  none:    '🤚  NONE',
};

// MediaPipe landmark indices
const THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_TIP = 12, RING_TIP = 16, PINKY_TIP = 20;
const THUMB_IP = 3, INDEX_PIP = 6, MIDDLE_PIP = 10, RING_PIP = 14, PINKY_PIP = 18;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const inputVideo    = document.getElementById('inputVideo');
const cameraCanvas  = document.getElementById('cameraCanvas');
const airCanvas     = document.getElementById('airCanvas');
const cameraOverlay = document.getElementById('cameraOverlay');
const gestureBadge  = document.getElementById('gestureBadge');
const canvasHint    = document.getElementById('canvasHint');

const resultIdle    = document.getElementById('resultIdle');
const resultContent = document.getElementById('resultContent');
const resultLoading = document.getElementById('resultLoading');
const resultError   = document.getElementById('resultError');
const resultCard    = document.getElementById('resultCard');

const predictedWord = document.getElementById('predictedWord');
const suggestedRow  = document.getElementById('suggestedRow');
const suggestedWord = document.getElementById('suggestedWord');
const confBar       = document.getElementById('confBar');
const confPct       = document.getElementById('confPct');
const errorMsg      = document.getElementById('errorMsg');

const charsList     = document.getElementById('charsList');
const top3List      = document.getElementById('top3List');
const historyList   = document.getElementById('historyList');
const charsSection  = document.getElementById('charsSection');
const top3Section   = document.getElementById('top3Section');

const modelStatusDot  = document.getElementById('statusDot');
const statusText      = document.getElementById('statusText');
const fpsDisplay      = document.getElementById('fps');
const handStatusEl    = document.getElementById('handStatus');
const canvasInfoEl    = document.getElementById('canvasInfo');

const btnClear        = document.getElementById('btnClear');
const btnPredict      = document.getElementById('btnPredict');
const btnClearHistory = document.getElementById('btnClearHistory');
const strokeSizeInput = document.getElementById('strokeSize');
const strokeSizeVal   = document.getElementById('strokeSizeVal');
const swatches        = document.querySelectorAll('.swatch');

// ── State ─────────────────────────────────────────────────────────────────────
const camCtx      = cameraCanvas.getContext('2d');
const drawCtx     = airCanvas.getContext('2d');

let prevPoint     = null;          // Last drawn point on air canvas
let smoothWindow  = [];            // Smoothing buffer
const SMOOTH_SIZE = 3;

let strokeColor   = '#ffffff';
let strokeWidth   = 10;
let currentGesture = GESTURE.NONE;
let gestureHeld   = false;         // True while predict/clear gesture held
let isPredicting  = false;         // API call in flight

let handLandmarks = null;          // Current frame landmarks (or null)
let handedness    = 'Right';       // Detected hand handedness

let fpsCounter    = 0;
let fpsTime       = performance.now();

let canvasHasStrokes = false;      // Whether the canvas has any drawing

const history = [];

// ── Utility ───────────────────────────────────────────────────────────────────

function showOnly(...elements) {
  [resultIdle, resultContent, resultLoading, resultError].forEach(el => el.classList.add('hidden'));
  elements.forEach(el => el.classList.remove('hidden'));
}

function setGestureBadge(gesture) {
  gestureBadge.textContent = GESTURE_EMOJI[gesture] || '🤚 NONE';
  gestureBadge.className = `gesture-badge ${gesture}`;
}

function hideCanvasHint() {
  canvasHint.classList.add('hidden');
}

// ── Gesture classification (mirrors drawing/gestures.py) ─────────────────────

function classifyGesture(lm, handSide) {
  if (!lm || lm.length < 21) return GESTURE.NONE;

  const thumbTip  = lm[THUMB_TIP];
  const thumbIp   = lm[THUMB_IP];
  const thumbMcp  = lm[2];
  const indexTip  = lm[INDEX_TIP];
  const indexPip  = lm[INDEX_PIP];
  const middleTip = lm[MIDDLE_TIP];
  const middlePip = lm[MIDDLE_PIP];
  const ringTip   = lm[RING_TIP];
  const ringPip   = lm[RING_PIP];
  const pinkyTip  = lm[PINKY_TIP];
  const pinkyPip  = lm[PINKY_PIP];

  // Finger up/down (normalized 0-1 coords, y increases downward)
  const thumbExtended = handSide === 'Right'
    ? thumbTip.x < thumbIp.x
    : thumbTip.x > thumbIp.x;
  const isThumbsUp = (thumbTip.y < thumbMcp.y) && (Math.abs(thumbTip.x - thumbMcp.x) < 0.12);
  const thumb  = (thumbExtended || isThumbsUp) ? 1 : 0;
  const index  = indexTip.y  < indexPip.y  ? 1 : 0;
  const middle = middleTip.y < middlePip.y ? 1 : 0;
  const ring   = ringTip.y   < ringPip.y   ? 1 : 0;
  const pinky  = pinkyTip.y  < pinkyPip.y  ? 1 : 0;

  // PREDICT: thumb only
  if (thumb === 1 && index === 0 && middle === 0 && ring === 0 && pinky === 0)
    return GESTURE.PREDICT;

  // DRAW: index only (thumb relaxed)
  if (index === 1 && middle === 0 && ring === 0 && pinky === 0)
    return GESTURE.DRAW;

  // PAUSE: index + middle
  if (index === 1 && middle === 1 && ring === 0 && pinky === 0)
    return GESTURE.PAUSE;

  // CLEAR: closed fist
  if (thumb === 0 && index === 0 && middle === 0 && ring === 0 && pinky === 0)
    return GESTURE.CLEAR;

  return GESTURE.NONE;
}

// ── Air Canvas drawing ────────────────────────────────────────────────────────

function getIndexTipCanvas(lm) {
  if (!lm || lm.length <= INDEX_TIP) return null;
  const tip = lm[INDEX_TIP];
  // MediaPipe gives normalized [0,1] coords; map to air canvas pixel space
  // Note: webcam is mirrored, so x is already flipped
  const x = tip.x * airCanvas.width;
  const y = tip.y * airCanvas.height;
  return { x, y };
}

function smoothPoint(pt) {
  smoothWindow.push(pt);
  if (smoothWindow.length > SMOOTH_SIZE) smoothWindow.shift();
  const sx = smoothWindow.reduce((a, p) => a + p.x, 0) / smoothWindow.length;
  const sy = smoothWindow.reduce((a, p) => a + p.y, 0) / smoothWindow.length;
  return { x: sx, y: sy };
}

function drawStroke(pt) {
  const smoothed = smoothPoint(pt);
  if (prevPoint) {
    drawCtx.beginPath();
    drawCtx.lineWidth   = strokeWidth;
    drawCtx.lineCap     = 'round';
    drawCtx.lineJoin    = 'round';
    drawCtx.strokeStyle = strokeColor;
    drawCtx.moveTo(prevPoint.x, prevPoint.y);
    drawCtx.lineTo(smoothed.x, smoothed.y);
    drawCtx.stroke();

    // Rounded cap
    drawCtx.beginPath();
    drawCtx.arc(smoothed.x, smoothed.y, strokeWidth / 2, 0, Math.PI * 2);
    drawCtx.fillStyle = strokeColor;
    drawCtx.fill();
  }
  prevPoint = smoothed;
  canvasHasStrokes = true;
  hideCanvasHint();
  updateCanvasInfo();
}

function liftPen() {
  prevPoint = null;
  smoothWindow = [];
}

function clearCanvas() {
  drawCtx.clearRect(0, 0, airCanvas.width, airCanvas.height);
  liftPen();
  canvasHasStrokes = false;
  showOnly(resultIdle);
  resultCard.classList.remove('has-result');
  charsList.innerHTML = '';
  top3List.innerHTML = '';
  charsSection.style.display = 'none';
  top3Section.style.display = 'none';
  canvasHint.classList.remove('hidden');
  updateCanvasInfo();
}

function updateCanvasInfo() {
  canvasInfoEl.textContent = canvasHasStrokes ? 'Canvas: has strokes' : 'Canvas: empty';
}

// ── API call ──────────────────────────────────────────────────────────────────

async function runPredict() {
  if (isPredicting) return;
  if (!canvasHasStrokes) {
    showOnly(resultError);
    errorMsg.textContent = 'Canvas is empty – draw something first!';
    return;
  }

  isPredicting = true;
  showOnly(resultLoading);

  const b64 = airCanvas.toDataURL('image/png');

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64 }),
    });

    const data = await resp.json();

    if (!resp.ok || !data.success) {
      throw new Error(data.error || 'Prediction failed');
    }

    displayResult(data);
  } catch (err) {
    showOnly(resultError);
    errorMsg.textContent = err.message || 'Network error';
  } finally {
    isPredicting = false;
  }
}

function displayResult(data) {
  showOnly(resultContent);
  resultCard.classList.add('has-result');

  predictedWord.textContent = data.word || '?';

  // Suggested word
  if (data.suggested && data.suggested !== data.word) {
    suggestedRow.classList.remove('hidden');
    suggestedWord.textContent = data.suggested;
  } else {
    suggestedRow.classList.add('hidden');
  }

  // Confidence bar
  const pct = Math.min(100, Math.max(0, data.confidence || 0));
  confBar.style.width = pct + '%';
  confPct.textContent = pct.toFixed(1) + '%';

  // Character breakdown
  charsList.innerHTML = '';
  if (data.characters && data.characters.length > 1) {
    charsSection.style.display = 'block';
    data.characters.forEach(({ char, confidence }) => {
      const chip = document.createElement('div');
      chip.className = 'char-chip';
      chip.innerHTML = `<span class="char-letter">${char}</span><span class="char-conf">${confidence.toFixed(0)}%</span>`;
      charsList.appendChild(chip);
    });
  } else {
    charsSection.style.display = 'none';
  }

  // Top-3 alternatives (single character mode)
  top3List.innerHTML = '';
  if (data.top3 && data.top3.length > 0) {
    top3Section.style.display = 'block';
    data.top3.forEach(({ label, confidence }, i) => {
      const item = document.createElement('div');
      item.className = 'top3-item';
      item.innerHTML = `
        <span class="top3-rank">#${i + 1}</span>
        <span class="top3-label">${label}</span>
        <span class="top3-conf">${confidence.toFixed(1)}%</span>
      `;
      top3List.appendChild(item);
    });
  } else {
    top3Section.style.display = 'none';
  }

  // Add to history
  addHistory(data.suggested || data.word);
}

function addHistory(word) {
  if (!word) return;
  history.unshift(word);
  if (history.length > 20) history.pop();
  renderHistory();
}

function renderHistory() {
  if (history.length === 0) {
    historyList.innerHTML = '<p class="history-empty">No predictions yet</p>';
    return;
  }
  historyList.innerHTML = '';
  history.forEach(w => {
    const chip = document.createElement('span');
    chip.className = 'history-chip';
    chip.textContent = w;
    historyList.appendChild(chip);
  });
}

// ── Camera overlay drawing ────────────────────────────────────────────────────

function drawCameraFrame(videoEl) {
  cameraCanvas.width  = videoEl.videoWidth  || 640;
  cameraCanvas.height = videoEl.videoHeight || 480;

  // Mirror the feed
  camCtx.save();
  camCtx.translate(cameraCanvas.width, 0);
  camCtx.scale(-1, 1);
  camCtx.drawImage(videoEl, 0, 0);
  camCtx.restore();

  if (handLandmarks) {
    drawHandLandmarks(handLandmarks);
  }

  // FPS
  fpsCounter++;
  const now = performance.now();
  if (now - fpsTime >= 1000) {
    fpsDisplay.textContent = `FPS: ${fpsCounter}`;
    fpsCounter = 0;
    fpsTime = now;
  }
}

function drawHandLandmarks(lm) {
  const W = cameraCanvas.width;
  const H = cameraCanvas.height;

  const connections = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],
  ];

  // In mirrored view, x is flipped: draw_x = W - lm.x * W
  const px = (lm) => W - lm.x * W;
  const py = (lm) => lm.y * H;

  // Connections
  camCtx.strokeStyle = 'rgba(0,245,255,0.6)';
  camCtx.lineWidth = 2;
  connections.forEach(([a, b]) => {
    camCtx.beginPath();
    camCtx.moveTo(px(lm[a]), py(lm[a]));
    camCtx.lineTo(px(lm[b]), py(lm[b]));
    camCtx.stroke();
  });

  // Joints
  lm.forEach((pt, i) => {
    const x = px(pt), y = py(pt);
    const r = i === INDEX_TIP ? 10 : 5;
    const color = i === INDEX_TIP ? '#00f5ff' : 'rgba(168,85,247,0.8)';
    camCtx.beginPath();
    camCtx.arc(x, y, r, 0, Math.PI * 2);
    camCtx.fillStyle = color;
    camCtx.fill();
    if (i === INDEX_TIP) {
      camCtx.beginPath();
      camCtx.arc(x, y, 16, 0, Math.PI * 2);
      camCtx.strokeStyle = 'rgba(0,245,255,0.5)';
      camCtx.lineWidth = 2;
      camCtx.stroke();
    }
  });
}

// ── MediaPipe Hands setup ────────────────────────────────────────────────────

function initMediaPipe() {
  const hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });

  hands.onResults(onHandResults);

  const camera = new Camera(inputVideo, {
    onFrame: async () => {
      await hands.send({ image: inputVideo });
      drawCameraFrame(inputVideo);
    },
    width: 640,
    height: 480,
  });

  camera.start()
    .then(() => {
      cameraOverlay.classList.add('hidden');
    })
    .catch((err) => {
      cameraOverlay.querySelector('p').textContent = '⚠️ Camera access denied';
      console.error('Camera error:', err);
    });
}

let predictGestureCooldown = false;
let clearGestureCooldown   = false;

function onHandResults(results) {
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    handLandmarks = results.multiHandLandmarks[0];

    // Handedness
    if (results.multiHandedness && results.multiHandedness.length > 0) {
      handedness = results.multiHandedness[0].classification[0].label;
    }

    const gesture = classifyGesture(handLandmarks, handedness);
    currentGesture = gesture;
    setGestureBadge(gesture);
    handStatusEl.textContent = `Hand: ${handedness} | ${gesture.toUpperCase()}`;

    // Execute gesture actions
    if (gesture === GESTURE.DRAW) {
      const tip = getIndexTipCanvas(handLandmarks);
      if (tip) drawStroke(tip);
      airCanvas.classList.add('drawing');
      clearGestureCooldown   = false;
      predictGestureCooldown = false;
    } else {
      liftPen();
      airCanvas.classList.remove('drawing');
    }

    if (gesture === GESTURE.PAUSE) {
      clearGestureCooldown   = false;
      predictGestureCooldown = false;
    }

    if (gesture === GESTURE.PREDICT && !predictGestureCooldown) {
      predictGestureCooldown = true;
      runPredict();
    }

    if (gesture === GESTURE.CLEAR && !clearGestureCooldown) {
      clearGestureCooldown = true;
      clearCanvas();
    }

    if (gesture !== GESTURE.PREDICT) predictGestureCooldown = false;
    if (gesture !== GESTURE.CLEAR)   clearGestureCooldown   = false;

  } else {
    // No hand
    handLandmarks = null;
    currentGesture = GESTURE.NONE;
    setGestureBadge(GESTURE.NONE);
    liftPen();
    airCanvas.classList.remove('drawing');
    handStatusEl.textContent = 'No hand detected';
  }
}

// ── Model health check ────────────────────────────────────────────────────────

async function checkModelHealth() {
  try {
    const resp = await fetch('/api/health');
    const data = await resp.json();
    if (data.model_loaded) {
      modelStatusDot.className = 'status-dot ready';
      statusText.textContent = 'Model ready';
    } else {
      modelStatusDot.className = 'status-dot error';
      statusText.textContent = 'Model not loaded';
    }
  } catch {
    modelStatusDot.className = 'status-dot error';
    statusText.textContent = 'Server offline';
  }
}

// ── UI event listeners ────────────────────────────────────────────────────────

btnClear.addEventListener('click', clearCanvas);
btnPredict.addEventListener('click', runPredict);
btnClearHistory.addEventListener('click', () => {
  history.length = 0;
  renderHistory();
});

strokeSizeInput.addEventListener('input', () => {
  strokeWidth = parseInt(strokeSizeInput.value, 10);
  strokeSizeVal.textContent = strokeWidth + 'px';
});

swatches.forEach(sw => {
  sw.addEventListener('click', () => {
    swatches.forEach(s => s.classList.remove('active'));
    sw.classList.add('active');
    strokeColor = sw.dataset.color;
  });
});

// Mouse / touch fallback drawing directly on canvas
let mouseDown = false;

airCanvas.addEventListener('mousedown', (e) => {
  mouseDown = true;
  hideCanvasHint();
  const rect = airCanvas.getBoundingClientRect();
  const scaleX = airCanvas.width  / rect.width;
  const scaleY = airCanvas.height / rect.height;
  const pt = {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top)  * scaleY,
  };
  drawStroke(pt);
});

airCanvas.addEventListener('mousemove', (e) => {
  if (!mouseDown) return;
  const rect = airCanvas.getBoundingClientRect();
  const scaleX = airCanvas.width  / rect.width;
  const scaleY = airCanvas.height / rect.height;
  drawStroke({
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top)  * scaleY,
  });
});

airCanvas.addEventListener('mouseup',    () => { mouseDown = false; liftPen(); });
airCanvas.addEventListener('mouseleave', () => { mouseDown = false; liftPen(); });

// Touch support
airCanvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  const rect = airCanvas.getBoundingClientRect();
  const touch = e.touches[0];
  const scaleX = airCanvas.width  / rect.width;
  const scaleY = airCanvas.height / rect.height;
  drawStroke({
    x: (touch.clientX - rect.left) * scaleX,
    y: (touch.clientY - rect.top)  * scaleY,
  });
});

airCanvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  const rect = airCanvas.getBoundingClientRect();
  const touch = e.touches[0];
  const scaleX = airCanvas.width  / rect.width;
  const scaleY = airCanvas.height / rect.height;
  drawStroke({
    x: (touch.clientX - rect.left) * scaleX,
    y: (touch.clientY - rect.top)  * scaleY,
  });
});

airCanvas.addEventListener('touchend', () => liftPen());

// ── Init ──────────────────────────────────────────────────────────────────────

// Black canvas background
drawCtx.fillStyle = '#000000';
drawCtx.fillRect(0, 0, airCanvas.width, airCanvas.height);

charsSection.style.display = 'none';
top3Section.style.display  = 'none';

checkModelHealth();
initMediaPipe();

console.log('🚀 AirWrite AI Web Edition ready');

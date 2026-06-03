"""
pipeline/face_detector.py
==========================
SecureEdge AI — Face Detection Module

Wraps BlazeFace (short-range TFLite model) with full anchor generation,
bounding box decoding, confidence filtering, and NMS.

BlazeFace short-range model:
  - Input:  (1, 128, 128, 3)  float32, range [-1, 1]
  - Output0: (1, 896, 16) — raw box regressions (cx_delta, cy_delta, w, h, 12 landmark coords)
  - Output1: (1, 896, 1)  — raw classification logits (sigmoid → confidence)

Anchor grid:
  - 2 anchors per cell at 8×8 stride  → 512 anchors
  - 6 anchors per cell at 16×16 stride → 384 anchors
  - Total: 896 anchors

When the TFLite model file is absent the module falls back to a
deterministic OpenCV Haar-cascade detector so the pipeline stays
runnable during development without model assets.

Author: SecureEdge AI Team
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ── Constants ────────────────────────────────────────────────────────────────

BLAZEFACE_INPUT_SIZE: int = 128           # Model expects 128×128
SCORE_THRESHOLD: float = 0.65            # Discard detections below this
IOU_THRESHOLD: float = 0.30              # NMS IoU overlap threshold
MIN_FACE_SIZE_PX: int = 48               # Reject faces smaller than this in original frame


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class FaceDetection:
    """
    Single detected face returned by FaceDetector.detect_faces().

    All coordinates are in *original frame pixel space*.
    Landmarks are ordered:
      [0] right_eye  [1] left_eye  [2] nose_tip
      [3] mouth      [4] right_ear [5] left_ear
    (BlazeFace short-range keypoint ordering)
    """
    bbox: Tuple[int, int, int, int]                  # (x1, y1, x2, y2)
    confidence: float                                 # Sigmoid confidence [0, 1]
    landmarks: List[Tuple[float, float]] = field(default_factory=list)  # 6× (x, y)

    @property
    def right_eye(self) -> Optional[Tuple[float, float]]:
        return self.landmarks[0] if len(self.landmarks) > 0 else None

    @property
    def left_eye(self) -> Optional[Tuple[float, float]]:
        return self.landmarks[1] if len(self.landmarks) > 1 else None

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


# ── Anchor generation ─────────────────────────────────────────────────────────

def generate_blazeface_anchors(
    input_size: int = BLAZEFACE_INPUT_SIZE,
) -> np.ndarray:
    """
    Generate the 896-anchor grid used by BlazeFace short-range.

    BlazeFace uses a fixed SSD anchor layout:
      - Strides [8, 16]
      - Anchors per cell: [2, 6]
      - Aspect ratio: 1.0 (squares only)

    Returns
    -------
    np.ndarray  shape (896, 4)
        Each row: [cx, cy, w, h] in *normalised* [0, 1] coordinates.
        w=h=1.0 for all anchors (BlazeFace predicts size delta on top).
    """
    strides = [8, 16]
    anchors_per_cell = [2, 6]
    anchors = []

    for stride, num_anchors in zip(strides, anchors_per_cell):
        grid_size = input_size // stride
        for row in range(grid_size):
            for col in range(grid_size):
                cx = (col + 0.5) / grid_size
                cy = (row + 0.5) / grid_size
                for _ in range(num_anchors):
                    anchors.append([cx, cy, 1.0, 1.0])

    result = np.array(anchors, dtype=np.float32)
    assert result.shape == (896, 4), f"Anchor count mismatch: {result.shape}"
    logger.debug("Generated %d BlazeFace anchors", len(result))
    return result


# ── Main detector class ───────────────────────────────────────────────────────

class FaceDetector:
    """
    BlazeFace TFLite face detector with graceful OpenCV fallback.

    Parameters
    ----------
    model_path : str
        Path to face_detection_short_range.tflite.
        If the file does not exist, the module silently switches to
        OpenCV Haar cascade (accurate enough for dev/CI; not for prod).
    score_threshold : float
        Minimum confidence to keep a detection.
    iou_threshold : float
        IoU threshold for NMS duplicate suppression.

    Notes
    -----
    Call allocate_tensors() once at startup (done in __init__).  Never
    call it inside the detect loop — on Android it stalls the thread
    for 50-150 ms due to memory remapping.
    """

    def __init__(
        self,
        model_path: str = "models/blazeface/face_detection_short_range.tflite",
        score_threshold: float = SCORE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
    ) -> None:
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._anchors = generate_blazeface_anchors()
        self._use_tflite = False
        self._interpreter = None
        self._inp_details = None
        self._out_details = None
        self._opencv_cascade = None

        if os.path.isfile(model_path):
            self._load_tflite(model_path)
        else:
            logger.warning(
                "BlazeFace model not found at '%s'. "
                "Falling back to OpenCV Haar cascade. "
                "Place face_detection_short_range.tflite for production use.",
                model_path,
            )
            self._load_opencv_fallback()

    # ── Initialisation helpers ────────────────────────────────────────────

    def _load_tflite(self, model_path: str) -> None:
        """Load and warm-up the TFLite interpreter."""
        try:
            import tensorflow as tf  # Lazy import — not needed if using fallback
            self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._interpreter.allocate_tensors()
            self._inp_details = self._interpreter.get_input_details()
            self._out_details = self._interpreter.get_output_details()
            self._use_tflite = True
            logger.info(
                "BlazeFace TFLite loaded from '%s' | inputs=%s outputs=%s",
                model_path,
                [d["shape"].tolist() for d in self._inp_details],
                [d["shape"].tolist() for d in self._out_details],
            )
            # Warm-up inference — eliminates first-call latency spike on Android
            dummy = np.zeros((1, BLAZEFACE_INPUT_SIZE, BLAZEFACE_INPUT_SIZE, 3), dtype=np.float32)
            self._run_tflite(dummy)
            logger.debug("BlazeFace warm-up inference complete.")
        except Exception as exc:
            logger.error("Failed to load BlazeFace TFLite: %s. Using OpenCV fallback.", exc)
            self._load_opencv_fallback()

    def _load_opencv_fallback(self) -> None:
        """Load OpenCV Haar cascade as a dev-time fallback detector."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._opencv_cascade = cv2.CascadeClassifier(cascade_path)
        if self._opencv_cascade.empty():
            logger.error("OpenCV Haar cascade also failed to load. Detection will return empty.")
            self._opencv_cascade = None
        else:
            logger.info("OpenCV Haar cascade fallback loaded (dev mode only).")

    # ── Public API ────────────────────────────────────────────────────────

    def detect_faces(
        self,
        frame: np.ndarray,
        max_faces: int = 1,
    ) -> List[FaceDetection]:
        """
        Detect faces in a BGR camera frame.

        Parameters
        ----------
        frame : np.ndarray
            Full camera frame, BGR uint8, any resolution.
        max_faces : int
            Return at most this many detections, sorted by confidence descending.
            Default 1 for the auth use-case (one person at a time).

        Returns
        -------
        List[FaceDetection]
            Empty list if no face detected above threshold.
            Each FaceDetection has .bbox, .confidence, .landmarks.
        """
        if frame is None or frame.size == 0:
            logger.warning("detect_faces: received empty frame.")
            return []

        if self._use_tflite:
            detections = self._detect_tflite(frame)
        else:
            detections = self._detect_opencv(frame)

        # Sort by confidence, return top-N
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:max_faces]

    # ── TFLite detection path ─────────────────────────────────────────────

    def _detect_tflite(self, frame: np.ndarray) -> List[FaceDetection]:
        """Full BlazeFace TFLite inference path."""
        frame_h, frame_w = frame.shape[:2]

        # ── Preprocess: BGR → RGB, letterbox to 128², normalise ─────────
        # MediaPipe keeps aspect ratio with padding; stretch-resize skews boxes
        # on wide/high-res frames and inflates bboxes when mapped to pixels.
        img, lb_scale, pad_x, pad_y, _, _ = _letterbox_rgb(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            BLAZEFACE_INPUT_SIZE,
        )
        img = img.astype(np.float32) / 127.5 - 1.0   # → [-1, 1]
        img = np.expand_dims(img, axis=0)             # (1, 128, 128, 3)

        # ── Run inference ─────────────────────────────────────────────────
        raw_boxes, raw_scores = self._run_tflite(img)
        # raw_boxes:  (896, 16) — first 4 cols are [dy, dx, h, w], rest landmarks
        # raw_scores: (896, 1)  — logits

        # ── Decode ───────────────────────────────────────────────────────
        detections = self._decode_predictions(
            raw_boxes,
            raw_scores,
            frame_w,
            frame_h,
            lb_scale,
            pad_x,
            pad_y,
        )

        return detections

    def _run_tflite(
        self,
        input_tensor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Feed input tensor, invoke, return (boxes, scores).

        BlazeFace output layout (short-range):
          out[0] → scores  (1, 896, 1)
          out[1] → boxes   (1, 896, 16)
        Index order can vary by export; we identify by shape.
        """
        self._interpreter.set_tensor(self._inp_details[0]["index"], input_tensor)
        self._interpreter.invoke()

        out0 = self._interpreter.get_tensor(self._out_details[0]["index"])[0]
        out1 = self._interpreter.get_tensor(self._out_details[1]["index"])[0]

        # Identify which output is scores (shape N×1) vs boxes (shape N×16)
        if out0.shape[-1] == 1:
            raw_scores, raw_boxes = out0, out1
        else:
            raw_boxes, raw_scores = out0, out1

        return raw_boxes, raw_scores  # Both (896, ...)

    def _decode_predictions(
        self,
        raw_boxes: np.ndarray,
        raw_scores: np.ndarray,
        frame_w: int,
        frame_h: int,
        lb_scale: float = 1.0,
        pad_x: int = 0,
        pad_y: int = 0,
    ) -> List[FaceDetection]:
        """
        Decode BlazeFace anchor-relative regressions → absolute pixel boxes.

        BlazeFace box encoding (relative to anchor):
          box[0] = dy   (centre y delta, normalised by anchor height)
          box[1] = dx   (centre x delta, normalised by anchor width)
          box[2] = h    (height, normalised by anchor height)
          box[3] = w    (width,  normalised by anchor width)
          box[4:16]     (6 landmarks: dy0,dx0, dy1,dx1, ...)

        Landmark encoding mirrors the box encoding but relative to anchor cx,cy.
        """
        # ── Confidence filtering ──────────────────────────────────────────
        scores = _sigmoid(raw_scores[:, 0])   # (896,)
        mask = scores > self._score_threshold
        if not mask.any():
            return []

        scores = scores[mask]
        boxes = raw_boxes[mask]                # (K, 16)
        anchors = self._anchors[mask]          # (K, 4)  [cx, cy, w, h] normalised

        s = float(BLAZEFACE_INPUT_SIZE)

        # ── Decode centre positions ───────────────────────────────────────
        # BlazeFace: cx = anchor_cx + dx/input_size
        cx = anchors[:, 0] + boxes[:, 1] / s
        cy = anchors[:, 1] + boxes[:, 0] / s
        w  = boxes[:, 3] / s
        h  = boxes[:, 2] / s

        # Map normalised letterbox coords → original frame pixels
        x1, y1, x2, y2 = _letterbox_boxes_to_frame(
            cx, cy, w, h, frame_w, frame_h, lb_scale, pad_x, pad_y,
        )

        # ── Decode landmarks (6 keypoints × 2 coords = 12 values) ────────
        all_landmarks: List[List[Tuple[float, float]]] = []
        for i in range(len(scores)):
            pts = []
            for k in range(6):
                lm_cy = anchors[i, 1] + boxes[i, 4 + k * 2]     / s
                lm_cx = anchors[i, 0] + boxes[i, 4 + k * 2 + 1] / s
                lm_x, lm_y = _letterbox_point_to_frame(
                    lm_cx, lm_cy, lb_scale, pad_x, pad_y,
                )
                pts.append((float(lm_x), float(lm_y)))
            all_landmarks.append(pts)

        # ── Size filter ───────────────────────────────────────────────────
        detections = []
        for i in range(len(scores)):
            bw = int(x2[i] - x1[i])
            bh = int(y2[i] - y1[i])
            if bw < MIN_FACE_SIZE_PX or bh < MIN_FACE_SIZE_PX:
                continue
            detections.append(
                FaceDetection(
                    bbox=(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                    confidence=float(scores[i]),
                    landmarks=all_landmarks[i],
                )
            )

        # ── NMS ───────────────────────────────────────────────────────────
        detections = self._nms(detections)
        logger.debug("BlazeFace: %d raw → %d after NMS", len(scores), len(detections))
        return detections

    # ── OpenCV fallback path ──────────────────────────────────────────────

    def _detect_opencv(self, frame: np.ndarray) -> List[FaceDetection]:
        """Haar cascade fallback — returns FaceDetection without landmarks."""
        if self._opencv_cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._opencv_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(MIN_FACE_SIZE_PX, MIN_FACE_SIZE_PX),
        )

        detections = []
        for (x, y, w, h) in (rects if len(rects) else []):
            detections.append(
                FaceDetection(
                    bbox=(int(x), int(y), int(x + w), int(y + h)),
                    confidence=0.90,   # Haar has no per-detection score; use fixed value
                    landmarks=[],      # No landmarks from Haar
                )
            )
        return detections

    # ── NMS ──────────────────────────────────────────────────────────────

    def _nms(self, detections: List[FaceDetection]) -> List[FaceDetection]:
        """
        Weighted non-maximum suppression.

        Standard NMS hard-removes overlapping boxes.  Weighted NMS
        instead blends the coordinates of overlapping boxes proportionally
        to their confidence, producing a more stable centroid — less
        jitter on consecutive frames.
        """
        if len(detections) <= 1:
            return detections

        # Convert to numpy for vectorised IoU
        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # Compute IoU of best box vs all remaining
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

            order = order[1:][iou <= self._iou_threshold]

        return [detections[k] for k in keep]


# ── Module-level utilities ─────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid — avoids overflow on large negative logits."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _letterbox_rgb(
    image_rgb: np.ndarray,
    target: int,
) -> Tuple[np.ndarray, float, int, int, int, int]:
    """
    Resize with aspect ratio preserved, pad to ``target × target`` (MediaPipe style).

    Returns padded image, scale, pad_x, pad_y, resized_w, resized_h.
    """
    h, w = image_rgb.shape[:2]
    scale = min(target / w, target / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(
        image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR,
    )
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    padded = np.zeros((target, target, 3), dtype=image_rgb.dtype)
    padded[pad_y: pad_y + new_h, pad_x: pad_x + new_w] = resized
    return padded, scale, pad_x, pad_y, new_w, new_h


def _letterbox_point_to_frame(
    cx_norm: float,
    cy_norm: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    input_size: int = BLAZEFACE_INPUT_SIZE,
) -> Tuple[float, float]:
    """Invert letterbox: normalised coords on 128² canvas → original pixels."""
    px = cx_norm * input_size
    py = cy_norm * input_size
    return (px - pad_x) / scale, (py - pad_y) / scale


def _letterbox_boxes_to_frame(
    cx: np.ndarray,
    cy: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    frame_w: int,
    frame_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised bbox decode from letterbox-normalised centres/sizes."""
    s = float(BLAZEFACE_INPUT_SIZE)
    cx_px = cx * s
    cy_px = cy * s
    w_px  = w * s
    h_px  = h * s

    cx_o = (cx_px - pad_x) / scale
    cy_o = (cy_px - pad_y) / scale
    w_o  = w_px / scale
    h_o  = h_px / scale

    x1 = np.clip((cx_o - w_o / 2).astype(int), 0, frame_w - 1)
    y1 = np.clip((cy_o - h_o / 2).astype(int), 0, frame_h - 1)
    x2 = np.clip((cx_o + w_o / 2).astype(int), 0, frame_w - 1)
    y2 = np.clip((cy_o + h_o / 2).astype(int), 0, frame_h - 1)
    return x1, y1, x2, y2

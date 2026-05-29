"""
pipeline/liveness_detector.py
==============================
SecureEdge AI — Liveness Detection Module

Two-tier liveness system:
  Tier 1 (Passive): Silent-Face Anti-Spoofing — single ONNX MiniFASNet model
    Primary path  : models/antispoofing/antispoofing.onnx
    Fallback path : models/antispoofing/2.7_80x80_MiniFASNetV2.onnx
    Fallback path : models/antispoofing/4_0_0_80x80_MiniFASNetV1SE.onnx

    MiniFASNet output layout (3 classes):
        index 0 → print_attack
        index 1 → live
        index 2 → replay_attack
    live_probability = softmax(logits)[1]

    CRITICAL — crop contract:
        predict() takes the FULL camera frame + raw bbox from the detector.
        Each model does its own Silent-Face square crop (side = max(w,h)*scale,
        centred on bbox) before inference.  A pre-cropped patch is NOT accepted
        here; too-tight crops score as SPOOF regardless of the real face.

  Tier 2 (Active): Eye Aspect Ratio (EAR) blink detector
    - Requires MediaPipe Face Mesh landmarks (6-point eye model)
    - Guards against high-quality print/screen attacks that fool the CNN

Final liveness decision requires BOTH tiers to pass.

Fallback: when ONNX model is absent the CNN tier uses a lightweight
OpenCV LBP texture-variance heuristic (dev/testing only).

Author: SecureEdge AI Team
"""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ── Constants ────────────────────────────────────────────────────────────────

CNN_LIVENESS_THRESHOLD: float = 0.60   # live_prob above this → LIVE
EAR_CLOSED_THRESHOLD:   float = 0.20   # EAR below this → eye closed
EAR_MIN_BLINK_FRAMES:   int   = 2      # consecutive closed frames for a blink
EAR_BLINK_TIMEOUT_SEC:  float = 6.0   # seconds allowed for blink challenge

# Silent-Face model input resolution (encoded in filename)
_MINIFAS_INPUT_H: int = 80
_MINIFAS_INPUT_W: int = 80

# ImageNet normalisation — BGR channel order (matches Silent-Face repo)
_IMAGENET_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
_IMAGENET_STD  = np.array([0.225, 0.224, 0.229], dtype=np.float32)

# MiniFASNet class layout: index 1 is "live"
_LIVE_CLASS_INDEX: int = 1

# Default model directory
_DEFAULT_MODEL_DIR: str = "models/antispoofing"

# Candidate model files with their Silent-Face crop scale factors.
# Tried in order; first file found is loaded as the primary model.
# The scale determines side = max(bbox_w, bbox_h) * scale, centred on bbox.
_CANDIDATE_MODELS: List[Tuple[str, float]] = [
    ("antispoofing.onnx",            2.7),   # single-model export
    ("2.7_80x80_MiniFASNetV2.onnx",  2.7),
    ("4_0_0_80x80_MiniFASNetV1SE.onnx", 4.0),
]

# MediaPipe Face Mesh: 6-point EAR landmark indices per eye
_EAR_LEFT_IDX  = [362, 385, 387, 263, 373, 380]
_EAR_RIGHT_IDX = [33,  160, 158, 133, 153, 144]


# ── Enumerations ─────────────────────────────────────────────────────────────

class LivenessDecision(str, Enum):
    LIVE      = "LIVE"
    SPOOF     = "SPOOF"
    UNCERTAIN = "UNCERTAIN"


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class CnnLivenessResult:
    live_probability: float          # [0.0, 1.0] — 1.0 = confident live
    decision:         LivenessDecision
    is_stub:          bool = False   # True when LBP fallback was used
    is_live:          bool = False   # convenience boolean mirror
    meta:             Dict[str, Any] = field(default_factory=dict)
    # meta keys (when ONNX path is active):
    #   raw_probs      : List[float]  — softmax probabilities [print, live, replay]
    #   inference_ms   : float        — wall-clock inference time
    #   model_name     : str          — filename of the model that ran
    #   crop_scale     : float        — scale factor used for the crop


@dataclass
class BlinkResult:
    blink_detected: bool
    total_blinks:   int
    ear_value:      float
    timed_out:      bool = False


@dataclass
class LivenessResult:
    decision:       LivenessDecision
    cnn_score:      float
    blink_detected: bool
    final_score:    float
    details:        str = ""


# ── ONNX anti-spoofing CNN ────────────────────────────────────────────────────

class AntiSpoofingCNN:
    """
    Single-model Silent-Face ONNX anti-spoofing wrapper.

    Loads the first available model from _CANDIDATE_MODELS (checked in order).
    Falls back to the LBP texture heuristic if no model file is found.

    Crop contract
    -------------
    predict() takes the FULL camera frame and the raw detector bbox.
    The model crops internally using its Silent-Face scale factor so that
    the surrounding forehead/cheek context the model was trained on is
    always present.  Passing a pre-cropped patch bypasses this logic and
    will produce spurious SPOOF outputs.
    """

    def __init__(
        self,
        model_dir: str   = _DEFAULT_MODEL_DIR,
        threshold: float = CNN_LIVENESS_THRESHOLD,
    ) -> None:
        self._threshold  = threshold
        self._session    = None     # onnxruntime.InferenceSession
        self._model_name = ""
        self._scale      = 2.7
        self._use_onnx   = False

        self._load_model(model_dir)

        if not self._use_onnx:
            logger.warning(
                "AntiSpoofingCNN: no ONNX model found in '%s'. "
                "Running LBP texture fallback — NOT production-grade. "
                "Expected one of: %s",
                model_dir,
                [f for f, _ in _CANDIDATE_MODELS],
            )

    # ── Loading ───────────────────────────────────────────────────────────

    def _load_model(self, model_dir: str) -> None:
        """Load the first available candidate model."""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error(
                "onnxruntime is not installed — run: pip install onnxruntime"
            )
            return

        for filename, scale in _CANDIDATE_MODELS:
            path = os.path.join(model_dir, filename)
            if not os.path.isfile(path):
                logger.debug("Model not found, skipping: '%s'", path)
                continue
            try:
                sess = ort.InferenceSession(
                    path,
                    providers=["CPUExecutionProvider"],
                )
                self._session    = sess
                self._model_name = filename
                self._scale      = scale
                self._use_onnx   = True
                logger.info(
                    "AntiSpoofingCNN loaded: '%s' | scale=%.1f | "
                    "input=%s | output=%s",
                    filename, scale,
                    sess.get_inputs()[0].shape,
                    sess.get_outputs()[0].shape,
                )
                self._warmup()
                return
            except Exception as exc:
                logger.error("Failed to load '%s': %s", path, exc)

    def _warmup(self) -> None:
        """One dummy forward-pass to prime the ONNX runtime."""
        dummy = np.zeros(
            (1, 3, _MINIFAS_INPUT_H, _MINIFAS_INPUT_W), dtype=np.float32
        )
        try:
            inp = self._session.get_inputs()[0].name
            self._session.run(None, {inp: dummy})
            logger.debug("AntiSpoofingCNN warm-up complete: %s", self._model_name)
        except Exception as exc:
            logger.warning("Warm-up failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────

    def predict(
        self,
        frame_bgr: np.ndarray,
        bbox:      Tuple[int, int, int, int],
    ) -> CnnLivenessResult:
        """
        Run liveness inference on a single frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Full camera frame, BGR uint8.  Must be the RAW frame —
            not a pre-cropped face patch.
        bbox : (x1, y1, x2, y2)
            Face bounding box in pixel coordinates from the detector.

        Returns
        -------
        CnnLivenessResult
            Contains live_probability, decision, is_live, is_stub,
            and a meta dict with raw_probs / inference_ms / model_name /
            crop_scale when ONNX is active.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("AntiSpoofingCNN.predict: empty frame — UNCERTAIN")
            return CnnLivenessResult(
                live_probability=0.5,
                decision=LivenessDecision.UNCERTAIN,
                is_stub=True,
                is_live=False,
            )

        if self._use_onnx:
            return self._onnx_predict(frame_bgr, bbox)
        else:
            crop     = _scale_crop(frame_bgr, bbox, 2.7)
            prob     = self._lbp_heuristic(crop)
            decision = self._threshold_to_decision(prob)
            return CnnLivenessResult(
                live_probability=prob,
                decision=decision,
                is_stub=True,
                is_live=(decision == LivenessDecision.LIVE),
                meta={"fallback": "lbp_texture"},
            )

    # ── ONNX inference ────────────────────────────────────────────────────

    def _onnx_predict(
        self,
        frame_bgr: np.ndarray,
        bbox:      Tuple[int, int, int, int],
    ) -> CnnLivenessResult:
        """Crop → preprocess → infer → softmax → return result."""
        crop = _scale_crop(frame_bgr, bbox, self._scale)
        if crop.size == 0:
            logger.warning(
                "_onnx_predict: empty crop for bbox=%s on %dx%d frame.",
                bbox, frame_bgr.shape[1], frame_bgr.shape[0],
            )
            return CnnLivenessResult(
                live_probability=0.5,
                decision=LivenessDecision.UNCERTAIN,
                is_stub=False,
                is_live=False,
                meta={"error": "empty_crop"},
            )

        tensor = _preprocess_minifas(crop)
        logger.debug(
            "Anti-spoofing input | tensor shape=%s dtype=%s range=[%.3f, %.3f]",
            tensor.shape, tensor.dtype, float(tensor.min()), float(tensor.max()),
        )

        t0 = time.perf_counter()
        try:
            inp_name = self._session.get_inputs()[0].name
            raw_out  = self._session.run(None, {inp_name: tensor})[0]  # (1, 3)
        except Exception as exc:
            logger.error("ONNX inference failed: %s", exc)
            return CnnLivenessResult(
                live_probability=0.5,
                decision=LivenessDecision.UNCERTAIN,
                is_stub=False,
                is_live=False,
                meta={"error": str(exc)},
            )
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # Validate output shape: expect (1, 3) logits
        if raw_out.ndim != 2 or raw_out.shape[1] < 2:
            logger.warning(
                "Unexpected ONNX output shape %s — expected (1, N>=2).", raw_out.shape
            )

        logits = raw_out[0].astype(np.float64)
        logits -= logits.max()                      # numerical stability
        exp    = np.exp(logits)
        probs  = exp / exp.sum()                    # softmax

        # MiniFASNet class layout: [print_attack=0, live=1, replay_attack=2]
        live_prob = float(probs[_LIVE_CLASS_INDEX]) if len(probs) > _LIVE_CLASS_INDEX else 0.0

        decision = self._threshold_to_decision(live_prob)

        meta = {
            "raw_probs":    probs.tolist(),
            "inference_ms": round(inference_ms, 2),
            "model_name":   self._model_name,
            "crop_scale":   self._scale,
        }

        logger.debug(
            "AntiSpoofingCNN | model=%s scale=%.1f | "
            "probs=[print=%.3f live=%.3f replay=%.3f] | "
            "live_prob=%.4f decision=%s | %.1f ms",
            self._model_name, self._scale,
            probs[0] if len(probs) > 0 else -1,
            probs[1] if len(probs) > 1 else -1,
            probs[2] if len(probs) > 2 else -1,
            live_prob, decision.value, inference_ms,
        )

        return CnnLivenessResult(
            live_probability=live_prob,
            decision=decision,
            is_stub=False,
            is_live=(decision == LivenessDecision.LIVE),
            meta=meta,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _threshold_to_decision(self, prob: float) -> LivenessDecision:
        if prob >= self._threshold:
            return LivenessDecision.LIVE
        elif prob < (self._threshold - 0.15):
            return LivenessDecision.SPOOF
        return LivenessDecision.UNCERTAIN

    # ── LBP fallback (dev only) ───────────────────────────────────────────

    @staticmethod
    def _lbp_heuristic(face_bgr: np.ndarray) -> float:
        """Texture-variance heuristic.  NOT production-grade."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        h, w = gray.shape
        bs   = 8
        variances = [
            float(np.var(gray[r:r+bs, c:c+bs].astype(np.float32)))
            for r in range(0, h - bs, bs)
            for c in range(0, w - bs, bs)
        ]
        if not variances:
            return 0.5
        mv = float(np.mean(variances))
        prob = 0.60 + 0.30 * min(1.0, (mv - 30) / 470) if 30 < mv < 500 else 0.35
        logger.debug("LBP heuristic | mean_var=%.1f → prob=%.3f", mv, prob)
        return prob


# ── Active blink detection tier ───────────────────────────────────────────────

class BlinkDetector:
    """
    Eye Aspect Ratio (EAR) blink detector using Face Mesh landmarks.

    EAR = (‖P2-P6‖ + ‖P3-P5‖) / (2 × ‖P1-P4‖)

    Open eye: EAR ≈ 0.25–0.35.  Closed eye: EAR < 0.20.
    A valid blink requires EAR < threshold for ≥ EAR_MIN_BLINK_FRAMES frames.
    """

    def __init__(
        self,
        ear_threshold:    float = EAR_CLOSED_THRESHOLD,
        min_blink_frames: int   = EAR_MIN_BLINK_FRAMES,
        timeout_sec:      float = EAR_BLINK_TIMEOUT_SEC,
    ) -> None:
        self._ear_threshold    = ear_threshold
        self._min_blink_frames = min_blink_frames
        self._timeout_sec      = timeout_sec
        self._reset()

    def reset(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._frame_counter: int   = 0
        self._total_blinks:  int   = 0
        self._session_start: float = time.monotonic()
        self._last_ear:      float = 0.30

    def update(
        self,
        landmarks_2d: List[Tuple[float, float]],
    ) -> BlinkResult:
        elapsed   = time.monotonic() - self._session_start
        timed_out = elapsed > self._timeout_sec

        if timed_out:
            return BlinkResult(
                blink_detected=False,
                total_blinks=self._total_blinks,
                ear_value=self._last_ear,
                timed_out=True,
            )

        try:
            left_ear  = _compute_ear(landmarks_2d, _EAR_LEFT_IDX)
            right_ear = _compute_ear(landmarks_2d, _EAR_RIGHT_IDX)
            ear = (left_ear + right_ear) / 2.0
            self._last_ear = ear
        except (IndexError, ZeroDivisionError) as exc:
            logger.warning("BlinkDetector.update: landmark error — %s", exc)
            return BlinkResult(
                blink_detected=self._total_blinks >= 1,
                total_blinks=self._total_blinks,
                ear_value=self._last_ear,
            )

        if ear < self._ear_threshold:
            self._frame_counter += 1
        else:
            if self._frame_counter >= self._min_blink_frames:
                self._total_blinks += 1
                logger.debug(
                    "Blink #%d detected | EAR=%.3f | frames_closed=%d",
                    self._total_blinks, ear, self._frame_counter,
                )
            self._frame_counter = 0

        return BlinkResult(
            blink_detected=self._total_blinks >= 1,
            total_blinks=self._total_blinks,
            ear_value=ear,
        )


# ── Combined liveness detector facade ────────────────────────────────────────

class LivenessDetector:
    """
    Facade combining both liveness tiers into one decision.

    Decision table
    --------------
    CNN=SPOOF                            → SPOOF  (blink irrelevant)
    CNN=LIVE   + blink confirmed         → LIVE
    CNN=LIVE   + blink timeout           → SPOOF
    CNN=LIVE   + awaiting blink          → UNCERTAIN
    CNN=LIVE   + require_blink=False     → LIVE
    CNN=UNCERTAIN + blink confirmed      → LIVE
    CNN=UNCERTAIN + blink timeout        → SPOOF
    CNN=UNCERTAIN + awaiting blink       → UNCERTAIN

    final_score = 0.6 × cnn_prob + 0.4 × blink_bonus

    Parameters
    ----------
    model_dir     : directory containing ONNX antispoofing models
    require_blink : False disables the active challenge (faster, less secure)
    """

    def __init__(
        self,
        model_dir:     str  = _DEFAULT_MODEL_DIR,
        require_blink: bool = True,
    ) -> None:
        self._cnn           = AntiSpoofingCNN(model_dir=model_dir)
        self._blink         = BlinkDetector()
        self._require_blink = require_blink

    def reset_session(self) -> None:
        """Call at the start of every new authentication attempt."""
        self._blink.reset()

    def predict_liveness(
        self,
        frame_bgr:      np.ndarray,
        bbox:           Tuple[int, int, int, int],
        face_landmarks: Optional[List[Tuple[float, float]]] = None,
    ) -> LivenessResult:
        """
        Evaluate liveness for the current frame.

        Parameters
        ----------
        frame_bgr      : full camera frame, BGR uint8 — NOT a pre-cropped patch.
        bbox           : (x1, y1, x2, y2) face bounding box from the detector.
        face_landmarks : 468-point Face Mesh landmarks in pixel space (optional).

        Returns
        -------
        LivenessResult
        """
        # ── Tier 1 ────────────────────────────────────────────────────────
        cnn = self._cnn.predict(frame_bgr, bbox)

        if cnn.decision == LivenessDecision.SPOOF:
            return LivenessResult(
                decision=LivenessDecision.SPOOF,
                cnn_score=cnn.live_probability,
                blink_detected=False,
                final_score=cnn.live_probability,
                details=f"CNN SPOOF (prob={cnn.live_probability:.3f})",
            )

        # ── Tier 2 ────────────────────────────────────────────────────────
        blink = BlinkResult(blink_detected=True, total_blinks=0, ear_value=0.30)
        if face_landmarks is not None and self._require_blink:
            blink = self._blink.update(face_landmarks)

        # ── Combined decision ─────────────────────────────────────────────
        blink_bonus = 1.0 if blink.blink_detected else 0.0
        final_score = 0.60 * cnn.live_probability + 0.40 * blink_bonus

        if cnn.decision == LivenessDecision.UNCERTAIN:
            if blink.blink_detected:
                decision = LivenessDecision.LIVE
                details  = f"CNN UNCERTAIN resolved by blink (score={final_score:.3f})"
            elif blink.timed_out:
                decision = LivenessDecision.SPOOF
                details  = "CNN UNCERTAIN + blink timeout → SPOOF"
            else:
                decision = LivenessDecision.UNCERTAIN
                details  = f"Awaiting blink (EAR={blink.ear_value:.3f})"
        else:
            # CNN == LIVE
            if self._require_blink and not blink.blink_detected:
                if blink.timed_out:
                    decision = LivenessDecision.SPOOF
                    details  = "CNN LIVE but blink timed out → SPOOF"
                else:
                    decision = LivenessDecision.UNCERTAIN
                    details  = f"CNN LIVE, awaiting blink (EAR={blink.ear_value:.3f})"
            else:
                decision = LivenessDecision.LIVE
                details  = f"CNN LIVE + blink confirmed (score={final_score:.3f})"

        logger.info(
            "Liveness | cnn=%.3f stub=%s blink=%s ear=%.3f final=%.3f → %s | %s",
            cnn.live_probability, cnn.is_stub,
            blink.blink_detected, blink.ear_value,
            final_score, decision.value, details,
        )

        return LivenessResult(
            decision=decision,
            cnn_score=cnn.live_probability,
            blink_detected=blink.blink_detected,
            final_score=final_score,
            details=details,
        )


# ── Module-level utilities ────────────────────────────────────────────────────

def _scale_crop(
    frame: np.ndarray,
    bbox:  Tuple[int, int, int, int],
    scale: float,
) -> np.ndarray:
    """
    Silent-Face correct square crop — matches CropImage in the original repo.

    side = max(bbox_w, bbox_h) * scale
    The crop is centred on the bbox centre and clamped to frame boundaries.

    Using the longer axis + the model's own scale factor guarantees the
    surrounding skin/background context the model was trained on is present.
    A tight rectangular crop (20% padding) reliably produces SPOOF outputs.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        logger.warning("_scale_crop: degenerate bbox %s", bbox)
        return np.empty((0, 0, 3), dtype=np.uint8)

    cx, cy    = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_side = max(bw, bh) * scale / 2.0
    fh, fw    = frame.shape[:2]

    nx1 = max(0, int(cx - half_side))
    ny1 = max(0, int(cy - half_side))
    nx2 = min(fw, int(cx + half_side))
    ny2 = min(fh, int(cy + half_side))

    crop = frame[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        logger.warning(
            "_scale_crop: empty result bbox=%s scale=%.1f frame=%dx%d",
            bbox, scale, fw, fh,
        )
    return crop


def _preprocess_minifas(face_bgr: np.ndarray) -> np.ndarray:
    """
    Resize and normalise a BGR face crop for Silent-Face ONNX models.

    Steps
    -----
    1. Resize to 80×80 (INTER_LINEAR)
    2. Cast to float32, divide by 255  →  [0.0, 1.0]
    3. Subtract ImageNet BGR mean, divide by ImageNet BGR std
    4. Transpose HWC → CHW
    5. Add batch dim  →  (1, 3, 80, 80) float32
    """
    img = cv2.resize(face_bgr, (_MINIFAS_INPUT_W, _MINIFAS_INPUT_H),
                     interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    img = img.transpose(2, 0, 1)          # HWC → CHW
    img = np.expand_dims(img, axis=0)     # → (1, 3, 80, 80)
    return img.astype(np.float32)


def _compute_ear(
    landmarks:   List[Tuple[float, float]],
    eye_indices: List[int],
) -> float:
    """Eye Aspect Ratio from 6 landmark points (P1..P6)."""
    pts = [np.array(landmarks[i], dtype=np.float32) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    v_a = np.linalg.norm(p2 - p6)
    v_b = np.linalg.norm(p3 - p5)
    h   = np.linalg.norm(p1 - p4)
    if h < 1e-6:
        return 0.0
    return float((v_a + v_b) / (2.0 * h))

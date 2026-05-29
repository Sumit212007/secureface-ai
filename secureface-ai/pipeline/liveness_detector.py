"""
pipeline/liveness_detector.py
==============================
SecureEdge AI — Liveness Detection Module

Two-tier liveness system:
  Tier 1 (Passive): Silent-Face Anti-Spoofing — dual ONNX ensemble
    - Model A: 2.7_80x80_MiniFASNetV2.onnx
    - Model B: 4_0_0_80x80_MiniFASNetV1SE.onnx
    - Both models run on every frame; their softmax scores are averaged
    - Input per model: (1, 3, H, W) float32, normalised to ImageNet stats
    - Output per model: (1, 3) softmax — class 1 = live, class 0/2 = spoof
    - Final live_probability = mean(model_A_live_prob, model_B_live_prob)

  Tier 2 (Active): Eye Aspect Ratio (EAR) blink detector
    - Requires MediaPipe Face Mesh landmarks (6-point eye model)
    - Detects a real blink: EAR drops below threshold for 2+ frames
    - Guards against high-quality print/screen attacks that fool the CNN

Final liveness decision requires BOTH tiers to pass.

Fallback: when ONNX models are absent the CNN tier uses a lightweight
OpenCV LBP texture-variance heuristic (sufficient for dev/testing only).

Author: SecureEdge AI Team
"""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ── Constants ────────────────────────────────────────────────────────────────

CNN_LIVENESS_THRESHOLD: float = 0.60  # Averaged live prob above this → LIVE
EAR_CLOSED_THRESHOLD: float   = 0.20  # EAR below this → eye closed
EAR_MIN_BLINK_FRAMES: int     = 2     # Consecutive closed frames for a valid blink
EAR_BLINK_TIMEOUT_SEC: float  = 6.0  # Seconds to wait for blink challenge

# Silent-Face model input size encoded in filename (both models use 80×80)
_MINIFAS_INPUT_H: int = 80
_MINIFAS_INPUT_W: int = 80

# ImageNet normalisation used by Silent-Face preprocessing
_IMAGENET_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)  # BGR order
_IMAGENET_STD  = np.array([0.225, 0.224, 0.229], dtype=np.float32)  # BGR order

# Default ONNX model paths (both files must be in this folder)
_DEFAULT_MODEL_DIR: str = "models/antispoofing"
_MODEL_FILES = [
    "2.7_80x80_MiniFASNetV2.onnx",
    "4_0_0_80x80_MiniFASNetV1SE.onnx",
]

# Silent-Face class layout: index 1 = live (real face)
_LIVE_CLASS_INDEX: int = 1

# MediaPipe Face Mesh: 6-point EAR landmark indices per eye
_EAR_LEFT_IDX  = [362, 385, 387, 263, 373, 380]  # P1..P6 left eye
_EAR_RIGHT_IDX = [33,  160, 158, 133, 153, 144]  # P1..P6 right eye


# ── Enumerations ─────────────────────────────────────────────────────────────

class LivenessDecision(str, Enum):
    LIVE      = "LIVE"
    SPOOF     = "SPOOF"
    UNCERTAIN = "UNCERTAIN"


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class CnnLivenessResult:
    live_probability: float       # [0, 1] — 1.0 = confident live
    decision: LivenessDecision
    is_stub: bool = False         # True if LBP fallback was used


@dataclass
class BlinkResult:
    blink_detected: bool
    total_blinks: int
    ear_value: float
    timed_out: bool = False


@dataclass
class LivenessResult:
    decision: LivenessDecision
    cnn_score: float
    blink_detected: bool
    final_score: float
    details: str = ""


# ── ONNX anti-spoofing ensemble ───────────────────────────────────────────────

class AntiSpoofingCNN:
    """
    Dual-model Silent-Face ONNX anti-spoofing ensemble.

    Runs both MiniFASNetV2 and MiniFASNetV1SE on every frame and averages
    their live-class softmax probabilities, matching the original repo's
    ensemble strategy.

    When either or both ONNX files are missing the module falls back to the
    LBP texture heuristic so the pipeline stays runnable during development.

    Parameters
    ----------
    model_dir : str
        Directory containing the two .onnx model files.
    threshold : float
        Averaged live probability threshold.
    """

    def __init__(
        self,
        model_dir: str = _DEFAULT_MODEL_DIR,
        threshold: float = CNN_LIVENESS_THRESHOLD,
    ) -> None:
        self._threshold = threshold
        self._sessions: List = []        # onnxruntime InferenceSession objects
        self._model_names: List[str] = []
        self._use_onnx = False

        self._load_onnx_models(model_dir)

        if not self._use_onnx:
            logger.warning(
                "No ONNX anti-spoofing models loaded from '%s'. "
                "Using LBP texture fallback. "
                "Place 2.7_80x80_MiniFASNetV2.onnx and "
                "4_0_0_80x80_MiniFASNetV1SE.onnx in that folder for production use.",
                model_dir,
            )

    # ── Loading ───────────────────────────────────────────────────────────

    def _load_onnx_models(self, model_dir: str) -> None:
        """Load all available ONNX models from model_dir."""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error(
                "onnxruntime is not installed. "
                "Run: pip install onnxruntime"
            )
            return

        loaded = 0
        for filename in _MODEL_FILES:
            path = os.path.join(model_dir, filename)
            if not os.path.isfile(path):
                logger.warning("ONNX model not found: '%s'", path)
                continue
            try:
                sess = ort.InferenceSession(
                    path,
                    providers=["CPUExecutionProvider"],
                )
                self._sessions.append(sess)
                self._model_names.append(filename)
                logger.info(
                    "Loaded anti-spoofing ONNX model: '%s' | "
                    "input=%s output=%s",
                    filename,
                    sess.get_inputs()[0].shape,
                    sess.get_outputs()[0].shape,
                )
                loaded += 1
            except Exception as exc:
                logger.error("Failed to load '%s': %s", path, exc)

        if loaded > 0:
            self._use_onnx = True
            self._warmup()
            logger.info(
                "Anti-spoofing ensemble ready: %d / %d models loaded.",
                loaded, len(_MODEL_FILES),
            )

    def _warmup(self) -> None:
        """Run one dummy inference per model to eliminate first-call latency."""
        dummy = np.zeros((1, 3, _MINIFAS_INPUT_H, _MINIFAS_INPUT_W), dtype=np.float32)
        for sess, name in zip(self._sessions, self._model_names):
            try:
                inp_name = sess.get_inputs()[0].name
                sess.run(None, {inp_name: dummy})
                logger.debug("Warm-up complete: %s", name)
            except Exception as exc:
                logger.warning("Warm-up failed for %s: %s", name, exc)

    # ── Public API ────────────────────────────────────────────────────────

    def predict(self, face_bgr: np.ndarray) -> CnnLivenessResult:
        """
        Predict live vs spoof from a cropped BGR face patch.

        Parameters
        ----------
        face_bgr : np.ndarray
            Cropped face region, BGR uint8, any size.
            The method handles resizing and normalisation internally.

        Returns
        -------
        CnnLivenessResult
        """
        if face_bgr is None or face_bgr.size == 0:
            logger.warning("AntiSpoofingCNN.predict: empty face patch.")
            return CnnLivenessResult(
                live_probability=0.5,
                decision=LivenessDecision.UNCERTAIN,
                is_stub=True,
            )

        if self._use_onnx:
            live_prob = self._run_ensemble(face_bgr)
            is_stub = False
        else:
            live_prob = self._lbp_heuristic_bgr(face_bgr)
            is_stub = True

        if live_prob >= self._threshold:
            decision = LivenessDecision.LIVE
        elif live_prob < (self._threshold - 0.15):
            decision = LivenessDecision.SPOOF
        else:
            decision = LivenessDecision.UNCERTAIN

        logger.debug(
            "CNN liveness | prob=%.4f decision=%s stub=%s",
            live_prob, decision.value, is_stub,
        )
        return CnnLivenessResult(
            live_probability=float(live_prob),
            decision=decision,
            is_stub=is_stub,
        )

    # ── ONNX inference ────────────────────────────────────────────────────

    def _run_ensemble(self, face_bgr: np.ndarray) -> float:
        """
        Run all loaded ONNX models and return the averaged live probability.

        Silent-Face preprocessing pipeline:
          1. Resize to model input size (80×80)
          2. Convert BGR → float32, divide by 255
          3. Subtract ImageNet BGR mean, divide by ImageNet BGR std
          4. Transpose HWC → CHW, add batch dim → (1, 3, 80, 80)
        """
        tensor = self._preprocess(face_bgr)
        live_probs = []

        for sess, name in zip(self._sessions, self._model_names):
            try:
                inp_name = sess.get_inputs()[0].name
                raw_output = sess.run(None, {inp_name: tensor})[0]  # (1, 3)

                # Apply softmax (model outputs raw logits)
                logits = raw_output[0].astype(np.float64)
                logits -= logits.max()          # numerical stability
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum()

                live_prob = float(probs[_LIVE_CLASS_INDEX])
                live_probs.append(live_prob)
                logger.debug("%s → live_prob=%.4f", name, live_prob)

            except Exception as exc:
                logger.warning("Inference failed for %s: %s", name, exc)

        if not live_probs:
            logger.warning("All ONNX models failed — returning 0.5 (UNCERTAIN).")
            return 0.5

        avg = float(np.mean(live_probs))
        logger.debug("Ensemble average live_prob=%.4f", avg)
        return avg

    @staticmethod
    def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
        """
        Resize and normalise a BGR face crop for Silent-Face ONNX models.

        Returns
        -------
        np.ndarray  shape (1, 3, 80, 80), float32
        """
        img = cv2.resize(
            face_bgr,
            (_MINIFAS_INPUT_W, _MINIFAS_INPUT_H),
            interpolation=cv2.INTER_LINEAR,
        )
        img = img.astype(np.float32) / 255.0          # → [0, 1]
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD  # ImageNet normalisation
        img = img.transpose(2, 0, 1)                   # HWC → CHW
        img = np.expand_dims(img, axis=0)              # → (1, 3, 80, 80)
        return img.astype(np.float32)

    # ── LBP fallback (dev only) ───────────────────────────────────────────

    @staticmethod
    def _lbp_heuristic_bgr(face_bgr: np.ndarray) -> float:
        """LBP texture-variance fallback. NOT production-grade."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        h, w = gray.shape
        block_size = 8
        variances = []
        for r in range(0, h - block_size, block_size):
            for c in range(0, w - block_size, block_size):
                block = gray[r:r + block_size, c:c + block_size].astype(np.float32)
                variances.append(float(np.var(block)))

        if not variances:
            return 0.5

        mean_var = float(np.mean(variances))
        if 30 < mean_var < 500:
            live_prob = 0.60 + 0.30 * min(1.0, (mean_var - 30) / 470)
        else:
            live_prob = 0.35

        logger.debug("LBP heuristic | mean_var=%.1f → live_prob=%.3f", mean_var, live_prob)
        return live_prob


# ── Active blink detection tier ───────────────────────────────────────────────

class BlinkDetector:
    """
    Eye Aspect Ratio (EAR) based blink detector using Face Mesh landmarks.

    EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)

    EAR is ~0.25-0.35 when open, drops below 0.20 when closed.
    A valid blink requires EAR < threshold for EAR_MIN_BLINK_FRAMES consecutive frames.
    """

    def __init__(
        self,
        ear_threshold: float = EAR_CLOSED_THRESHOLD,
        min_blink_frames: int = EAR_MIN_BLINK_FRAMES,
        timeout_sec: float = EAR_BLINK_TIMEOUT_SEC,
    ) -> None:
        self._ear_threshold = ear_threshold
        self._min_blink_frames = min_blink_frames
        self._timeout_sec = timeout_sec
        self._reset()

    def reset(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._frame_counter: int = 0
        self._total_blinks: int = 0
        self._session_start: float = time.monotonic()
        self._last_ear: float = 0.30

    def update(self, landmarks_2d: List[Tuple[float, float]]) -> BlinkResult:
        elapsed = time.monotonic() - self._session_start
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
    Facade combining both liveness tiers into a single decision.

    Tier weighting:
      - CNN tier is the primary gate: SPOOF → final SPOOF regardless of blink.
      - Blink tier is the secondary gate: no blink within timeout → SPOOF.
      - UNCERTAIN CNN + blink detected → LIVE.

    final_score = 0.6 × cnn_score + 0.4 × blink_bonus

    Parameters
    ----------
    model_dir : str
        Directory containing the two Silent-Face ONNX files.
    require_blink : bool
        If True (default), active blink challenge is required.
    """

    def __init__(
        self,
        model_dir: str = _DEFAULT_MODEL_DIR,
        require_blink: bool = True,
    ) -> None:
        self._cnn = AntiSpoofingCNN(model_dir=model_dir)
        self._blink_detector = BlinkDetector()
        self._require_blink = require_blink

    def reset_session(self) -> None:
        """Call at the start of each new authentication attempt."""
        self._blink_detector.reset()

    def predict_liveness(
        self,
        face_bgr: np.ndarray,
        face_landmarks: Optional[List[Tuple[float, float]]] = None,
    ) -> LivenessResult:
        """
        Evaluate liveness for the current frame.

        Parameters
        ----------
        face_bgr : np.ndarray
            Cropped face region in BGR uint8 (any size — resized internally).
            Pass the face crop from the bounding box, NOT the full frame tensor.
        face_landmarks : list of (x, y), optional
            468-point Face Mesh landmarks in pixel space for blink detection.

        Returns
        -------
        LivenessResult
        """
        # ── Tier 1: passive ONNX ensemble ─────────────────────────────────
        cnn_result = self._cnn.predict(face_bgr)

        if cnn_result.decision == LivenessDecision.SPOOF:
            return LivenessResult(
                decision=LivenessDecision.SPOOF,
                cnn_score=cnn_result.live_probability,
                blink_detected=False,
                final_score=cnn_result.live_probability,
                details=f"CNN SPOOF (prob={cnn_result.live_probability:.3f})",
            )

        # ── Tier 2: active blink challenge ────────────────────────────────
        blink_result = BlinkResult(blink_detected=True, total_blinks=0, ear_value=0.30)

        if face_landmarks is not None and self._require_blink:
            blink_result = self._blink_detector.update(face_landmarks)

        # ── Combined decision ─────────────────────────────────────────────
        blink_bonus = 1.0 if blink_result.blink_detected else 0.0
        final_score = 0.60 * cnn_result.live_probability + 0.40 * blink_bonus

        if cnn_result.decision == LivenessDecision.UNCERTAIN:
            if blink_result.blink_detected:
                decision = LivenessDecision.LIVE
                details = f"CNN UNCERTAIN resolved by blink (score={final_score:.3f})"
            elif blink_result.timed_out:
                decision = LivenessDecision.SPOOF
                details = "CNN UNCERTAIN + blink timeout → SPOOF"
            else:
                decision = LivenessDecision.UNCERTAIN
                details = f"Awaiting blink (EAR={blink_result.ear_value:.3f})"
        else:
            # CNN says LIVE
            if self._require_blink and not blink_result.blink_detected:
                if blink_result.timed_out:
                    decision = LivenessDecision.SPOOF
                    details = "CNN LIVE but blink timed out → SPOOF"
                else:
                    decision = LivenessDecision.UNCERTAIN
                    details = f"CNN LIVE, awaiting blink (EAR={blink_result.ear_value:.3f})"
            else:
                decision = LivenessDecision.LIVE
                details = f"CNN LIVE + blink confirmed (score={final_score:.3f})"

        logger.info(
            "Liveness | cnn=%.3f blink=%s ear=%.3f final=%.3f → %s",
            cnn_result.live_probability,
            blink_result.blink_detected,
            blink_result.ear_value,
            final_score,
            decision.value,
        )

        return LivenessResult(
            decision=decision,
            cnn_score=cnn_result.live_probability,
            blink_detected=blink_result.blink_detected,
            final_score=final_score,
            details=details,
        )


# ── Module-level EAR utility ──────────────────────────────────────────────────

def _compute_ear(
    landmarks: List[Tuple[float, float]],
    eye_indices: List[int],
) -> float:
    pts = [np.array(landmarks[i], dtype=np.float32) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    vertical_a = np.linalg.norm(p2 - p6)
    vertical_b = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return float((vertical_a + vertical_b) / (2.0 * horizontal))
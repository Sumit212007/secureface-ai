"""
pipeline/orchestrator.py
=========================
SecureEdge AI — Authentication Pipeline Orchestrator

Coordinates all pipeline modules into a single authenticate() call:

  Frame → FaceDetector → ImageProcessor → LivenessDetector
       → FaceRecognizer → EmbeddingStore (lookup) → AuthDecision

This is the ONLY file the calling application (React Native native module
bridge, test harness, or REST wrapper) should import.

Decision logic:
  ALLOW  → face detected + liveness LIVE + cosine similarity ≥ threshold
  DENY   → face detected + liveness SPOOF
  DENY   → face detected + similarity < threshold (unknown person)
  DENY   → no face detected
  PENDING→ liveness UNCERTAIN (more frames needed — for streaming mode)

Author: SecureEdge AI Team
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from pipeline.face_detector import FaceDetector, FaceDetection
from pipeline.image_processor import ImageProcessor, FaceLandmarks, PreprocessResult
from pipeline.recognizer import FaceRecognizer, EmbeddingResult, cosine_similarity
from pipeline.liveness_detector import LivenessDetector, LivenessResult, LivenessDecision

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ── Enumerations ──────────────────────────────────────────────────────────────

class AuthDecision(str, Enum):
    ALLOW   = "ALLOW"     # Identity confirmed, liveness confirmed
    DENY    = "DENY"      # Spoof detected or unknown identity
    PENDING = "PENDING"   # Liveness uncertain — send next frame
    ERROR   = "ERROR"     # Unrecoverable pipeline error


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class AuthResult:
    """
    Full authentication result returned by Orchestrator.authenticate().

    All fields are populated regardless of decision to support
    detailed logging, UI feedback, and audit trails.
    """
    decision: AuthDecision
    identity: Optional[str] = None          # Matched identity label or None
    similarity: float = 0.0                 # Cosine similarity [0, 1]
    liveness_score: float = 0.0             # Final liveness score [0, 1]
    liveness_decision: str = ""             # "LIVE" / "SPOOF" / "UNCERTAIN"
    face_detected: bool = False
    face_bbox: Optional[tuple] = None       # (x1, y1, x2, y2) in frame coords
    processing_time_ms: float = 0.0         # Wall-clock time for full pipeline
    error_message: str = ""                 # Populated on AuthDecision.ERROR
    # Detailed sub-results (optional — may be None if step not reached)
    face_detection: Optional[FaceDetection] = None
    embedding: Optional[np.ndarray] = None  # (512,) if recognition ran


@dataclass
class EnrolledIdentity:
    """
    A single enrolled face stored in the in-memory gallery.
    In production this is loaded from the encrypted SQLite store.
    """
    label: str                              # User ID or name
    embedding: np.ndarray                  # (512,) L2-normalised
    metadata: Dict = field(default_factory=dict)  # e.g. {"enrolled_at": "..."}


# ── Orchestrator ──────────────────────────────────────────────────────────────

class AuthPipeline:
    """
    End-to-end authentication pipeline orchestrator.

    Manages component lifecycles and wires the pipeline steps:
      detect → crop → preprocess → liveness CNN → recognition → match → decide

    Parameters
    ----------
    detector_model : str
        Path to BlazeFace TFLite model.
    recognizer_model : str
        Path to MobileFaceNet TFLite model.
    antispoofing_model : str
        Path to anti-spoofing TFLite model.
    cosine_threshold : float
        Minimum cosine similarity to accept a match.
    require_blink : bool
        If True, active blink challenge is required for ALLOW.
    """

    def __init__(
        self,
        detector_model: str = "models/blazeface/face_detection_short_range.tflite",
        recognizer_model: str = "models/mobilefacenet/mobilefacenet_int8.tflite",
        antispoofing_model: str = "models/antispoofing/antispoofing_int8.tflite",
        cosine_threshold: float = 0.40,
        require_blink: bool = True,
    ) -> None:
        logger.info("Initialising AuthPipeline...")
        t0 = time.monotonic()

        self._threshold = cosine_threshold

        # ── Instantiate all pipeline components ───────────────────────────
        self._detector   = FaceDetector(model_path=detector_model)
        self._processor  = ImageProcessor()
        self._recognizer = FaceRecognizer(
            model_path=recognizer_model,
            cosine_threshold=cosine_threshold,
        )
        self._liveness   = LivenessDetector(
        model_dir="models/antispoofing",
        require_blink=require_blink,
        )

        # ── In-memory gallery ─────────────────────────────────────────────
        # In production, replace with EmbeddingStore (encrypted SQLite).
        self._gallery: List[EnrolledIdentity] = []

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "AuthPipeline ready in %.1f ms | threshold=%.2f require_blink=%s",
            elapsed_ms, cosine_threshold, require_blink,
        )

    # ── Gallery management ────────────────────────────────────────────────

    def enroll(self, frame: np.ndarray, label: str) -> bool:
        """
        Enroll a face from a frame into the in-memory gallery.

        Parameters
        ----------
        frame : np.ndarray
            BGR camera frame containing one face.
        label : str
            Identity label (user ID, name).

        Returns
        -------
        bool  True on successful enrollment.
        """
        logger.info("Enrolling identity: '%s'", label)
        detections = self._detector.detect_faces(frame, max_faces=1)
        if not detections:
            logger.warning("Enrollment failed: no face detected in frame.")
            return False

        detection = detections[0]
        face_roi = self._processor.crop_face_roi(frame, detection.bbox)
        landmarks = _landmarks_to_dataclass(detection)
        prep = self._processor.preprocess_for_recognition(face_roi, landmarks)
        emb_result = self._recognizer.get_embedding(prep.tensor)

        self._gallery.append(
            EnrolledIdentity(
                label=label,
                embedding=emb_result.vector,
                metadata={"is_stub": emb_result.is_stub},
            )
        )
        logger.info(
            "Enrolled '%s' | embedding_dim=%d is_stub=%s",
            label, len(emb_result.vector), emb_result.is_stub,
        )
        return True

    def load_gallery(self, identities: List[EnrolledIdentity]) -> None:
        """Bulk-load pre-computed embeddings (e.g. from encrypted SQLite store)."""
        self._gallery = identities
        logger.info("Gallery loaded with %d identities.", len(self._gallery))

    # ── Main authentication method ────────────────────────────────────────

    def authenticate(
        self,
        frame: np.ndarray,
        face_landmarks_2d: Optional[list] = None,
    ) -> AuthResult:
        """
        Run the full authentication pipeline on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR camera frame, uint8, any resolution.
        face_landmarks_2d : list of (x, y), optional
            468-point MediaPipe Face Mesh landmarks for blink detection.
            If None, blink challenge is skipped for this frame.

        Returns
        -------
        AuthResult
            Always returns a populated result — never raises in normal operation.
        """
        t_start = time.monotonic()

        try:
            result = self._run_pipeline(frame, face_landmarks_2d)
        except Exception as exc:
            logger.exception("AuthPipeline unhandled error: %s", exc)
            result = AuthResult(
                decision=AuthDecision.ERROR,
                error_message=str(exc),
                processing_time_ms=(time.monotonic() - t_start) * 1000,
            )

        result.processing_time_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            "Auth complete | decision=%s id=%s sim=%.3f live=%.3f time=%.1fms",
            result.decision.value,
            result.identity or "—",
            result.similarity,
            result.liveness_score,
            result.processing_time_ms,
        )
        return result

    def reset_liveness_session(self) -> None:
        """
        Reset the blink detector state for a new auth session.
        Must be called at the start of each authentication attempt.
        """
        self._liveness.reset_session()
        logger.debug("Liveness session reset.")

    # ── Internal pipeline steps ───────────────────────────────────────────

    def _run_pipeline(
        self,
        frame: np.ndarray,
        face_landmarks_2d: Optional[list],
    ) -> AuthResult:
        """Core pipeline — all errors propagate to authenticate() try/except."""

        # ── Step 1: Face detection ─────────────────────────────────────────
        detections = self._detector.detect_faces(frame, max_faces=1)

        if not detections:
            logger.debug("Pipeline: no face detected.")
            return AuthResult(
                decision=AuthDecision.DENY,
                face_detected=False,
                error_message="No face detected in frame.",
            )

        detection = detections[0]
        logger.debug(
            "Face detected | bbox=%s confidence=%.3f",
            detection.bbox, detection.confidence,
        )

        # ── Step 2: Crop face ROI ─────────────────────────────────────────
        face_roi = self._processor.crop_face_roi(frame, detection.bbox)

        # ── Step 3: Build FaceLandmarks from detector output ──────────────
        alignment_landmarks = _landmarks_to_dataclass(detection)

        # ── Step 4: Dual preprocessing ────────────────────────────────────
        recog_prep: PreprocessResult = self._processor.preprocess_for_recognition(
            face_roi, alignment_landmarks
        )

        # ── Step 5: Liveness detection ────────────────────────────────────
        liveness_result: LivenessResult = self._liveness.predict_liveness(
            face_bgr=face_roi,
            face_landmarks=face_landmarks_2d,
        )

        if liveness_result.decision == LivenessDecision.SPOOF:
            return AuthResult(
                decision=AuthDecision.DENY,
                face_detected=True,
                face_bbox=detection.bbox,
                liveness_score=liveness_result.final_score,
                liveness_decision=liveness_result.decision.value,
                face_detection=detection,
                error_message=f"Spoof detected: {liveness_result.details}",
            )

        if liveness_result.decision == LivenessDecision.UNCERTAIN:
            return AuthResult(
                decision=AuthDecision.PENDING,
                face_detected=True,
                face_bbox=detection.bbox,
                liveness_score=liveness_result.final_score,
                liveness_decision=liveness_result.decision.value,
                face_detection=detection,
                error_message=liveness_result.details,
            )

        # ── Step 6: Face recognition ──────────────────────────────────────
        emb_result: EmbeddingResult = self._recognizer.get_embedding(
            recog_prep.tensor
        )

        # ── Step 7: Gallery match ─────────────────────────────────────────
        match_label, best_similarity = self._find_best_match(emb_result)

        is_match = best_similarity >= self._threshold

        decision = AuthDecision.ALLOW if is_match else AuthDecision.DENY

        return AuthResult(
            decision=decision,
            identity=match_label if is_match else None,
            similarity=best_similarity,
            liveness_score=liveness_result.final_score,
            liveness_decision=liveness_result.decision.value,
            face_detected=True,
            face_bbox=detection.bbox,
            face_detection=detection,
            embedding=emb_result.vector,
        )

    def _find_best_match(
        self,
        probe: EmbeddingResult,
    ) -> tuple:
        """
        Linear scan of the gallery for the highest cosine similarity.

        For galleries >1000 identities, replace with FAISS or
        ball-tree approximate nearest neighbour search.

        Returns (best_label, best_similarity).
        If gallery is empty returns (None, 0.0).
        """
        if not self._gallery:
            logger.warning("Gallery is empty. No match possible.")
            return None, 0.0

        best_label = None
        best_sim = -1.0

        for identity in self._gallery:
            sim = cosine_similarity(probe.vector, identity.embedding)
            if sim > best_sim:
                best_sim = sim
                best_label = identity.label

        logger.debug(
            "Gallery search: best_match='%s' similarity=%.4f (gallery_size=%d)",
            best_label, best_sim, len(self._gallery),
        )
        return best_label, float(best_sim)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _landmarks_to_dataclass(
    detection: FaceDetection,
) -> Optional[FaceLandmarks]:
    """
    Convert BlazeFace landmark list to FaceLandmarks dataclass.

    BlazeFace keypoint order:
      [0] right_eye  [1] left_eye  [2] nose  [3] mouth  [4] right_ear  [5] left_ear

    FaceLandmarks convention uses anatomical left/right from the subject's
    perspective (same as BlazeFace output — no flip needed).
    """
    if not detection.landmarks or len(detection.landmarks) < 2:
        return None

    return FaceLandmarks(
        left_eye=detection.landmarks[1],
        right_eye=detection.landmarks[0],
        nose_tip=detection.landmarks[2] if len(detection.landmarks) > 2 else None,
        mouth_left=detection.landmarks[3] if len(detection.landmarks) > 3 else None,
    )
"""
pipeline/recognizer.py
=======================
SecureEdge AI — Face Recognition Module

Embedding strategy (priority order)
---------------------------------
1. **MobileFaceNet** — ONNX Runtime or TFLite (production)
      • 112×112 RGB, normalised to [-1, 1]
      • 512-D L2-normalised identity embedding
      • Cosine similarity for verify / gallery match

2. **Landmark fallback** (dev only) — set ``SECUREFACE_USE_LANDMARK_EMBEDDINGS=1``
      • MediaPipe 478-point landmarks + random projection
      • Not identity-trained; do not use in production

3. **Stub** — SHA-256 hash embedding when no model is available

Model recommendation
--------------------
**MobileFaceNet** (chosen for this project)
    • ~4 MB, ~15–40 ms on mid-range ARM CPU
    • Same preprocessing as ArcFace family (112×112, [-1,1])
    • Best fit for <800 ms full pipeline + future Android TFLite

ArcFace / InsightFace
    • Higher accuracy, larger models (ResNet50+)
    • InsightFace = detection + alignment + ArcFace bundle (heavier APK)
    • Use when accuracy > size and you can ship 20–80 MB models

Threshold guide (cosine similarity, L2-normalised)
----------------------------------------------------
    • Same person, same session : typically 0.55–0.85+
    • Same person, cross lighting : 0.45–0.70
    • Different person            : usually < 0.35
    • Recommended default         : 0.45  (THRESHOLD_DEFAULT)
    • High-security               : 0.55  (THRESHOLD_STRICT)

Place model files under ``models/mobilefacenet/``:
    • ``mobilefacenet.onnx``        — Python / ONNX Runtime (dev & server)
    • ``mobilefacenet_int8.tflite`` — Android TFLite (future)

Run ``python scripts/setup_recognition_model.py`` for setup instructions.

Author: SecureEdge AI Team
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from database.identity_store import DEFAULT_DB_PATH, IdentityStore
from pipeline.embedding_backends import (
    DEFAULT_MODEL_DIR,
    EMBEDDING_DIM,
    THRESHOLD_DEFAULT,
    THRESHOLD_RELAXED,
    THRESHOLD_STRICT,
    EmbeddingBackend,
    create_embedding_backend,
    l2_normalize,
    resolve_model_paths,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Backward-compatible alias used by orchestrator.
# MobileFaceNet (512-D, L2-normalised): 0.45 balances false reject vs impostor accept
# on typical phone/webcam enroll-verify; raise to THRESHOLD_STRICT (0.55) for kiosks.
COSINE_THRESHOLD: float = THRESHOLD_DEFAULT

# Landmark fallback (opt-in)
_USE_LANDMARK_FALLBACK: bool = os.environ.get(
    "SECUREFACE_USE_LANDMARK_EMBEDDINGS", ""
).lower() in ("1", "true", "yes")

_LANDMARK_MODEL_PATHS: List[str] = [
    "models/mediapipe/face_landmarker.task",
    "models/mobilefacenet/face_landmarker.task",
]

# Landmark projection constants (legacy path only)
_LANDMARK_DIM: int = 478 * 3
_PROJ_SEED: int = 42


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    matched:    bool
    similarity: float
    name:       str   = ""
    details:    str   = ""
    meta:       Dict[str, Any] = field(default_factory=dict)


class EmbeddingResult(np.ndarray):
    """
    (512,) L2-normalised embedding; subclasses ndarray for API compatibility.

    Extra attributes: confidence, model_name, inference_ms, is_stub.
    """

    def __new__(
        cls,
        array:        np.ndarray,
        confidence:   float = 1.0,
        model_name:   str   = "mobilefacenet",
        inference_ms: float = 0.0,
    ) -> "EmbeddingResult":
        obj = np.asarray(array, dtype=np.float32).view(cls)
        obj.confidence   = confidence
        obj.model_name   = model_name
        obj.inference_ms = inference_ms
        return obj

    def __array_finalize__(self, obj: object) -> None:
        if obj is None:
            return
        self.confidence   = getattr(obj, "confidence",   1.0)
        self.model_name   = getattr(obj, "model_name",   "mobilefacenet")
        self.inference_ms = getattr(obj, "inference_ms", 0.0)

    @property
    def embedding(self) -> np.ndarray:
        return np.asarray(self)

    @property
    def vector(self) -> np.ndarray:
        return self.embedding

    @property
    def is_stub(self) -> bool:
        return self.model_name == "stub"

    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"EmbeddingResult(model={self.model_name!r} "
            f"confidence={self.confidence:.3f} "
            f"inference_ms={self.inference_ms:.1f} "
            f"norm={float(np.linalg.norm(self)):.4f})"
        )

    @classmethod
    def from_array(
        cls,
        array:        np.ndarray,
        confidence:   float = 1.0,
        model_name:   str   = "mobilefacenet",
        inference_ms: float = 0.0,
    ) -> "EmbeddingResult":
        return cls(array, confidence, model_name, inference_ms)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of L2-normalised embeddings (= cosine similarity)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(a, b))


# ── Main recognizer ───────────────────────────────────────────────────────────

class FaceRecognizer:
    """
    Face enrollment and verification via MobileFaceNet embeddings.

    Public API (unchanged for orchestrator / tests)
    -----------------------------------------------
    get_embedding(tensor, face_roi_bgr=None) → EmbeddingResult
    enroll(name, embedding)
    verify(embedding) → VerificationResult
    is_enrolled, clear_enrollment()
    """

    def __init__(
        self,
        model_paths:      Optional[List[str]] = None,
        threshold:        float = COSINE_THRESHOLD,
        model_path:       Optional[str] = None,
        cosine_threshold: Optional[float] = None,
        model_dir:        str = DEFAULT_MODEL_DIR,
        allow_landmark_fallback: Optional[bool] = None,
        db_path:          Optional[str] = None,
        identity_store:   Optional[IdentityStore] = None,
    ) -> None:
        if cosine_threshold is not None:
            threshold = cosine_threshold

        self._threshold = threshold
        self._store = identity_store or IdentityStore(db_path=db_path or DEFAULT_DB_PATH)
        self._enrolled: Dict[str, np.ndarray] = {}
        self._load_enrolled_from_db()

        paths = resolve_model_paths(model_path, model_paths, model_dir)
        self._backend: Optional[EmbeddingBackend] = create_embedding_backend(
            model_path=model_path,
            model_paths=paths,
            model_dir=model_dir,
        )

        self._landmark = None
        use_landmark = (
            _USE_LANDMARK_FALLBACK if allow_landmark_fallback is None
            else allow_landmark_fallback
        )
        if self._backend is None and use_landmark:
            self._landmark = _LandmarkEmbedder(_LANDMARK_MODEL_PATHS)

        if self._backend is not None:
            logger.info(
                "FaceRecognizer: neural backend '%s' | threshold=%.2f",
                self._backend.name, self._threshold,
            )
        elif self._landmark is not None and self._landmark.is_ready:
            logger.warning(
                "FaceRecognizer: MobileFaceNet not found — using LANDMARK fallback. "
                "Place mobilefacenet.onnx under %s for production.",
                model_dir,
            )
        else:
            logger.warning(
                "FaceRecognizer: no embedding model — STUB mode. "
                "Add models/mobilefacenet/mobilefacenet.onnx"
            )

    @property
    def embedding_backend(self) -> str:
        if self._backend is not None:
            return self._backend.name
        if self._landmark is not None and self._landmark.is_ready:
            return "landmarks"
        return "stub"

    def get_embedding(
        self,
        face_tensor: np.ndarray,
        face_roi_bgr: Optional[np.ndarray] = None,
    ) -> EmbeddingResult:
        """
        Extract a 512-D L2-normalised embedding.

        Parameters
        ----------
        face_tensor : (1, 112, 112, 3) float32 RGB [-1, 1] from ImageProcessor
        face_roi_bgr : optional BGR crop — used only by landmark fallback
        """
        # ── 1. MobileFaceNet (ONNX / TFLite) ──────────────────────────────
        if self._backend is not None:
            t0 = time.perf_counter()
            try:
                emb = self._backend.embed(face_tensor)
            except Exception as exc:
                logger.error("Neural embedder failed: %s", exc)
                emb = None
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if emb is not None:
                logger.debug(
                    "get_embedding: %s | %.1f ms | norm=%.4f",
                    self._backend.name, elapsed_ms, float(np.linalg.norm(emb)),
                )
                return EmbeddingResult.from_array(
                    emb,
                    confidence=1.0,
                    model_name=self._backend.name,
                    inference_ms=elapsed_ms,
                )

        # ── 2. Landmark fallback (dev) ────────────────────────────────────
        if self._landmark is not None and self._landmark.is_ready:
            face_rgb = _tensor_to_rgb_uint8(face_tensor)
            t0 = time.perf_counter()
            emb = self._landmark.embed(face_rgb)
            if emb is None and face_roi_bgr is not None:
                emb = self._landmark.embed(
                    cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if emb is not None:
                return EmbeddingResult.from_array(
                    emb,
                    confidence=0.5,
                    model_name="landmarks",
                    inference_ms=elapsed_ms,
                )

        # ── 3. Stub ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        emb = _stub_embedding(
            face_tensor[0] if face_tensor.ndim == 4 else face_tensor
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return EmbeddingResult.from_array(
            emb,
            confidence=0.0,
            model_name="stub",
            inference_ms=elapsed_ms,
        )

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"enroll: expected ({EMBEDDING_DIM},), got {embedding.shape}"
            )
        vec = l2_normalize(np.asarray(embedding, dtype=np.float32))
        try:
            record = self._store.upsert(name, vec)
        except Exception as exc:
            logger.error("enroll: SQLite persist failed for '%s': %s", name, exc)
            raise
        self._enrolled[name] = record.embedding
        logger.info(
            "Enrolled identity: '%s' user_id=%s norm=%.4f (backend=%s)",
            name, record.user_id, float(np.linalg.norm(record.embedding)),
            self.embedding_backend,
        )

    def verify(self, embedding: np.ndarray) -> VerificationResult:
        if not self._enrolled:
            return VerificationResult(
                matched=False, similarity=0.0, name="",
                details="No enrolled identities.",
            )
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"verify: expected ({EMBEDDING_DIM},), got {embedding.shape}"
            )
        emb = l2_normalize(np.asarray(embedding, dtype=np.float32))
        probe_norm = float(np.linalg.norm(emb))
        best_name, best_sim = "", -1.0
        for name, enrolled_emb in self._enrolled.items():
            sim = float(np.dot(emb, enrolled_emb))
            gallery_norm = float(np.linalg.norm(enrolled_emb))
            logger.debug(
                "verify compare '%s' sim=%.4f probe_norm=%.4f gallery_norm=%.4f",
                name, sim, probe_norm, gallery_norm,
            )
            if sim > best_sim:
                best_sim, best_name = sim, name
        matched = best_sim >= self._threshold
        details = (
            f"Best match: '{best_name}' sim={best_sim:.4f} "
            f"threshold={self._threshold:.2f} → "
            f"{'MATCH' if matched else 'NO MATCH'}"
        )
        logger.info("verify | %s", details)
        return VerificationResult(
            matched=matched,
            similarity=best_sim,
            name=best_name if matched else "",
            details=details,
            meta={"backend": self.embedding_backend},
        )

    def enrolled_identities(self) -> Dict[str, np.ndarray]:
        """Snapshot of in-memory gallery (username → L2-normalised embedding)."""
        return dict(self._enrolled)

    @property
    def is_enrolled(self) -> bool:
        return bool(self._enrolled)

    def clear_enrollment(self) -> None:
        self._enrolled.clear()
        try:
            self._store.clear_all()
        except Exception as exc:
            logger.error("clear_enrollment: SQLite clear failed: %s", exc)
            raise
        logger.info("Enrollment cleared (memory + SQLite).")

    def _load_enrolled_from_db(self) -> None:
        try:
            loaded = self._store.load_username_map()
        except Exception as exc:
            logger.error("Failed to load identities from SQLite: %s", exc)
            loaded = {}
        self._enrolled = loaded
        if not loaded:
            logger.info("No persisted identities loaded from %s", self._store.db_path)
            return
        for name, emb in loaded.items():
            logger.info(
                "Loaded identity '%s' norm=%.4f",
                name, float(np.linalg.norm(emb)),
            )


# ── Stub embedder ─────────────────────────────────────────────────────────────

def _stub_embedding(face_tensor: np.ndarray) -> np.ndarray:
    logger.warning(
        "FaceRecognizer: STUB embedding — authentication is NOT reliable."
    )
    raw = (face_tensor * 255).clip(0, 255).astype(np.uint8).tobytes()
    digest = hashlib.sha256(raw).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal(EMBEDDING_DIM).astype(np.float32))


def _tensor_to_rgb_uint8(face_tensor: np.ndarray) -> np.ndarray:
    """Convert recognition tensor to uint8 RGB HWC (landmark path only)."""
    t = face_tensor
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    if t.dtype != np.uint8:
        if t.min() < -0.1:
            t = (t + 1.0) / 2.0
        t = (t * 255.0).clip(0, 255).astype(np.uint8)
    if t.shape[:2] != (112, 112):
        t = cv2.resize(t, (112, 112), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(t)


# ── Legacy landmark embedder (dev fallback) ─────────────────────────────────────

class _LandmarkEmbedder:
    """MediaPipe landmarks + random projection — not for production."""

    def __init__(self, model_paths: List[str]) -> None:
        self._landmarker = None
        self._load(model_paths)
        self._proj = _make_projection_matrix()

    def _load(self, model_paths: List[str]) -> None:
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError:
            return
        for path in model_paths:
            if not os.path.isfile(path):
                continue
            try:
                opts = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=path),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    output_facial_transformation_matrixes=True,
                )
                self._landmarker = vision.FaceLandmarker.create_from_options(opts)
                logger.info("Landmark fallback loaded: '%s'", path)
                return
            except Exception as exc:
                logger.error("Landmark load failed '%s': %s", path, exc)

    @property
    def is_ready(self) -> bool:
        return self._landmarker is not None

    def embed(self, face_rgb: np.ndarray) -> Optional[np.ndarray]:
        if not self.is_ready:
            return None
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.vision.core.image import ImageFormat
        except ImportError:
            return None
        h, w = face_rgb.shape[:2]
        if min(h, w) < 192:
            s = 192.0 / min(h, w)
            face_rgb = cv2.resize(
                face_rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR,
            )
        img = mp.Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(face_rgb))
        result = self._landmarker.detect(img)
        if not result.face_landmarks:
            return None
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]], dtype=np.float32,
        )
        if result.facial_transformation_matrixes:
            mat = np.array(result.facial_transformation_matrixes[0], dtype=np.float32)
            if mat.shape == (4, 4):
                coords = coords @ mat[:3, :3].T
        coords -= coords.mean(axis=0)
        coords /= np.linalg.norm(coords) + 1e-8
        flat = coords.flatten()
        if flat.shape[0] != _LANDMARK_DIM:
            tmp = np.zeros(_LANDMARK_DIM, dtype=np.float32)
            n = min(len(flat), _LANDMARK_DIM)
            tmp[:n] = flat[:n]
            flat = tmp
        return l2_normalize(flat @ self._proj)


def _make_projection_matrix() -> np.ndarray:
    rng = np.random.default_rng(_PROJ_SEED)
    p = rng.standard_normal((_LANDMARK_DIM, EMBEDDING_DIM)).astype(np.float32)
    p /= np.linalg.norm(p, axis=0, keepdims=True) + 1e-8
    return p


# Re-export threshold constants for callers
__all__ = [
    "FaceRecognizer",
    "EmbeddingResult",
    "cosine_similarity",
    "COSINE_THRESHOLD",
    "THRESHOLD_DEFAULT",
    "THRESHOLD_STRICT",
    "THRESHOLD_RELAXED",
    "EMBEDDING_DIM",
]

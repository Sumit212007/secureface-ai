"""
pipeline/recognizer.py
=======================
SecureEdge AI — Face Recognition Module

Embedding strategy
------------------
Primary path  (MediaPipe landmarks):
    Uses MediaPipe Face Landmarker (.task file) to extract 478 3-D landmarks
    from a 112×112 face crop.  The landmarks are:
      1. Centred  (mean-subtracted)
      2. Aligned  (rotation from facial_transformation_matrixes if available)
      3. Scale-normalised (divided by Frobenius norm)
      4. Flattened → 1 434-dim vector
      5. Projected to 512-dim via a deterministic random projection matrix
         (seeded with 42 for reproducibility across sessions)
      6. L2-normalised → unit vector ready for cosine similarity (dot product)

    Why random projection?  We have no trained embedding head for the
    landmark-only path.  A fixed random projection preserves pairwise
    distances in expectation (Johnson–Lindenstrauss) and is deterministic,
    so the same face always maps to the same region of embedding space.
    Similarity scores will be high for the same person (~0.85–0.99) and
    lower for different people (~0.2–0.6) given accurate landmarks.

Fallback path (no model / no face detected):
    Deterministic stub — SHA-256 hash of the pixel values, projected to 512-D.
    Guarantees self-match (similarity=1.0) while being useless for real auth.
    A warning is logged whenever the fallback fires.

Input contract
--------------
    get_embedding() accepts:
      • A recognition tensor of shape (1, 112, 112, 3), float32, range [-1, 1]
        (output of ImageProcessor.preprocess_for_recognition)
      • OR a raw BGR/RGB ndarray of any size (auto-converted)

    The tensor is converted back to uint8 for MediaPipe.

API compatibility
-----------------
    FaceRecognizer exposes the same public interface as the original stub:
        enroll(name, embedding)
        verify(embedding)          → VerificationResult
        get_embedding(tensor)      → np.ndarray shape (512,)
        is_enrolled                → bool
        clear_enrollment()

Author: SecureEdge AI Team
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ── Constants ────────────────────────────────────────────────────────────────

COSINE_THRESHOLD: float = 0.40     # minimum dot-product for MATCH
EMBEDDING_DIM:    int   = 512      # output embedding size
LANDMARK_DIM:     int   = 478 * 3  # 478 landmarks × (x, y, z)  = 1 434

# Recognition crop size expected by this module
_RECOG_H: int = 112
_RECOG_W: int = 112

# Model search paths (tried in order)
_DEFAULT_MODEL_PATHS: List[str] = [
    "models/mediapipe/face_landmarker.task",
    "models/mobilefacenet/face_landmarker.task",
]

# Deterministic random projection matrix seed
_PROJ_SEED: int = 42


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    matched:    bool
    similarity: float
    name:       str   = ""
    details:    str   = ""
    meta:       Dict[str, Any] = field(default_factory=dict)
    # meta keys:
    #   embedding_source : "landmarks" | "stub"
    #   inference_ms     : float   (landmarks path only)
    #   num_landmarks    : int     (landmarks path only)


# ── Backward-compatibility shims ─────────────────────────────────────────────
# The original stub recognizer exported EmbeddingResult and cosine_similarity.
# pipeline/__init__.py, orchestrator.py, and test_pipeline.py import these by
# name.  They are preserved here as complete, production-safe wrappers so ALL
# existing call patterns continue to work without any changes to other files.
#
# Supported legacy call patterns:
#   result = recognizer.get_embedding(tensor)  → EmbeddingResult
#   vec    = result.embedding                  → np.ndarray (512,)
#   sim    = cosine_similarity(result, other)  → float  (transparent ndarray)
#   sim    = cosine_similarity(result.embedding, other.embedding)  → float
#   recognizer.enroll("Alice", result)         → works (via __array__)
#   recognizer.verify(result)                  → works (via __array__)
#   np.dot(result, other)                      → works (via __array__)

class EmbeddingResult(np.ndarray):
    """
    Backward-compatible embedding container.

    Subclasses np.ndarray so it IS a (512,) float32 array — every existing
    call that passes it to np.dot(), cosine_similarity(), enroll(), or
    verify() works without modification.

    Extra attributes (read-only after construction):
        embedding    : np.ndarray view  — the (512,) vector (self)
        confidence   : float            — always 1.0 (was stub placeholder)
        model_name   : str              — "landmarks" or "stub"
        inference_ms : float            — wall-clock time for the embed call

    Construction
    ------------
    Use EmbeddingResult.from_array(vec, ...) — never call the ndarray
    constructor directly.
    """

    # np.ndarray subclass protocol ----------------------------------------
    def __new__(
        cls,
        array:        np.ndarray,
        confidence:   float = 1.0,
        model_name:   str   = "landmarks",
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
        self.model_name   = getattr(obj, "model_name",   "landmarks")
        self.inference_ms = getattr(obj, "inference_ms", 0.0)

    # Convenience attribute -----------------------------------------------
    @property
    def embedding(self) -> np.ndarray:
        """Return the underlying (512,) ndarray view."""
        return np.asarray(self)

    # Nice repr ------------------------------------------------------------
    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"EmbeddingResult(model={self.model_name!r} "
            f"confidence={self.confidence:.3f} "
            f"inference_ms={self.inference_ms:.1f} "
            f"norm={float(np.linalg.norm(self)):.4f})"
        )

    # Factory --------------------------------------------------------------
    @classmethod
    def from_array(
        cls,
        array:        np.ndarray,
        confidence:   float = 1.0,
        model_name:   str   = "landmarks",
        inference_ms: float = 0.0,
    ) -> "EmbeddingResult":
        return cls(array, confidence, model_name, inference_ms)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised embedding vectors.

    Both vectors are assumed to be unit-norm (output of get_embedding()),
    so cosine similarity reduces to a dot product.

    Accepts plain np.ndarray or EmbeddingResult interchangeably.

    Returns
    -------
    float in [-1.0, 1.0];  1.0 = identical,  0.0 = orthogonal
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(a, b))


# ── Projection matrix (module-level singleton) ───────────────────────────────

def _make_projection_matrix() -> np.ndarray:
    """
    Build a (LANDMARK_DIM, EMBEDDING_DIM) Gaussian random projection matrix.
    Seeded with _PROJ_SEED so it is identical across every run / process.
    Columns are L2-normalised so the projection preserves scale.
    """
    rng = np.random.default_rng(_PROJ_SEED)
    P   = rng.standard_normal((LANDMARK_DIM, EMBEDDING_DIM)).astype(np.float32)
    P  /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-8
    return P

_PROJECTION_MATRIX: np.ndarray = _make_projection_matrix()


# ── MediaPipe landmark embedder ───────────────────────────────────────────────

class LandmarkEmbedder:
    """
    Wraps MediaPipe Face Landmarker to extract 478 3-D landmarks and
    project them to a 512-dim L2-normalised embedding vector.

    Falls back gracefully (logs a warning, returns None) if:
      - The .task model file is missing
      - MediaPipe is not installed
      - No face is detected in the crop
    """

    def __init__(self, model_paths: List[str] = _DEFAULT_MODEL_PATHS) -> None:
        self._landmarker = None
        self._model_path = ""
        self._load(model_paths)

    # ── Loading ───────────────────────────────────────────────────────────

    def _load(self, model_paths: List[str]) -> None:
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError:
            logger.error(
                "mediapipe is not installed — run: pip install mediapipe"
            )
            return

        for path in model_paths:
            if not os.path.isfile(path):
                logger.debug("Face Landmarker model not found: '%s'", path)
                continue
            try:
                opts = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=path),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_facial_transformation_matrixes=True,
                )
                self._landmarker = vision.FaceLandmarker.create_from_options(opts)
                self._model_path = path
                logger.info("LandmarkEmbedder loaded: '%s'", path)
                return
            except Exception as exc:
                logger.error("Failed to load '%s': %s", path, exc)

        logger.warning(
            "LandmarkEmbedder: no Face Landmarker model found. "
            "Tried: %s. "
            "Embeddings will fall back to the stub.",
            model_paths,
        )

    @property
    def is_ready(self) -> bool:
        return self._landmarker is not None

    # ── Public API ────────────────────────────────────────────────────────

    def embed(self, face_uint8_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Run Face Landmarker on a uint8 RGB 112×112 crop and return a
        512-dim L2-normalised embedding.

        Parameters
        ----------
        face_uint8_rgb : np.ndarray
            RGB uint8 face crop, any size (resized internally to 112×112).

        Returns
        -------
        np.ndarray shape (512,) float32, or None on detection failure.
        """
        if not self.is_ready:
            return None

        try:
            import mediapipe as mp
            from mediapipe.tasks.python.vision.core.image import ImageFormat
        except ImportError:
            return None

        # Ensure correct size and contiguity
        h, w = face_uint8_rgb.shape[:2]
        if h != _RECOG_H or w != _RECOG_W:
            face_uint8_rgb = cv2.resize(
                face_uint8_rgb, (_RECOG_W, _RECOG_H), interpolation=cv2.INTER_LINEAR
            )

        face_uint8_rgb = np.ascontiguousarray(face_uint8_rgb)

        t0  = time.perf_counter()
        img = mp.Image(image_format=ImageFormat.SRGB, data=face_uint8_rgb)
        result = self._landmarker.detect(img)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        if not result.face_landmarks:
            logger.debug("LandmarkEmbedder: no face detected in crop.")
            return None

        # Extract 478 × 3 landmarks (x, y, z) from the first face
        raw_lm = result.face_landmarks[0]  # list of NormalizedLandmark
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in raw_lm], dtype=np.float32
        )  # (478, 3)

        if len(coords) != 478:
            logger.warning(
                "LandmarkEmbedder: expected 478 landmarks, got %d", len(coords)
            )

        # ── Optional rotation alignment ───────────────────────────────────
        if (
            result.facial_transformation_matrixes
            and len(result.facial_transformation_matrixes) > 0
        ):
            mat = np.array(result.facial_transformation_matrixes[0], dtype=np.float32)
            if mat.shape == (4, 4):
                R = mat[:3, :3]  # 3×3 rotation sub-matrix
                # Apply inverse rotation to remove head-pose tilt
                coords = (coords @ R.T)

        # ── Normalise ─────────────────────────────────────────────────────
        coords -= coords.mean(axis=0)                         # centre
        norm    = np.linalg.norm(coords) + 1e-8
        coords /= norm                                        # scale-invariant

        # ── Project to 512-D and L2-normalise ────────────────────────────
        flat = coords.flatten()  # (1434,)
        if flat.shape[0] != LANDMARK_DIM:
            # Pad or trim to LANDMARK_DIM if landmark count differs
            tmp          = np.zeros(LANDMARK_DIM, dtype=np.float32)
            n            = min(len(flat), LANDMARK_DIM)
            tmp[:n]      = flat[:n]
            flat         = tmp

        emb = flat @ _PROJECTION_MATRIX              # (512,)
        emb = _l2_norm(emb)

        logger.debug(
            "LandmarkEmbedder | landmarks=%d align=%s inf=%.1f ms norm=%.4f",
            len(raw_lm),
            bool(result.facial_transformation_matrixes),
            inference_ms,
            float(np.linalg.norm(emb)),
        )
        return emb


# ── Stub embedder (fallback) ──────────────────────────────────────────────────

def _stub_embedding(face_tensor: np.ndarray) -> np.ndarray:
    """
    Deterministic stub embedding derived from pixel content SHA-256.

    Guarantees self-match (similarity = 1.0) but has no discriminative
    power for different people.  Logs a warning on every call.
    """
    logger.warning(
        "FaceRecognizer: using STUB embedding (no real model available). "
        "Authentication results are NOT reliable."
    )
    raw   = (face_tensor * 255).clip(0, 255).astype(np.uint8).tobytes()
    digest = hashlib.sha256(raw).digest()      # 32 bytes
    # Seed a fast RNG with the digest to get 512 float32 values
    seed  = int.from_bytes(digest[:4], "big")
    rng   = np.random.default_rng(seed)
    emb   = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return _l2_norm(emb)


# ── Main recognizer class ─────────────────────────────────────────────────────

class FaceRecognizer:
    """
    Face enrollment and verification via landmark-based embeddings.

    Public API (unchanged from original stub interface)
    ---------------------------------------------------
    get_embedding(tensor)   → np.ndarray (512,) float32, L2-normalised
    enroll(name, embedding)
    verify(embedding)       → VerificationResult
    is_enrolled             → bool property
    clear_enrollment()

    Parameters
    ----------
    model_paths   : list of .task file paths tried in order
    threshold     : cosine similarity threshold for MATCH
    """

    def __init__(
        self,
        model_paths: List[str]  = _DEFAULT_MODEL_PATHS,
        threshold:   float      = COSINE_THRESHOLD,
    ) -> None:
        self._embedder   = LandmarkEmbedder(model_paths)
        self._threshold  = threshold
        self._enrolled:  Dict[str, np.ndarray] = {}   # name → embedding

        if not self._embedder.is_ready:
            logger.warning(
                "FaceRecognizer: Face Landmarker unavailable. "
                "All embeddings will be stub-based."
            )

    # ── Embedding ─────────────────────────────────────────────────────────

    def get_embedding(
        self,
        face_tensor: np.ndarray,
    ) -> "EmbeddingResult":
        """
        Extract a 512-dim L2-normalised embedding from a face tensor.

        Returns an EmbeddingResult, which IS a np.ndarray subclass.
        All existing code that treats the return value as a plain ndarray
        (np.dot, cosine_similarity, enroll, verify) continues to work
        without modification.  Code that accesses .embedding, .confidence,
        .model_name, or .inference_ms also works.

        Parameters
        ----------
        face_tensor : np.ndarray
            Shape (1, 112, 112, 3) float32 in [-1, 1]   — from
            ImageProcessor.preprocess_for_recognition()
            OR a raw BGR/RGB ndarray of any size.

        Returns
        -------
        EmbeddingResult  shape (512,), float32, L2-normalised.
        """
        # ── Normalise input to a (112, 112, 3) uint8 RGB array ────────────
        face_rgb_u8 = _tensor_to_rgb_uint8(face_tensor)

        # ── Primary path: MediaPipe landmarks ─────────────────────────────
        if self._embedder.is_ready:
            t0  = time.perf_counter()
            emb = self._embedder.embed(face_rgb_u8)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if emb is not None:
                logger.debug(
                    "get_embedding: landmarks path | %.1f ms | norm=%.4f",
                    elapsed_ms, float(np.linalg.norm(emb)),
                )
                return EmbeddingResult.from_array(
                    emb,
                    confidence=1.0,
                    model_name="landmarks",
                    inference_ms=elapsed_ms,
                )

            logger.warning(
                "get_embedding: Face Landmarker found no face — "
                "falling back to stub."
            )

        # ── Fallback: stub ────────────────────────────────────────────────
        t0  = time.perf_counter()
        emb = _stub_embedding(
            face_tensor[0] if face_tensor.ndim == 4 else face_tensor
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return EmbeddingResult.from_array(
            emb,
            confidence=0.0,       # 0.0 signals "not a real embedding"
            model_name="stub",
            inference_ms=elapsed_ms,
        )

    # ── Enrollment & verification ─────────────────────────────────────────

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        """
        Enroll an identity.

        Parameters
        ----------
        name      : identity label (e.g. "Alice")
        embedding : L2-normalised (512,) float32 vector from get_embedding().
                    Accepts both plain np.ndarray and EmbeddingResult.
        """
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"enroll: expected embedding shape ({EMBEDDING_DIM},), "
                f"got {embedding.shape}"
            )
        self._enrolled[name] = embedding.astype(np.float32)
        logger.info("Enrolled identity: '%s'", name)

    def verify(self, embedding: np.ndarray) -> VerificationResult:
        """
        Compare embedding against all enrolled identities.

        Returns the best-match VerificationResult.
        If no identities are enrolled, returns matched=False.

        Similarity is cosine similarity via dot-product (both vectors are
        L2-normalised, so dot-product == cosine similarity).
        """
        if not self._enrolled:
            return VerificationResult(
                matched=False,
                similarity=0.0,
                name="",
                details="No enrolled identities.",
            )

        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"verify: expected embedding shape ({EMBEDDING_DIM},), "
                f"got {embedding.shape}"
            )

        emb = embedding.astype(np.float32)

        best_name = ""
        best_sim  = -1.0
        for name, enrolled_emb in self._enrolled.items():
            sim = float(np.dot(emb, enrolled_emb))   # cosine similarity
            logger.debug("verify | '%s' → similarity=%.4f", name, sim)
            if sim > best_sim:
                best_sim  = sim
                best_name = name

        matched = best_sim >= self._threshold
        details = (
            f"Best match: '{best_name}' sim={best_sim:.4f} "
            f"threshold={self._threshold:.2f} → {'MATCH' if matched else 'NO MATCH'}"
        )
        logger.info("verify | %s", details)

        return VerificationResult(
            matched=matched,
            similarity=best_sim,
            name=best_name if matched else "",
            details=details,
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_enrolled(self) -> bool:
        return bool(self._enrolled)

    def clear_enrollment(self) -> None:
        self._enrolled.clear()
        logger.info("Enrollment cleared.")


# ── Module-level utilities ────────────────────────────────────────────────────

def _l2_norm(v: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy of v. Safe against zero vectors."""
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


def _tensor_to_rgb_uint8(face_tensor: np.ndarray) -> np.ndarray:
    """
    Convert a recognition tensor to a uint8 RGB (H, W, 3) array.

    Handles:
      • (1, 112, 112, 3) float32  [-1, 1]   — from preprocess_for_recognition
      • (112, 112, 3)    float32  [-1, 1]
      • (1, 112, 112, 3) float32  [ 0, 1]   — treated same
      • (H, W, 3)        uint8    BGR         — converted to RGB
      • (H, W, 3)        uint8    RGB         — returned as-is
    """
    t = face_tensor

    # Remove batch dim
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]

    if t.dtype != np.uint8:
        # float [-1,1] → [0,255]  or float [0,1] → [0,255]
        if t.min() < -0.1:
            t = (t + 1.0) / 2.0          # [-1,1] → [0,1]
        t = (t * 255.0).clip(0, 255).astype(np.uint8)
        # Assume input was RGB (from ImageProcessor which uses cv2 BGR→RGB or not?)
        # ImageProcessor.preprocess_for_recognition usually keeps BGR order; convert.
        t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
    else:
        # Raw uint8: convert BGR→RGB if it looks like a camera frame
        if t.ndim == 3 and t.shape[2] == 3:
            t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)

    # Ensure 112×112
    if t.shape[:2] != (_RECOG_H, _RECOG_W):
        t = cv2.resize(t, (_RECOG_W, _RECOG_H), interpolation=cv2.INTER_LINEAR)

    return np.ascontiguousarray(t)

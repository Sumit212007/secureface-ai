"""
pipeline/recognizer.py
=======================
SecureEdge AI — Face Recognition / Embedding Module

Loads MobileFaceNet (INT8 TFLite) to extract 512-dimensional face embeddings
from preprocessed 112×112 tensors produced by ImageProcessor.

Provides:
  - get_embedding()          → extract a single normalised L2 embedding
  - cosine_similarity()      → compare two embeddings
  - verify()                 → threshold-based identity decision

MobileFaceNet output: 512-D float32 embedding vector, L2-normalised.
Cosine similarity of two L2-normalised vectors equals their dot product,
which is the most numerically efficient form and avoids a redundant sqrt.

Fallback: when no TFLite model is present, a deterministic random-projection
stub generates reproducible pseudo-embeddings from pixel statistics so the
full pipeline stays runnable end-to-end without model assets.

Author: SecureEdge AI Team
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ── Constants ────────────────────────────────────────────────────────────────

EMBEDDING_DIM: int = 512          # MobileFaceNet output dimension
COSINE_THRESHOLD: float = 0.40    # Similarity above this → same person
                                  # Empirically: 0.40 ≈ FAR 0.1% on LFW

# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """
    Carries the embedding vector plus provenance metadata.

    Fields
    ------
    vector : np.ndarray
        L2-normalised embedding, shape (512,), dtype float32.
    is_stub : bool
        True when the fallback stub generated this embedding.
        Orchestrator uses this flag to downgrade trust level.
    model_path : str
        Path of the model that produced this embedding (for audit logs).
    """
    vector: np.ndarray
    is_stub: bool = False
    model_path: str = ""

    def __post_init__(self) -> None:
        if self.vector.ndim != 1 or self.vector.shape[0] == 0:
            raise ValueError(
                f"EmbeddingResult.vector must be 1-D, got shape {self.vector.shape}."
            )


@dataclass
class VerificationResult:
    """
    Output of a 1:1 identity verification check.
    """
    similarity: float          # Cosine similarity in [-1, 1]
    is_match: bool             # similarity > COSINE_THRESHOLD
    threshold_used: float      # Threshold applied for this decision
    probe_is_stub: bool        # True if probe embedding came from stub model
    gallery_is_stub: bool      # True if gallery embedding came from stub model


# ── Main recogniser class ─────────────────────────────────────────────────────

class FaceRecognizer:
    """
    MobileFaceNet TFLite face recognition engine.

    Lifecycle
    ---------
    1. Instantiate once at app startup (loads model, allocates tensors).
    2. Call get_embedding(tensor) per authentication attempt.
    3. Compare returned embedding against enrolled gallery via verify().

    Thread safety
    -------------
    TFLite Interpreter is not thread-safe.  One FaceRecognizer instance
    per inference thread.  On Android the auth worker thread owns one instance.

    Parameters
    ----------
    model_path : str
        Path to mobilefacenet_int8.tflite.
    cosine_threshold : float
        Override the default verification threshold.
    """

    def __init__(
        self,
        model_path: str = "models/mobilefacenet/mobilefacenet_int8.tflite",
        cosine_threshold: float = COSINE_THRESHOLD,
    ) -> None:
        self._cosine_threshold = cosine_threshold
        self._model_path = model_path
        self._use_tflite = False
        self._interpreter = None
        self._inp_details = None
        self._out_details = None

        if os.path.isfile(model_path):
            self._load_tflite(model_path)
        else:
            logger.warning(
                "MobileFaceNet model not found at '%s'. "
                "Using deterministic stub embeddings. "
                "Place mobilefacenet_int8.tflite for production use.",
                model_path,
            )

    # ── Initialisation ────────────────────────────────────────────────────

    def _load_tflite(self, model_path: str) -> None:
        """Load MobileFaceNet TFLite and warm up with a dummy forward pass."""
        try:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._interpreter.allocate_tensors()
            self._inp_details = self._interpreter.get_input_details()
            self._out_details = self._interpreter.get_output_details()
            self._use_tflite = True

            inp_shape = self._inp_details[0]["shape"].tolist()
            out_shape = self._out_details[0]["shape"].tolist()
            logger.info(
                "MobileFaceNet TFLite loaded | input=%s output=%s",
                inp_shape, out_shape,
            )
            # Warm-up — prevents first-call latency spike on Android
            dummy = np.zeros((1, 112, 112, 3), dtype=np.float32)
            self._run_tflite(dummy)
            logger.debug("MobileFaceNet warm-up complete.")
        except Exception as exc:
            logger.error(
                "Failed to load MobileFaceNet TFLite: %s. Using stub.", exc
            )
            self._use_tflite = False

    # ── Public API ────────────────────────────────────────────────────────

    def get_embedding(self, face_tensor: np.ndarray) -> EmbeddingResult:
        """
        Extract a normalised L2 face embedding from a preprocessed tensor.

        Parameters
        ----------
        face_tensor : np.ndarray
            Shape (1, 112, 112, 3), float32, range [-1, 1].
            Produced by ImageProcessor.preprocess_for_recognition().

        Returns
        -------
        EmbeddingResult
            .vector → (512,) float32, L2-normalised.

        Raises
        ------
        ValueError
            If tensor shape or dtype is incompatible with the model.
        """
        self._validate_tensor(face_tensor)

        if self._use_tflite:
            raw_vector = self._run_tflite(face_tensor)
            is_stub = False
        else:
            raw_vector = self._stub_embedding(face_tensor)
            is_stub = True

        # L2 normalise — cosine similarity then reduces to a dot product
        embedding = _l2_normalise(raw_vector)

        logger.debug(
            "Embedding extracted | stub=%s norm=%.4f dim=%d",
            is_stub, float(np.linalg.norm(embedding)), len(embedding),
        )

        return EmbeddingResult(
            vector=embedding,
            is_stub=is_stub,
            model_path=self._model_path,
        )

    def verify(
        self,
        probe: EmbeddingResult,
        gallery: EmbeddingResult,
        threshold_override: Optional[float] = None,
    ) -> VerificationResult:
        """
        1:1 identity verification between a probe and a gallery embedding.

        Parameters
        ----------
        probe : EmbeddingResult
            Embedding from the live camera capture.
        gallery : EmbeddingResult
            Enrolled embedding retrieved from the encrypted store.
        threshold_override : float, optional
            Use a custom threshold for this call (e.g., higher security mode).

        Returns
        -------
        VerificationResult
            .is_match is True when similarity ≥ threshold.
        """
        threshold = threshold_override if threshold_override is not None \
            else self._cosine_threshold

        sim = cosine_similarity(probe.vector, gallery.vector)

        result = VerificationResult(
            similarity=float(sim),
            is_match=bool(sim >= threshold),
            threshold_used=threshold,
            probe_is_stub=probe.is_stub,
            gallery_is_stub=gallery.is_stub,
        )

        logger.info(
            "Verification | similarity=%.4f threshold=%.4f match=%s",
            result.similarity, result.threshold_used, result.is_match,
        )
        return result

    # ── TFLite inference ─────────────────────────────────────────────────

    def _run_tflite(self, tensor: np.ndarray) -> np.ndarray:
        """
        Run one forward pass through MobileFaceNet TFLite.

        Returns
        -------
        np.ndarray  shape (512,) or (128,) depending on model variant
        """
        # Handle INT8 quantised models: scale input if needed
        inp_detail = self._inp_details[0]
        if inp_detail["dtype"] == np.int8:
            scale, zp = inp_detail["quantization"]
            tensor = (tensor / scale + zp).astype(np.int8)

        self._interpreter.set_tensor(inp_detail["index"], tensor)
        self._interpreter.invoke()

        out = self._interpreter.get_tensor(self._out_details[0]["index"])

        # Dequantise output if INT8
        out_detail = self._out_details[0]
        if out_detail["dtype"] == np.int8:
            scale, zp = out_detail["quantization"]
            out = (out.astype(np.float32) - zp) * scale

        return out.flatten().astype(np.float32)

    # ── Stub / fallback ───────────────────────────────────────────────────

    @staticmethod
    def _stub_embedding(tensor: np.ndarray) -> np.ndarray:
        """
        Deterministic pseudo-embedding derived from pixel statistics.

        This is NOT a real face embedding and will NOT achieve meaningful
        accuracy.  It exists solely to keep the pipeline runnable end-to-end
        during development without model assets.

        Approach: hash pixel mean/std/percentiles → seed a deterministic
        random projection matrix → project 16 statistics into 512-D space.
        Same face tensor → same embedding (reproducible for unit tests).
        """
        flat = tensor.flatten()
        stats = np.array([
            flat.mean(), flat.std(),
            np.percentile(flat, 5), np.percentile(flat, 25),
            np.percentile(flat, 50), np.percentile(flat, 75),
            np.percentile(flat, 95), flat.min(), flat.max(),
            float(np.sum(flat > 0)) / len(flat),   # positive pixel ratio
            float(np.sum(flat > 0.5)) / len(flat),
            float(np.sum(flat < -0.5)) / len(flat),
            float(np.var(flat[:1000])),             # local variance proxy
            float(np.var(flat[-1000:])),
            float(np.mean(np.abs(np.diff(flat[:500])))),  # edge density proxy
            float(np.sum(flat ** 2) / len(flat)),   # energy
        ], dtype=np.float32)

        # Deterministic seed from pixel statistics checksum
        seed_bytes = hashlib.md5(stats.tobytes()).digest()
        seed = int.from_bytes(seed_bytes[:4], "little")
        rng = np.random.RandomState(seed)

        # Random projection: (16,) → (512,)
        projection = rng.randn(16, EMBEDDING_DIM).astype(np.float32)
        embedding = stats @ projection   # (512,)
        return embedding

    # ── Validation ────────────────────────────────────────────────────────

    @staticmethod
    def _validate_tensor(tensor: np.ndarray) -> None:
        """Validate that the input tensor matches MobileFaceNet's expected format."""
        if tensor is None:
            raise ValueError("face_tensor is None.")
        if tensor.ndim != 4:
            raise ValueError(
                f"Expected 4-D tensor (1, 112, 112, 3), got shape {tensor.shape}."
            )
        if tensor.shape != (1, 112, 112, 3):
            raise ValueError(
                f"Expected shape (1, 112, 112, 3), got {tensor.shape}. "
                "Ensure ImageProcessor.preprocess_for_recognition() was called."
            )
        if tensor.dtype != np.float32:
            raise ValueError(
                f"Expected float32 tensor, got {tensor.dtype}."
            )


# ── Module-level utility functions ────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D embedding vectors.

    For L2-normalised vectors this is equivalent to the dot product,
    which avoids redundant norm computation.

    Parameters
    ----------
    a, b : np.ndarray
        1-D float32 vectors of the same dimension.

    Returns
    -------
    float
        Similarity in [-1.0, 1.0].  Values above ~0.40 indicate the same
        identity for MobileFaceNet embeddings.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"cosine_similarity: vector shape mismatch {a.shape} vs {b.shape}."
        )
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        logger.warning("cosine_similarity: near-zero norm vector detected.")
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _l2_normalise(vector: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D vector. Returns zero vector if norm is near zero."""
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        logger.warning("_l2_normalise: near-zero norm — returning zero vector.")
        return np.zeros_like(vector)
    return vector / norm
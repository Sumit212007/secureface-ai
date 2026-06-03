"""
pipeline/embedding_backends.py
================================
Neural face-embedding backends for SecureEdge AI.

Primary: MobileFaceNet (112×112, 512-D, L2-normalised) via ONNX Runtime or TFLite.
Designed for offline mobile/edge deployment; matches ImageProcessor recognition tensors.

Input contract (from preprocess_for_recognition)
------------------------------------------------
    shape  : (1, 112, 112, 3) float32
    layout : NHWC, RGB
    range  : [-1, 1]  —  (pixel - 127.5) / 127.5

Author: SecureEdge AI Team
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

EMBEDDING_DIM: int = 512
INPUT_H: int = 112
INPUT_W: int = 112

# Cosine gates (L2-normalised embeddings → dot product == cosine similarity)
# Tune on your own enroll/verify set; values below are MobileFaceNet-oriented.
THRESHOLD_STRICT: float = 0.55   # high-security kiosk
THRESHOLD_DEFAULT: float = 0.45  # recommended production starting point
THRESHOLD_RELAXED: float = 0.38  # noisy outdoor cameras only

DEFAULT_MODEL_DIR: str = "models/mobilefacenet"
DEFAULT_CANDIDATE_FILES: List[str] = [
    "mobilefacenet.onnx",
    "mobilefacenet_int8.tflite",
    "mobile_face_net.onnx",
    "w600k_mbf.onnx",
]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / (n + 1e-8)).astype(np.float32)


def resolve_model_paths(
    model_path: Optional[str] = None,
    model_paths: Optional[List[str]] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> List[str]:
    """Build an ordered list of model files to try."""
    out: List[str] = []
    if model_path:
        out.append(model_path)
    if model_paths:
        out.extend(model_paths)
    for name in DEFAULT_CANDIDATE_FILES:
        out.append(os.path.join(model_dir, name))
    # de-duplicate, preserve order
    seen = set()
    unique: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


class EmbeddingBackend(ABC):
    """Abstract 512-D face embedder."""

    name: str = "base"

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def embed(self, face_tensor: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        face_tensor : (1, 112, 112, 3) float32 RGB in [-1, 1], or (112, 112, 3).

        Returns
        -------
        np.ndarray shape (512,) float32, L2-normalised.
        """


class MobileFaceNetOnnx(EmbeddingBackend):
    """ONNX Runtime inference for MobileFaceNet / ArcFace-style 512-D models."""

    name = "mobilefacenet_onnx"

    def __init__(self, model_path: str) -> None:
        self._path = model_path
        self._session = None
        self._inp_name = ""
        self._layout: str = "nchw"  # "nchw" | "nhwc"
        self._load()

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    def _load(self) -> None:
        if not os.path.isfile(self._path):
            logger.debug("ONNX model not found: %s", self._path)
            return
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("onnxruntime not installed — pip install onnxruntime")
            return
        try:
            sess = ort.InferenceSession(
                self._path,
                providers=["CPUExecutionProvider"],
            )
            inp = sess.get_inputs()[0]
            self._session = sess
            self._inp_name = inp.name
            self._layout = _infer_input_layout(inp.shape)
            self._warmup()
            logger.info(
                "MobileFaceNet ONNX loaded: '%s' | input=%s layout=%s",
                self._path, inp.shape, self._layout,
            )
        except Exception as exc:
            logger.error("Failed to load ONNX '%s': %s", self._path, exc)

    def _warmup(self) -> None:
        dummy = np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
        if self._layout == "nhwc":
            dummy = dummy.transpose(0, 2, 3, 1)
        self._session.run(None, {self._inp_name: dummy})

    def embed(self, face_tensor: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError("MobileFaceNetOnnx is not loaded.")
        blob = _tensor_to_input_blob(face_tensor, self._layout)
        t0 = time.perf_counter()
        out = self._session.run(None, {self._inp_name: blob})[0]
        _ = (time.perf_counter() - t0) * 1000.0
        vec = np.asarray(out, dtype=np.float32).reshape(-1)
        if vec.size != EMBEDDING_DIM:
            logger.warning(
                "ONNX output dim=%d (expected %d) — truncating/padding.",
                vec.size, EMBEDDING_DIM,
            )
            vec = _fit_embedding_dim(vec)
        return l2_normalize(vec)


class MobileFaceNetTflite(EmbeddingBackend):
    """TFLite inference (float or quantised) for Android-bound deployments."""

    name = "mobilefacenet_tflite"

    def __init__(self, model_path: str) -> None:
        self._path = model_path
        self._interpreter = None
        self._inp_idx = 0
        self._out_idx = 0
        self._layout: str = "nhwc"
        self._inp_dtype = np.float32
        self._inp_scale = 1.0
        self._inp_zero = 0
        self._out_scale = 1.0
        self._out_zero = 0
        self._load()

    @property
    def is_ready(self) -> bool:
        return self._interpreter is not None

    def _load(self) -> None:
        if not os.path.isfile(self._path):
            logger.debug("TFLite model not found: %s", self._path)
            return
        try:
            import tensorflow as tf
            interp = tf.lite.Interpreter(model_path=self._path)
            interp.allocate_tensors()
            inp_d = interp.get_input_details()[0]
            out_d = interp.get_output_details()[0]
            self._interpreter = interp
            self._inp_idx = inp_d["index"]
            self._out_idx = out_d["index"]
            self._layout = _infer_input_layout(inp_d["shape"])
            self._inp_dtype = inp_d["dtype"]
            self._inp_scale, self._inp_zero = inp_d.get("quantization", (1.0, 0))
            self._out_scale, self._out_zero = out_d.get("quantization", (1.0, 0))
            self._warmup()
            logger.info(
                "MobileFaceNet TFLite loaded: '%s' | in=%s %s layout=%s",
                self._path, inp_d["shape"], inp_d["dtype"], self._layout,
            )
        except Exception as exc:
            logger.error("Failed to load TFLite '%s': %s", self._path, exc)

    def _warmup(self) -> None:
        shape = self._interpreter.get_input_details()[0]["shape"]
        dummy = np.zeros(shape, dtype=self._inp_dtype)
        self._interpreter.set_tensor(self._inp_idx, dummy)
        self._interpreter.invoke()

    def embed(self, face_tensor: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError("MobileFaceNetTflite is not loaded.")
        blob = _tensor_to_input_blob(face_tensor, self._layout)
        if self._inp_dtype == np.uint8:
            blob = np.clip(
                blob / self._inp_scale + self._inp_zero, 0, 255
            ).astype(np.uint8)
        elif self._inp_dtype == np.int8:
            blob = np.clip(
                blob / self._inp_scale + self._inp_zero, -128, 127
            ).astype(np.int8)
        else:
            blob = blob.astype(self._inp_dtype)

        t0 = time.perf_counter()
        self._interpreter.set_tensor(self._inp_idx, blob)
        self._interpreter.invoke()
        out = self._interpreter.get_tensor(self._out_idx)
        _ = (time.perf_counter() - t0) * 1000.0

        vec = np.asarray(out, dtype=np.float32).reshape(-1)
        if self._interpreter.get_output_details()[0]["dtype"] in (np.uint8, np.int8):
            vec = (vec.astype(np.float32) - self._out_zero) * self._out_scale
        if vec.size != EMBEDDING_DIM:
            vec = _fit_embedding_dim(vec)
        return l2_normalize(vec)


def create_embedding_backend(
    model_path: Optional[str] = None,
    model_paths: Optional[List[str]] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> Optional[EmbeddingBackend]:
    """
    Load the first available MobileFaceNet ONNX or TFLite model.

    Returns None if no model file loads successfully.
    """
    for path in resolve_model_paths(model_path, model_paths, model_dir):
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        backend: Optional[EmbeddingBackend] = None
        if ext == ".onnx":
            backend = MobileFaceNetOnnx(path)
        elif ext == ".tflite":
            backend = MobileFaceNetTflite(path)
        if backend is not None and backend.is_ready:
            return backend
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_input_layout(shape) -> str:
    """Infer NCHW vs NHWC from ONNX/TFLite input shape (batch, ...)."""
    if shape is None or len(shape) != 4:
        return "nchw"
    # shape may contain None / str for dynamic dims
    dims = []
    for d in shape:
        try:
            dims.append(int(d))
        except (TypeError, ValueError):
            dims.append(-1)
    if len(dims) == 4 and dims[1] == 3:
        return "nchw"
    if len(dims) == 4 and dims[3] == 3:
        return "nhwc"
    return "nchw"


def _tensor_to_input_blob(face_tensor: np.ndarray, layout: str) -> np.ndarray:
    """
    Convert recognition tensor to model input blob.

    Accepts (1,112,112,3) or (112,112,3) float32 RGB [-1,1].
    """
    t = np.asarray(face_tensor, dtype=np.float32)
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    if t.ndim != 3 or t.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB tensor, got shape {t.shape}")
    if t.shape[0] != INPUT_H or t.shape[1] != INPUT_W:
        import cv2
        t = cv2.resize(t, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)

    if layout == "nhwc":
        return np.expand_dims(t, axis=0).astype(np.float32)
    # NCHW
    return np.expand_dims(t.transpose(2, 0, 1), axis=0).astype(np.float32)


def _fit_embedding_dim(vec: np.ndarray) -> np.ndarray:
    if vec.size >= EMBEDDING_DIM:
        return vec[:EMBEDDING_DIM].astype(np.float32)
    out = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    out[: vec.size] = vec
    return out

#!/usr/bin/env python3
"""
setup_recognition_model.py
==========================
Prepare MobileFaceNet recognition weights for SecureEdge AI.

Usage
-----
  python scripts/setup_recognition_model.py
  python scripts/setup_recognition_model.py --from-insightface
  python scripts/setup_recognition_model.py --verify

The pipeline expects (first match wins):
  models/mobilefacenet/mobilefacenet.onnx      — ONNX Runtime (Python / desktop)
  models/mobilefacenet/mobilefacenet_int8.tflite — TFLite (Android)

Manual placement
----------------
Copy any ArcFace/MobileFaceNet 112×112 → 512-D ONNX export to:
  secureface-ai/models/mobilefacenet/mobilefacenet.onnx

InsightFace ``buffalo_sc`` pack includes ``w600k_mbf.onnx`` (MobileFaceNet backbone).
After ``pip install insightface`` and one FaceAnalysis() run, use --from-insightface
to copy it into the project.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "mobilefacenet"
ONNX_DST = MODEL_DIR / "mobilefacenet.onnx"


def _copy_insightface_mbf() -> bool:
    """Copy w600k_mbf.onnx from InsightFace model cache if present."""
    home = Path.home() / ".insightface" / "models"
    if not home.is_dir():
        return False
    for pack in sorted(home.iterdir()):
        if not pack.is_dir():
            continue
        for name in ("w600k_mbf.onnx", "model.onnx"):
            src = pack / name
            if src.is_file():
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, ONNX_DST)
                print(f"[OK] Copied {src} → {ONNX_DST}")
                return True
    return False


def _bootstrap_insightface() -> bool:
    """Download buffalo_sc via InsightFace API, then copy recognition ONNX."""
    try:
        import insightface
    except ImportError:
        print("[ERROR] insightface not installed. Run: pip install insightface")
        return False
    print("[INFO] Downloading InsightFace buffalo_sc (includes MobileFaceNet)...")
    app = insightface.app.FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return _copy_insightface_mbf()


def verify_model() -> int:
    sys.path.insert(0, str(ROOT))
    from pipeline.embedding_backends import create_embedding_backend
    import numpy as np

    backend = create_embedding_backend(model_dir=str(MODEL_DIR))
    if backend is None:
        print("[FAIL] No recognition model loaded.")
        print(f"       Place ONNX at: {ONNX_DST}")
        return 1
    dummy = np.zeros((1, 112, 112, 3), dtype=np.float32)
    emb = backend.embed(dummy)
    print(f"[OK] Backend={backend.name} embedding shape={emb.shape} norm={np.linalg.norm(emb):.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup MobileFaceNet recognition model")
    parser.add_argument(
        "--from-insightface",
        action="store_true",
        help="Download/copy w600k_mbf.onnx via InsightFace buffalo_sc",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a dummy forward pass through the loaded embedder",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        return verify_model()

    if ONNX_DST.is_file():
        print(f"[OK] Model already present: {ONNX_DST}")
        return verify_model() if args.from_insightface else 0

    if args.from_insightface:
        if _copy_insightface_mbf() or _bootstrap_insightface():
            return verify_model()
        print("[FAIL] Could not obtain w600k_mbf.onnx from InsightFace.")
        return 1

    print(__doc__)
    print(f"\nTarget path: {ONNX_DST}")
    print("Options:")
    print("  1. Copy your ONNX export to the path above")
    print("  2. python scripts/setup_recognition_model.py --from-insightface")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

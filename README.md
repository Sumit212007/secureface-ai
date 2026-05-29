# SecureEdge AI — Offline Face Recognition & Liveness Detection

Offline mobile-first AI pipeline for facial recognition and liveness detection.  
Target: Android 8+, Snapdragon 660+, <800 ms total auth, <20 MB models, fully offline.

---

## Folder Structure

```
secureedge_ai/
├── pipeline/
│   ├── __init__.py
│   ├── image_processor.py      # CLAHE, alignment, normalisation
│   ├── face_detector.py        # BlazeFace TFLite + OpenCV fallback
│   ├── recognizer.py           # MobileFaceNet + cosine similarity
│   ├── liveness_detector.py    # Anti-spoof CNN + EAR blink detector
│   └── orchestrator.py         # Full pipeline controller
├── models/
│   ├── blazeface/              # face_detection_short_range.tflite
│   ├── mobilefacenet/          # mobilefacenet_int8.tflite
│   └── antispoofing/           # antispoofing_int8.tflite
├── test_pipeline.py            # CLI test harness
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### 2. (Optional) Download model files

Place these files in the correct `models/` subdirectories:

| File | Source | Size |
|------|--------|------|
| `models/blazeface/face_detection_short_range.tflite` | [MediaPipe GitHub](https://github.com/google/mediapipe/tree/master/mediapipe/models) | ~0.8 MB |
| `models/mobilefacenet/mobilefacenet_int8.tflite` | [Repo / training output] | ~4.9 MB |
| `models/antispoofing/antispoofing_int8.tflite` | [Repo / training output] | ~3.5 MB |

**Without model files:** The pipeline still runs using built-in fallbacks (OpenCV Haar cascade + LBP heuristic + stub embeddings). Accuracy is reduced but all code paths execute correctly.

### 3. Run tests

```bash
# Smoke test — no images or models needed
python test_pipeline.py --smoke

# Test on a static image (enroll + verify same image)
python test_pipeline.py --image path/to/face.jpg --show

# Separate enroll / verify images
python test_pipeline.py --enroll enroll.jpg --verify verify.jpg

# Live webcam
python test_pipeline.py --webcam --show

# Disable blink challenge (faster, useful for still images)
python test_pipeline.py --image face.jpg --no-blink

# Verbose logging
python test_pipeline.py --smoke --verbose
```

---

## Pipeline Flow

```
BGR Frame
    │
    ▼
FaceDetector.detect_faces()         → FaceDetection (bbox, landmarks, confidence)
    │
    ▼
ImageProcessor.crop_face_roi()      → face ROI (padded)
    │
    ├──► preprocess_for_recognition() → (1,112,112,3) float32 [-1,1]
    │
    └──► preprocess_for_liveness()    → (1,224,224,3) float32 [0,1]
                │                              │
                ▼                              ▼
    FaceRecognizer.get_embedding()    LivenessDetector.predict_liveness()
         → (512,) L2-normalised            → CNN score + blink check
                │                              │
                └──────────┬────────────────────┘
                           ▼
                  AuthPipeline decision
                  ALLOW / DENY / PENDING
```

---

## Decision Logic

| Condition | Decision |
|-----------|----------|
| No face detected | DENY |
| CNN predicts SPOOF | DENY |
| CNN LIVE + blink timeout | DENY |
| CNN LIVE + blink confirmed + similarity ≥ threshold | **ALLOW** |
| CNN LIVE + awaiting blink | PENDING (send next frame) |
| Similarity < threshold | DENY (unknown person) |

---

## Configuration

```python
from pipeline.orchestrator import AuthPipeline

pipeline = AuthPipeline(
    detector_model="models/blazeface/face_detection_short_range.tflite",
    recognizer_model="models/mobilefacenet/mobilefacenet_int8.tflite",
    antispoofing_model="models/antispoofing/antispoofing_int8.tflite",
    cosine_threshold=0.40,   # Higher = stricter identity match
    require_blink=True,      # False = passive CNN only (faster)
)
```

---

## Android Deployment Notes

- Use `tflite-runtime` (not full TensorFlow) for the APK
- Each `ImageProcessor`, `FaceDetector`, `FaceRecognizer` instance is per-thread
- Call `allocate_tensors()` once at startup — never per-frame
- Target NNAPI delegate for Snapdragon 660 hardware acceleration
- Minimum heap: 512 MB (models + frame buffer)
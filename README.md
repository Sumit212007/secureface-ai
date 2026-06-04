# 🔐 SecureFace AI — Offline Face Recognition & Liveness Detection

> **Government Hackathon Project** — A fully offline, mobile-first AI system for secure facial authentication, combining a Python AI pipeline with a React Native (Expo) mobile frontend.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TFLite](https://img.shields.io/badge/TFLite-2.13%2B-orange?logo=tensorflow)](https://www.tensorflow.org/lite)
[![Expo](https://img.shields.io/badge/Expo-SDK%2051%2B-black?logo=expo)](https://expo.dev/)
[![React Native](https://img.shields.io/badge/React%20Native-0.74%2B-61DAFB?logo=react)](https://reactnative.dev/)
[![Platform](https://img.shields.io/badge/Platform-Android%208%2B-green?logo=android)](https://www.android.com/)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

---

## 📌 Overview

SecureFace AI is a two-part system:

| Part | Folder | Description |
|---|---|---|
| 🧠 **AI Pipeline** | `secureface-ai/` | Python backend: face detection, recognition, liveness |
| 📱 **Mobile App** | `secure-verify/` | Expo (React Native) frontend for Android & iOS |

All AI inference runs **fully on-device** — no internet connection, no cloud calls, no biometric data ever leaves the device.

**Key targets:**
- ⚡ **< 800 ms** total authentication latency
- 📦 **< 20 MB** total model footprint
- 📱 **Android 8+**, Snapdragon 660+
- 🔒 **100% offline** operation

---

## 🗂️ Repository Structure

```
secureface-ai/                  ← Root repo
├── secureface-ai/              ← Python AI pipeline
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── image_processor.py      # CLAHE, alignment, normalisation
│   │   ├── face_detector.py        # BlazeFace TFLite + OpenCV fallback
│   │   ├── recognizer.py           # MobileFaceNet + cosine similarity
│   │   ├── liveness_detector.py    # Anti-spoof CNN + EAR blink detector
│   │   └── orchestrator.py         # Full pipeline controller
│   ├── models/
│   │   ├── blazeface/              # face_detection_short_range.tflite
│   │   ├── mobilefacenet/          # mobilefacenet_int8.tflite  ← download required
│   │   └── antispoofing/           # antispoofing_int8.tflite
│   ├── test_processor.py
│   └── requirements.txt
│
├── secure-verify/              ← Expo React Native frontend
│   ├── app/                        # File-based routing (Expo Router)
│   ├── assets/
│   ├── components/
│   ├── package.json
│   └── README.md
│
└── README.md                   ← You are here
```

---

## 🧠 AI Pipeline (`secureface-ai/`)

### Models Used

| Model | Purpose | Size | Format |
|---|---|---|---|
| **BlazeFace** | Face detection | ~0.8 MB | TFLite |
| **MobileFaceNet** | Face recognition (512-d embeddings) | ~4.9 MB | TFLite |
| **Anti-Spoofing CNN** | Liveness detection | ~3.5 MB | TFLite |

### ⬇️ Download MobileFaceNet Model

> **Required** — you must download this manually and place it in the correct directory.

📥 **[Download `mobilefacenet_int8.tflite` from Google Drive](https://drive.google.com/file/d/16ef2AUanz2ta-x_vWPkYqbDN1A_7_2Ep/view?usp=sharing)**

After downloading, place it at:
```
secureface-ai/models/mobilefacenet/mobilefacenet_int8.tflite
```

For the other models:

| File | Source |
|---|---|
| `models/blazeface/face_detection_short_range.tflite` | [MediaPipe GitHub](https://github.com/google/mediapipe/tree/master/mediapipe/models) |
| `models/antispoofing/antispoofing_int8.tflite` | Included in repo / training output |

> **Without model files:** The pipeline still runs using built-in fallbacks (OpenCV Haar cascade + LBP heuristic + stub embeddings). All code paths execute correctly but accuracy is reduced.

### Setup & Run

```bash
cd secureface-ai

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Smoke test — no images or models needed
python test_processor.py --smoke

# Test on a static image
python test_processor.py --image path/to/face.jpg --show

# Separate enroll / verify images
python test_processor.py --enroll enroll.jpg --verify verify.jpg

# Live webcam
python test_processor.py --webcam --show

# Disable blink challenge (faster, useful for still images)
python test_processor.py --image face.jpg --no-blink

# Verbose logging
python test_processor.py --smoke --verbose
```

### Pipeline Flow

```
BGR Frame
    │
    ▼
FaceDetector.detect_faces()         → FaceDetection (bbox, landmarks, confidence)
    │
    ▼
ImageProcessor.crop_face_roi()      → face ROI (padded)
    │
    ├──► preprocess_for_recognition() → (1, 112, 112, 3) float32 [-1, 1]
    │
    └──► preprocess_for_liveness()    → (1, 224, 224, 3) float32 [0, 1]
                │                                │
                ▼                                ▼
    FaceRecognizer.get_embedding()    LivenessDetector.predict_liveness()
         → (512,) L2-normalised            → CNN score + blink check
                │                                │
                └──────────┬─────────────────────┘
                           ▼
                  AuthPipeline Decision
                  ALLOW / DENY / PENDING
```

### Decision Logic

| Condition | Decision |
|---|---|
| No face detected | ❌ DENY |
| CNN predicts SPOOF | ❌ DENY |
| CNN LIVE + blink timeout | ❌ DENY |
| Similarity < threshold | ❌ DENY (unknown person) |
| CNN LIVE + awaiting blink | ⏳ PENDING (send next frame) |
| CNN LIVE + blink confirmed + similarity ≥ threshold | ✅ **ALLOW** |

### Configuration

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

### Dependencies

| Package | Purpose |
|---|---|
| `numpy >= 1.24.0` | Array ops (TFLite compatible, `< 2.0`) |
| `opencv-python >= 4.8.0` | Camera, image I/O, CLAHE, cascade fallback |
| `tensorflow >= 2.13.0` | Full TF for training + inference |
| `tflite-runtime >= 2.13.0` | Inference-only runtime (recommended for production) |
| `mediapipe >= 0.10.3` | Face Mesh for EAR blink detection |
| `scipy >= 1.11.0` | KD-tree for large gallery search |
| `onnx`, `tf2onnx`, `onnxruntime` | Model conversion & ONNX validation |
| `matplotlib`, `Pillow` | Evaluation visualisations |
| `pytest`, `pytest-cov` | Testing |

> **Android / CI:** Use `opencv-python-headless` instead of `opencv-python` (no GUI dependency).

---

## 📱 Mobile App (`secure-verify/`)

The `secure-verify` folder is an **Expo (React Native)** app that provides the mobile UI for the SecureFace AI authentication system. It communicates with the AI pipeline for live face verification.

### Setup & Run

```bash
cd secure-verify

# Install dependencies
npm install

# Start the development server
npx expo start
```

From the Expo CLI output you can open the app in:
- 📱 **Expo Go** — scan QR code on your phone (quickest way to try)
- 🤖 **Android Emulator** — via Android Studio
- 🍎 **iOS Simulator** — via Xcode (macOS only)
- 🔧 **Development Build** — for full native module support

### Development

Edit files inside the `app/` directory. The project uses **file-based routing** via [Expo Router](https://docs.expo.dev/router/introduction/).

```bash
# Reset to a blank project (moves starter code to app-example/)
npm run reset-project

# Lint
npx expo lint

# Run tests
npx jest
```

### Optional Setup

| Task | Command / Guide |
|---|---|
| ESLint + Prettier | `npx expo lint` or [guide](https://docs.expo.dev/guides/using-eslint/) |
| Unit testing (Jest) | [Unit Testing with Jest](https://docs.expo.dev/develop/unit-testing/) |
| TypeScript config | [Using TypeScript](https://docs.expo.dev/guides/typescript/) |

---

## 📱 Android Deployment Notes

- Use `tflite-runtime` (not full TensorFlow) to keep APK size minimal (~30 MB vs ~500 MB)
- Each `ImageProcessor`, `FaceDetector`, `FaceRecognizer` should be **per-thread**
- Call `allocate_tensors()` **once at startup** — never per-frame
- Target **NNAPI delegate** for Snapdragon 660 hardware acceleration
- Minimum heap: **512 MB** (models + frame buffer)

---

## 🛡️ Security Notes

- All biometric data is processed and stored **on-device only**
- Face embeddings can be stored in an **SQLCipher-encrypted** database
- No raw face images are ever stored — only 512-dimensional vectors
- Liveness detection prevents replay attacks using photos or videos

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📚 Resources

**Expo / React Native**
- [Expo Documentation](https://docs.expo.dev/)
- [Learn Expo Tutorial](https://docs.expo.dev/tutorial/introduction/)
- [Expo on GitHub](https://github.com/expo/expo)
- [Expo Discord Community](https://chat.expo.dev)

**AI / TFLite**
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [MediaPipe Face Detection](https://developers.google.com/mediapipe/solutions/vision/face_detector)
- [MobileFaceNet Paper](https://arxiv.org/abs/1804.07573)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Built with ❤️ for a Government Hackathon — <strong>SecureFace AI</strong>
</div>

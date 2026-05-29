"""
pipeline/image_processor.py
============================
SecureEdge AI — Image Preprocessing Module

Responsible for all image normalization, enhancement, alignment, and format
preparation before any model inference. Acts as the single entry point for
every raw frame entering the pipeline.

Two public preprocessing paths:
  - preprocess_for_recognition()  → MobileFaceNet  (112×112, [-1, 1])
  - preprocess_for_liveness()     → Anti-Spoofing CNN  (224×224, [0, 1])

Both paths share a common enhancement stack (CLAHE → Gaussian denoise → BGR→RGB)
and diverge at resize + normalization.

Author: SecureEdge AI Team
Target: Android 8+, Snapdragon 660+, TFLite runtime
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger — integrates with any logging config the caller sets up
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Silent by default; caller configures


# ---------------------------------------------------------------------------
# Constants — centralised so Android-side JNI wrapper can mirror them
# ---------------------------------------------------------------------------
MOBILEFACENET_INPUT_SIZE: int = 112   # MobileFaceNet expects (112, 112, 3)
ANTISPOOFING_INPUT_SIZE: int = 224    # MobileNetV2 anti-spoof CNN: (224, 224, 3)

# CLAHE hyperparams — tuned for typical Indian field lighting (high-contrast sun,
# dark interior shadow). clipLimit=2.0 avoids over-amplifying noise in uniform areas.
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)

# Gaussian kernel — 3×3 is enough for camera sensor noise without blurring edges
GAUSSIAN_KERNEL: Tuple[int, int] = (3, 3)

# MobileFaceNet normalization: (pixel - 127.5) / 127.5 → maps [0, 255] → [-1, 1]
MOBILEFACENET_MEAN: float = 127.5
MOBILEFACENET_STD: float = 127.5

# Canonical eye positions inside the 112×112 output frame (ArcFace standard)
# These anchors keep inter-eye spacing consistent across all faces.
_LEFT_EYE_CANONICAL: Tuple[float, float] = (30.2946, 51.6963)
_RIGHT_EYE_CANONICAL: Tuple[float, float] = (65.5318, 51.5014)


# ---------------------------------------------------------------------------
# Data classes — clean, self-documenting inter-module contracts
# ---------------------------------------------------------------------------

@dataclass
class FaceLandmarks:
    """
    Minimal landmark set required for face alignment.

    Coordinates are in *pixel space* of the **cropped face ROI** (not the
    full frame).  The detector module is responsible for projecting
    full-frame landmarks into ROI-relative coordinates before passing them
    to ImageProcessor.

    All fields are (x, y) tuples of floats.
    """
    left_eye: Tuple[float, float]    # Centre of left eye
    right_eye: Tuple[float, float]   # Centre of right eye
    nose_tip: Optional[Tuple[float, float]] = None  # Optional; unused here
    mouth_left: Optional[Tuple[float, float]] = None
    mouth_right: Optional[Tuple[float, float]] = None


@dataclass
class PreprocessResult:
    """
    Carries a preprocessed tensor alongside diagnostic metadata.
    The pipeline orchestrator only needs `tensor`; metadata aids debugging.
    """
    tensor: np.ndarray          # Shape: (1, H, W, 3), dtype float32
    original_shape: Tuple[int, int]   # (height, width) before resize
    was_aligned: bool           # True if landmark-based alignment was applied
    clahe_applied: bool         # True if CLAHE was run (False on near-black frames)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ImageProcessor:
    """
    Stateless (except CLAHE object) image preprocessing engine.

    All heavy OpenCV objects (CLAHE) are created once at init to avoid
    per-frame allocation overhead — critical on Android where GC pauses
    can push total auth latency over the 800 ms budget.

    Usage
    -----
    >>> processor = ImageProcessor()
    >>> tensor = processor.preprocess_for_recognition(face_roi, landmarks)
    >>> liveness_tensor = processor.preprocess_for_liveness(face_roi)

    Thread safety
    -------------
    Not thread-safe. Each inference thread should hold its own instance.
    On Android, the main camera callback thread owns one instance.
    """

    def __init__(self) -> None:
        """
        Initialise reusable OpenCV objects.

        CLAHE is created once here. cv2.createCLAHE() allocates internal
        tiling buffers; re-creating it per frame wastes ~0.3 ms and triggers
        extra allocations that Android's dalvik GC must later collect.
        """
        self._clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID,
        )

        # Precompute the 2×3 affine destination matrix for canonical alignment.
        # This avoids redundant trig per frame when landmarks are available.
        self._canonical_dst = np.array(
            [_LEFT_EYE_CANONICAL, _RIGHT_EYE_CANONICAL],
            dtype=np.float32,
        )

        logger.info(
            "ImageProcessor initialised | CLAHE clip=%.1f grid=%s | "
            "Recognition input=%dpx | Liveness input=%dpx",
            CLAHE_CLIP_LIMIT,
            CLAHE_TILE_GRID,
            MOBILEFACENET_INPUT_SIZE,
            ANTISPOOFING_INPUT_SIZE,
        )

    # ------------------------------------------------------------------
    # Public API — Recognition path
    # ------------------------------------------------------------------

    def preprocess_for_recognition(
        self,
        face_roi: np.ndarray,
        landmarks: Optional[FaceLandmarks] = None,
    ) -> PreprocessResult:
        """
        Full preprocessing pipeline for MobileFaceNet recognition input.

        Pipeline
        --------
        BGR → RGB → CLAHE (LAB L-channel) → Gaussian denoise →
        [optional face alignment] → resize (112×112) →
        normalise [-1, 1] → add batch dim

        Parameters
        ----------
        face_roi : np.ndarray
            Cropped face region in **BGR** format (OpenCV default).
            Shape: (H, W, 3), dtype uint8.
            Minimum recommended size: 64×64 px.  Below this, CLAHE tile
            artefacts appear and alignment geometry degrades.
        landmarks : FaceLandmarks, optional
            Eye centre coordinates in ROI-pixel space.
            When provided, applies affine alignment to canonical positions.
            When None, skips alignment (slightly lower recognition accuracy
            but safe for fallback or re-enrollment flows).

        Returns
        -------
        PreprocessResult
            .tensor → shape (1, 112, 112, 3), float32, range [-1, 1]
                      Ready for tf.lite.Interpreter.set_tensor()

        Raises
        ------
        ValueError
            If face_roi is None, not 3-channel, or smaller than 16×16.
        cv2.error
            Propagated from OpenCV on corrupt image data.
        """
        self._validate_roi(face_roi, min_size=16)
        original_shape = (face_roi.shape[0], face_roi.shape[1])

        # ── Step 1: BGR → RGB ────────────────────────────────────────────
        img_rgb = self._bgr_to_rgb(face_roi)

        # ── Step 2: CLAHE (adaptive contrast) ───────────────────────────
        img_enhanced, clahe_applied = self._apply_clahe(img_rgb)

        # ── Step 3: Gaussian denoising ───────────────────────────────────
        img_denoised = self._gaussian_denoise(img_enhanced)

        # ── Step 4: Face alignment (optional) ────────────────────────────
        was_aligned = False
        if landmarks is not None:
            img_denoised, was_aligned = self._align_face_affine(
                img_denoised, landmarks
            )

        # ── Step 5: Resize to model input size ───────────────────────────
        img_resized = cv2.resize(
            img_denoised,
            (MOBILEFACENET_INPUT_SIZE, MOBILEFACENET_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,  # Bilinear — good balance speed/quality
        )

        # ── Step 6: Normalise to [-1, 1] (MobileFaceNet / ArcFace standard) ─
        img_float = self._normalise_mobilefacenet(img_resized)

        # ── Step 7: Add batch dimension ──────────────────────────────────
        tensor = np.expand_dims(img_float, axis=0)  # (1, 112, 112, 3)

        logger.debug(
            "Recognition preprocess | orig=%s aligned=%s clahe=%s "
            "output=%s dtype=%s range=[%.2f, %.2f]",
            original_shape, was_aligned, clahe_applied,
            tensor.shape, tensor.dtype, tensor.min(), tensor.max(),
        )

        return PreprocessResult(
            tensor=tensor,
            original_shape=original_shape,
            was_aligned=was_aligned,
            clahe_applied=clahe_applied,
        )

    # ------------------------------------------------------------------
    # Public API — Liveness / Anti-Spoofing path
    # ------------------------------------------------------------------

    def preprocess_for_liveness(
        self,
        face_roi: np.ndarray,
    ) -> PreprocessResult:
        """
        Lighter preprocessing path for the anti-spoofing CNN (224×224).

        Intentionally preserves more texture detail than the recognition
        path — the anti-spoofing model relies on Moiré patterns, screen
        reflections, and micro-texture differences between live skin and
        printed/screen spoofs.  Heavy smoothing would destroy these cues.

        Differences from recognition path
        ----------------------------------
        * Gaussian blur kernel is skipped to preserve spoof texture cues.
        * No face alignment (spoofing CNN does not require precise geometry).
        * Output size: 224×224 (MobileNetV2 standard input).
        * Normalisation: [0, 1] via /255.0 (not [-1,1]).

        Parameters
        ----------
        face_roi : np.ndarray
            Cropped face region in BGR format, uint8, min size 32×32.

        Returns
        -------
        PreprocessResult
            .tensor → shape (1, 224, 224, 3), float32, range [0, 1]
        """
        self._validate_roi(face_roi, min_size=32)
        original_shape = (face_roi.shape[0], face_roi.shape[1])

        # ── BGR → RGB ────────────────────────────────────────────────────
        img_rgb = self._bgr_to_rgb(face_roi)

        # ── CLAHE: apply but preserve texture (lower clip limit) ─────────
        # We use a lower clip limit for liveness to avoid enhancing print
        # artefacts that would fool the CNN into thinking it is real skin.
        img_enhanced, clahe_applied = self._apply_clahe(
            img_rgb, clip_override=1.5
        )

        # NOTE: No Gaussian blur here — texture is a primary spoof signal.

        # ── Resize to anti-spoofing CNN input size ───────────────────────
        img_resized = cv2.resize(
            img_enhanced,
            (ANTISPOOFING_INPUT_SIZE, ANTISPOOFING_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        # ── Normalise [0, 1] ─────────────────────────────────────────────
        img_float = img_resized.astype(np.float32) / 255.0

        # ── Batch dimension ──────────────────────────────────────────────
        tensor = np.expand_dims(img_float, axis=0)  # (1, 224, 224, 3)

        logger.debug(
            "Liveness preprocess | orig=%s clahe=%s "
            "output=%s dtype=%s range=[%.3f, %.3f]",
            original_shape, clahe_applied,
            tensor.shape, tensor.dtype, tensor.min(), tensor.max(),
        )

        return PreprocessResult(
            tensor=tensor,
            original_shape=original_shape,
            was_aligned=False,
            clahe_applied=clahe_applied,
        )

    # ------------------------------------------------------------------
    # Public utility — crop face ROI from full frame
    # ------------------------------------------------------------------

    def crop_face_roi(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding_factor: float = 0.20,
    ) -> np.ndarray:
        """
        Crop and pad a face bounding box from a full camera frame.

        Adding padding ensures hair, ears, and forehead are included —
        these regions carry identity and liveness cues. 20% is the
        empirically-tuned value for MobileFaceNet accuracy.

        Parameters
        ----------
        frame : np.ndarray
            Full camera frame, BGR, uint8.
        bbox : tuple of (x1, y1, x2, y2)
            Bounding box from BlazeFace detector, in pixel coordinates.
        padding_factor : float
            Fractional expansion applied symmetrically on all sides.
            0.20 = 20% of bounding box width/height added on each side.

        Returns
        -------
        np.ndarray
            Padded face crop, BGR uint8.  May be smaller than expected if
            the face is near the frame edge (boundary-safe clamping applied).
        """
        if frame is None or frame.ndim != 3:
            raise ValueError("crop_face_roi: frame must be a 3-channel BGR array.")

        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame.shape[:2]

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            raise ValueError(
                f"crop_face_roi: degenerate bbox {bbox}. "
                "Ensure detector output is valid."
            )

        pad_w = int(bw * padding_factor)
        pad_h = int(bh * padding_factor)

        # Clamp to frame boundaries — faces near edges must not wrap around
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(frame_w, x2 + pad_w)
        cy2 = min(frame_h, y2 + pad_h)

        roi = frame[cy1:cy2, cx1:cx2]

        if roi.size == 0:
            raise ValueError(
                f"crop_face_roi: empty crop from bbox {bbox} on "
                f"frame {frame_w}×{frame_h}. Check detector output."
            )

        return roi

    # ------------------------------------------------------------------
    # Private helpers — image enhancement
    # ------------------------------------------------------------------

    @staticmethod
    def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
        """
        Convert BGR (OpenCV default) to RGB in-place-safe fashion.

        Using cvtColor rather than img[:,:,::-1] because cvtColor is
        NEON-accelerated on ARM and produces a properly contiguous array
        that avoids NumPy stride issues downstream.
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _apply_clahe(
        self,
        img_rgb: np.ndarray,
        clip_override: Optional[float] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)
        on the L* channel of CIE LAB colour space.

        Why LAB and not HSV?
        --------------------
        LAB separates luminance (L*) from chrominance (a*, b*). Applying
        CLAHE only to L* enhances perceptual contrast without shifting hue
        or saturation — critical for consistent face colour that the
        recognition model uses as a weak identity cue.

        Why CLAHE over global HE?
        -------------------------
        Global histogram equalisation destroys local contrast in mixed-
        lighting frames (e.g. one side lit by sun, other in shadow). CLAHE's
        tiling preserves regional detail while preventing over-amplification
        via the clip limit.

        Skip condition
        --------------
        If the frame mean luminance < 5 (near-black, camera shuttered or
        lens covered), CLAHE would just amplify sensor noise. Skip it and
        return the original with clahe_applied=False.

        Parameters
        ----------
        img_rgb : np.ndarray  RGB uint8
        clip_override : float, optional
            Override the default CLAHE clip limit (used by liveness path).

        Returns
        -------
        tuple[np.ndarray, bool]
            (enhanced_rgb_uint8, was_clahe_applied)
        """
        # Convert to LAB (expects RGB input → produces LAB)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

        l_channel = lab[:, :, 0]
        mean_luminance = float(l_channel.mean())

        # Skip on near-black frames — CLAHE adds noise, not signal
        if mean_luminance < 5.0:
            logger.debug(
                "_apply_clahe: skipped (mean_luminance=%.1f < 5.0)", mean_luminance
            )
            return img_rgb, False

        if clip_override is not None:
            # Temporarily swap clip limit — thread-safe only because each
            # ImageProcessor instance is per-thread (see class docstring).
            original_clip = self._clahe.getClipLimit()
            self._clahe.setClipLimit(clip_override)
            l_enhanced = self._clahe.apply(l_channel)
            self._clahe.setClipLimit(original_clip)
        else:
            l_enhanced = self._clahe.apply(l_channel)

        lab[:, :, 0] = l_enhanced
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return img_enhanced, True

    @staticmethod
    def _gaussian_denoise(img: np.ndarray) -> np.ndarray:
        """
        Apply light Gaussian blur to reduce high-frequency camera sensor noise.

        3×3 kernel with sigmaX=0 (OpenCV auto-computes from kernel size).
        This is intentionally gentle — strong blurring destroys fine
        facial texture needed for recognition accuracy.

        On Android ARM, a 3×3 Gaussian is NEON-vectorised and takes <0.2 ms
        at 112×112.  At 224×224 it is still under 0.5 ms.
        """
        return cv2.GaussianBlur(img, GAUSSIAN_KERNEL, sigmaX=0)

    # ------------------------------------------------------------------
    # Private helpers — face alignment
    # ------------------------------------------------------------------

    def _align_face_affine(
        self,
        img: np.ndarray,
        landmarks: FaceLandmarks,
    ) -> Tuple[np.ndarray, bool]:
        """
        Align face to canonical eye positions using a similarity transform.

        A similarity transform (uniform scale + rotation + translation, no
        shear) is the minimum warp needed to normalise inter-eye geometry.
        It preserves aspect ratio and avoids the keystone distortion that
        a full affine (with shear) can introduce on faces at extreme angles.

        Canonical target positions (ArcFace standard, 112×112 output):
            left_eye  → (30.29, 51.70)
            right_eye → (65.53, 51.50)

        The eyes are positioned slightly above vertical centre so the
        forehead and chin are approximately symmetrically included — an
        empirically-verified optimum for MobileFaceNet accuracy.

        Parameters
        ----------
        img : np.ndarray
            RGB uint8 face image, any size.
        landmarks : FaceLandmarks
            Eye centre coordinates in img pixel space.

        Returns
        -------
        tuple[np.ndarray, bool]
            (aligned_img, success_flag)
            On numerical failure (degenerate eye positions), returns
            (original_img, False) rather than raising — the pipeline
            continues with the un-aligned crop.
        """
        src_pts = np.array(
            [landmarks.left_eye, landmarks.right_eye],
            dtype=np.float32,
        )
        dst_pts = self._canonical_dst  # Precomputed at init

        # Guard against degenerate eye positions (same point, or off-image)
        if not self._landmarks_are_valid(src_pts, img.shape):
            logger.warning(
                "_align_face_affine: degenerate landmarks %s — skipping alignment.",
                src_pts.tolist(),
            )
            return img, False

        # Compute similarity transform: 2 point correspondences → 4 DOF
        # (tx, ty, scale, rotation).  OpenCV's estimateAffinePartial2D is
        # more robust than manual trig for this use case.
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,  # RANSAC handles one bad landmark gracefully
            ransacReprojThreshold=5.0,
        )

        if M is None:
            logger.warning(
                "_align_face_affine: estimateAffinePartial2D failed — "
                "skipping alignment."
            )
            return img, False

        # Apply warp — output size matches the canonical 112×112 target
        aligned = cv2.warpAffine(
            img,
            M,
            (MOBILEFACENET_INPUT_SIZE, MOBILEFACENET_INPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,  # Replicate avoids black borders
        )

        return aligned, True

    @staticmethod
    def _landmarks_are_valid(
        pts: np.ndarray,
        img_shape: Tuple[int, ...],
    ) -> bool:
        """
        Quick sanity check for eye landmark positions.

        Rejects:
        - NaN or Inf values
        - Both eyes at the same pixel (detector artefact)
        - Eye distance < 10 px (face too small or partially off-frame)
        - Coordinates outside [0, img_width / img_height]
        """
        if not np.isfinite(pts).all():
            return False

        h, w = img_shape[:2]
        if (pts < 0).any():
            return False
        if pts[0, 0] >= w or pts[1, 0] >= w:
            return False
        if pts[0, 1] >= h or pts[1, 1] >= h:
            return False

        inter_eye_dist = float(np.linalg.norm(pts[0] - pts[1]))
        if inter_eye_dist < 10.0:
            return False

        return True

    # ------------------------------------------------------------------
    # Private helpers — normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_mobilefacenet(img: np.ndarray) -> np.ndarray:
        """
        Normalise pixel values to [-1, 1] for MobileFaceNet / ArcFace.

        Formula: (pixel - 127.5) / 127.5

        This maps:
            0   → -1.0
            128 → ~0.0
            255 → +1.0

        float32 output avoids implicit float64 promotion (which would
        double memory footprint) and matches TFLite's expected input dtype
        for float32-input INT8-weight models.

        The explicit astype() before arithmetic avoids a subtle NumPy
        footgun: uint8 arithmetic wraps around (255 + 1 = 0), so converting
        BEFORE subtraction is mandatory.
        """
        return (img.astype(np.float32) - MOBILEFACENET_MEAN) / MOBILEFACENET_STD

    # ------------------------------------------------------------------
    # Private helpers — validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_roi(img: np.ndarray, min_size: int = 16) -> None:
        """
        Validate that the input image is usable before expensive processing.

        Raises ValueError with a descriptive message so the pipeline
        orchestrator can log the specific failure reason and skip the frame
        rather than crashing the auth session.
        """
        if img is None:
            raise ValueError(
                "ImageProcessor received None as face_roi. "
                "Ensure the face detector returned a valid bounding box."
            )
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel BGR image, got shape {img.shape}. "
                "Pass the raw OpenCV camera frame without channel manipulation."
            )
        if img.shape[0] < min_size or img.shape[1] < min_size:
            raise ValueError(
                f"Face ROI {img.shape[:2]} is below minimum {min_size}×{min_size}. "
                "The detected face is too small for reliable preprocessing. "
                "Consider adjusting the detector's minimum face size threshold."
            )
        if img.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 input, got {img.dtype}. "
                "ImageProcessor expects raw uint8 camera frames. "
                "Do not pre-convert to float before calling this method."
            )
"""
test_pipeline.py
================
SecureEdge AI — Full Pipeline Test Harness

Runs the complete authentication pipeline and prints a structured report:
  - Face detection result
  - Liveness (CNN + blink) score
  - Embedding similarity
  - Final decision (ALLOW / DENY / PENDING)

Usage
-----
  # Test on a static image (enroll + verify from same file for a quick demo):
  python test_pipeline.py --image path/to/face.jpg

  # Enroll from one image, verify from another:
  python test_pipeline.py --enroll path/to/enroll.jpg --verify path/to/verify.jpg

  # Live webcam mode:
  python test_pipeline.py --webcam

  # Show annotated frame window:
  python test_pipeline.py --webcam --show

  # Verbose logging:
  python test_pipeline.py --image face.jpg --verbose
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.orchestrator import AuthPipeline, AuthResult, AuthDecision
from pipeline.image_processor import ImageProcessor


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Result printing ───────────────────────────────────────────────────────────

DECISION_COLOUR = {
    AuthDecision.ALLOW:   "\033[92m",   # Green
    AuthDecision.DENY:    "\033[91m",   # Red
    AuthDecision.PENDING: "\033[93m",   # Yellow
    AuthDecision.ERROR:   "\033[95m",   # Magenta
}
RESET = "\033[0m"


def print_result(result: AuthResult, frame_idx: int = 0) -> None:
    """Print a structured, colour-coded auth result to stdout."""
    colour = DECISION_COLOUR.get(result.decision, "")
    decision_str = f"{colour}{result.decision.value}{RESET}"

    print("\n" + "─" * 55)
    print(f"  Frame #{frame_idx:04d}   Decision: {decision_str}")
    print("─" * 55)
    print(f"  Face detected   : {'✓' if result.face_detected else '✗'}")
    if result.face_bbox:
        x1, y1, x2, y2 = result.face_bbox
        print(f"  Bounding box    : ({x1},{y1}) → ({x2},{y2})")
    print(f"  Liveness score  : {result.liveness_score:.3f}  [{result.liveness_decision}]")
    print(f"  Similarity      : {result.similarity:.4f}")
    print(f"  Matched identity: {result.identity or '—'}")
    print(f"  Pipeline time   : {result.processing_time_ms:.1f} ms")
    if result.error_message:
        print(f"  Info            : {result.error_message}")
    print("─" * 55)


def annotate_frame(frame: np.ndarray, result: AuthResult) -> np.ndarray:
    """Draw bounding box and decision label on the frame for visual feedback."""
    display = frame.copy()

    colour_map = {
        AuthDecision.ALLOW:   (0, 220, 0),
        AuthDecision.DENY:    (0, 0, 220),
        AuthDecision.PENDING: (0, 200, 255),
        AuthDecision.ERROR:   (200, 0, 200),
    }
    colour = colour_map.get(result.decision, (128, 128, 128))

    if result.face_bbox:
        x1, y1, x2, y2 = result.face_bbox
        cv2.rectangle(display, (x1, y1), (x2, y2), colour, 2)

        label = result.decision.value
        if result.identity:
            label += f" | {result.identity}"
        label += f" | sim={result.similarity:.2f} | live={result.liveness_score:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            display, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # FPS / timing overlay
    timing = f"{result.processing_time_ms:.0f} ms"
    cv2.putText(
        display, timing, (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 0), 1, cv2.LINE_AA,
    )
    return display


# ── Test modes ────────────────────────────────────────────────────────────────

def run_image_test(
    pipeline: AuthPipeline,
    enroll_path: str,
    verify_path: str,
    show: bool = False,
) -> None:
    """Enroll from enroll_path, authenticate from verify_path."""
    print(f"\n[IMAGE MODE]  Enroll: {enroll_path}  |  Verify: {verify_path}")

    # ── Load images ────────────────────────────────────────────────────────
    enroll_frame = cv2.imread(enroll_path)
    if enroll_frame is None:
        print(f"[ERROR] Could not load enroll image: {enroll_path}")
        sys.exit(1)

    verify_frame = cv2.imread(verify_path)
    if verify_frame is None:
        print(f"[ERROR] Could not load verify image: {verify_path}")
        sys.exit(1)

    # ── Enrollment ─────────────────────────────────────────────────────────
    print("\n[1/2] Enrolling identity from enroll image...")
    enrolled = pipeline.enroll(enroll_frame, label="test_user")
    if not enrolled:
        print("[WARN] Enrollment failed — gallery is empty. Auth will return DENY.")
    else:
        print("[OK]  Enrollment successful.")

    # ── Authentication ─────────────────────────────────────────────────────
    print("\n[2/2] Running authentication on verify image...")
    pipeline.reset_liveness_session()
    result = pipeline.authenticate(verify_frame)
    print_result(result, frame_idx=0)

    if show:
        annotated = annotate_frame(verify_frame, result)
        h, w = annotated.shape[:2]
        max_dim = 900
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            annotated = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.namedWindow("SecureEdge AI — Auth Result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("SecureEdge AI — Auth Result", annotated)
        print("\n[Press any key to close window]")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_webcam_test(
    pipeline: AuthPipeline,
    camera_index: int = 0,
    show: bool = True,
    enroll_first: bool = True,
) -> None:
    """
    Live webcam authentication loop.

    First 3 seconds: enroll mode (captures enrollment frame).
    After that: continuous authentication.
    Press 'q' to quit, 'r' to re-enroll.
    """
    print(f"\n[WEBCAM MODE] Camera index: {camera_index}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}.")
        sys.exit(1)

    # Set reasonable capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\nControls: [SPACE] = Enroll current face  |  [r] = Re-enroll  |  [q] = Quit")
    print("         Waiting for enrollment... Press SPACE when face is visible.\n")

    enrolled = False
    frame_idx = 0
    pipeline.reset_liveness_session()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Webcam read failed — skipping frame.")
                continue

            frame_idx += 1

            if not enrolled:
                # Show a prompt overlay
                prompt_frame = frame.copy()
                cv2.putText(
                    prompt_frame,
                    "Press SPACE to enroll face",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
                )
                if show:
                    cv2.imshow("SecureEdge AI", prompt_frame)
            else:
                # Run authentication on every frame
                result = pipeline.authenticate(frame)

                # Reset blink state on ALLOW or DENY so next attempt starts fresh
                if result.decision in (AuthDecision.ALLOW, AuthDecision.DENY):
                    if frame_idx % 30 == 0:  # Re-log every 30 frames
                        print_result(result, frame_idx)
                    pipeline.reset_liveness_session()

                if show:
                    annotated = annotate_frame(frame, result)
                    cv2.imshow("SecureEdge AI", annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[Quit]")
                break

            elif key == ord(" "):  # SPACE → enroll
                print(f"\n[Enrolling from frame #{frame_idx}]...")
                ok = pipeline.enroll(frame, label="live_user")
                if ok:
                    enrolled = True
                    pipeline.reset_liveness_session()
                    print("[OK]  Enrolled. Starting authentication loop.")
                else:
                    print("[WARN] Enrollment failed — no face detected. Try again.")

            elif key == ord("r"):
                pipeline.load_gallery([])  # Clear gallery
                enrolled = False
                pipeline.reset_liveness_session()
                print("[Re-enroll] Gallery cleared.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ── Quick smoke-test (no model files needed) ──────────────────────────────────

def run_smoke_test(pipeline: AuthPipeline) -> None:
    """
    Run a self-contained smoke test using a synthetic face-like frame.
    No image files or models required.

    Verifies:
      - Pipeline initialises without error
      - authenticate() returns a valid AuthResult for any input
      - Processing time is within a reasonable bound
    """
    print("\n[SMOKE TEST] Generating synthetic test frame...")

    # Create a 480×640 BGR frame with a skin-tone rectangle as a fake face
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Background
    frame[:] = (80, 80, 80)
    # Fake face oval (brownish skin tone)
    cv2.ellipse(frame, (320, 240), (100, 130), 0, 0, 360, (120, 160, 200), -1)
    # Fake eyes
    cv2.circle(frame, (280, 210), 15, (40, 40, 40), -1)
    cv2.circle(frame, (360, 210), 15, (40, 40, 40), -1)
    # Fake mouth
    cv2.ellipse(frame, (320, 285), (40, 20), 0, 0, 180, (60, 80, 100), 2)

    print("[1/3] Enrolling from synthetic frame...")
    enrolled = pipeline.enroll(frame, label="synthetic_user")
    print(f"      Enrollment: {'OK' if enrolled else 'FAILED (expected without real face)'}")

    print("[2/3] Running authentication on same synthetic frame...")
    pipeline.reset_liveness_session()
    result = pipeline.authenticate(frame)
    print_result(result, frame_idx=0)

    print("[3/3] Verifying timing budget...")
    times = []
    for i in range(5):
        pipeline.reset_liveness_session()
        r = pipeline.authenticate(frame)
        times.append(r.processing_time_ms)
    avg_ms = sum(times) / len(times)
    budget_ok = avg_ms < 800
    print(
        f"      Avg pipeline time: {avg_ms:.1f} ms  "
        f"({'✓ within 800ms budget' if budget_ok else '✗ exceeds 800ms budget'})"
    )

    print("\n[SMOKE TEST COMPLETE]\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SecureEdge AI — Pipeline Test Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--image",  metavar="PATH", help="Path to verify image (enroll + verify same image)")
    mode.add_argument("--webcam", action="store_true", help="Live webcam authentication loop")
    mode.add_argument("--smoke",  action="store_true", help="Run synthetic smoke test (no models needed)")

    p.add_argument("--enroll", metavar="PATH", help="Separate enroll image (use with --image)")
    p.add_argument("--verify", metavar="PATH", help="Separate verify image (use with --image)")
    p.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    p.add_argument("--show",   action="store_true", help="Open OpenCV display window")
    p.add_argument(
        "--threshold", type=float, default=0.40,
        help="Cosine similarity threshold (default: 0.40)",
    )
    p.add_argument(
        "--no-blink", action="store_true",
        help="Disable active blink challenge (faster, less secure)",
    )
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(args.verbose)

    print("=" * 55)
    print("  SecureEdge AI — Authentication Pipeline Test")
    print("=" * 55)

    # ── Initialise pipeline ────────────────────────────────────────────────
    print("\nInitialising pipeline...")
    t0 = time.monotonic()
    pipeline = AuthPipeline(
        cosine_threshold=args.threshold,
        require_blink=not args.no_blink,
    )
    print(f"Pipeline ready in {(time.monotonic()-t0)*1000:.0f} ms\n")

    # ── Route to test mode ─────────────────────────────────────────────────
    if args.smoke or (not args.image and not args.webcam and not args.enroll):
        run_smoke_test(pipeline)

    elif args.webcam:
        run_webcam_test(pipeline, camera_index=args.camera, show=args.show)

    elif args.image:
        enroll_path = args.enroll or args.image
        verify_path = args.verify or args.image
        run_image_test(pipeline, enroll_path, verify_path, show=args.show)

    elif args.enroll and args.verify:
        run_image_test(pipeline, args.enroll, args.verify, show=args.show)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
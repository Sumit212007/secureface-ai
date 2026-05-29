import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.image_processor import ImageProcessor, FaceLandmarks

proc = ImageProcessor()
frame = cv2.imread("test_face.jpg")  # BGR uint8

# Test recognition path
result = proc.preprocess_for_recognition(frame)
assert result.tensor.shape == (1, 112, 112, 3)
assert result.tensor.dtype == np.float32
assert result.tensor.min() >= -1.0 and result.tensor.max() <= 1.0

# Test with landmarks
lm = FaceLandmarks(left_eye=(35.0, 55.0), right_eye=(77.0, 54.0))
result_aligned = proc.preprocess_for_recognition(frame, landmarks=lm)
assert result_aligned.was_aligned == True

# Test liveness path
liveness_result = proc.preprocess_for_liveness(frame)
assert liveness_result.tensor.shape == (1, 224, 224, 3)
assert liveness_result.tensor.min() >= 0.0 and liveness_result.tensor.max() <= 1.0
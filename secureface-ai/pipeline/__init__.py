"""
pipeline/__init__.py
SecureEdge AI — Pipeline package exports
"""

from pipeline.image_processor import ImageProcessor, FaceLandmarks, PreprocessResult
from pipeline.face_detector import FaceDetector, FaceDetection
from pipeline.recognizer import FaceRecognizer, EmbeddingResult, cosine_similarity
from pipeline.liveness_detector import LivenessDetector, LivenessResult, LivenessDecision
from pipeline.orchestrator import AuthPipeline, AuthResult, AuthDecision, EnrolledIdentity

__all__ = [
    "ImageProcessor", "FaceLandmarks", "PreprocessResult",
    "FaceDetector", "FaceDetection",
    "FaceRecognizer", "EmbeddingResult", "cosine_similarity",
    "LivenessDetector", "LivenessResult", "LivenessDecision",
    "AuthPipeline", "AuthResult", "AuthDecision", "EnrolledIdentity",
]
"""
MediaPipe LanguageDetector wrapper: lazy model loading and the low-level
detect() call used by both public API functions.
"""

from typing import Any

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import text as mp_text

from .config import _MODEL_PATH
from .text_cleaning import _is_nan, _clean_text

_detector = None


def _get_detector():
    """
    Lazily construct the MediaPipe LanguageDetector (loads the .tflite model on first use).
    """
    global _detector
    if _detector is None:
        base_options = mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
        options = mp_text.LanguageDetectorOptions(base_options=base_options)
        _detector = mp_text.LanguageDetector.create_from_options(options)
    return _detector


def _detect(text: Any):
    """
    Run the MediaPipe detector on cleaned text. Returns the ranked list of
    detections (language_code, probability), sorted descending by probability,
    or an empty list if the text is unusable.
    """
    if _is_nan(text):
        return []

    s = str(text)
    if not s or len(s.strip()) < 3:
        return []

    cleaned = _clean_text(s)
    if not cleaned or len(cleaned.strip()) < 3:
        return []

    result = _get_detector().detect(cleaned)
    return result.detections

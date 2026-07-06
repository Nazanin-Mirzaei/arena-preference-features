"""
Language detection utility built on Google MediaPipe's LanguageDetector task.

This package provides `detectLanguage` to detect the primary language of a text,
and `detectMultiLanguage` to detect whether a text mixes multiple languages,
using MediaPipe's native ranked probability distribution over 110 languages.
"""

from .api import detectLanguage, detectMultiLanguage

__all__ = [
    "detectLanguage",
    "detectMultiLanguage",
]

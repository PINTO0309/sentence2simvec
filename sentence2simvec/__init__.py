"""Sentence similarity (0.0-1.0) + SBERT vectors - Python 3.10+ only."""
from .core import similarity_score, sentence_vector

__all__ = ["similarity_score", "sentence_vector"]
__version__ = "0.0.1"
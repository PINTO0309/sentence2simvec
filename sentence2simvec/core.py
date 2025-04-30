"""
sentence2simvec.core
====================

Core implementation for
-----------------------
* `similarity_score` : 0-to-1 similarity of two sentences
* `sentence_vector`  : obtain a (384-dim) SBERT embedding for a sentence

Algorithm
---------
1. **Surface** similarity
  3-character-gram Jaccard index (insensitive to Japanese word boundaries).

2. **Semantic** similarity
  Sentence-BERT (MiniLM-L12-v2, multilingual) cosine similarity
  with L2-normalised embeddings (hence already in [0, 1]).

3. **Fusion**
  final_score = w_ngram * jaccard + w_embed * cosine

Both vectors can be returned for downstream tasks (e.g. time-series ML).
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from typing import Tuple

import numpy as np

__all__ = ["similarity_score", "sentence_vector"]

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _normalize(text: str) -> str:
    """
    Unicode NFKC normalisation + lower-casing + whitespace squeeze.

    Parameters
    ----------
    text : str
        Raw input sentence.

    Returns
    -------
    str
        Normalised sentence suitable for n-gram & embedding.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """
    Jaccard index between character n-gram sets.

    Parameters
    ----------
    a, b : str
        Normalised sentences.
    n : int, default 3
        Length of character n-grams.

    Returns
    -------
    float in [0, 1]
    """
    def to_set(t: str) -> set[str]:
        return {t[i : i + n] for i in range(len(t) - n + 1)} or {t}

    sa, sb = to_set(a), to_set(b)
    return len(sa & sb) / len(sa | sb)


def _embed_model(name: str):
    """Lazily load SBERT model / raise helpful error if missing."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers not installed.\n"
            "Install via  uv pip install sentence-transformers"
        ) from e
    return SentenceTransformer(name)


def _embed_vectors(a: str, b: str, model_name: str):
    """
    Encode two sentences -> (vec1, vec2, cosine).

    Returns
    -------
    np.ndarray, np.ndarray, float
        L2-normalised vectors and cosine similarity (clipped to [0, 1]).
    """
    model = _embed_model(model_name)
    v1, v2 = model.encode([a, b], normalize_embeddings=True)
    cosine = float(np.clip(np.dot(v1, v2), 0.0, 1.0))
    return v1, v2, cosine


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def sentence_vector(text: str, *, model_name: str = _DEFAULT_MODEL) -> np.ndarray:
    """
    Return the SBERT embedding (384-dim NumPy vector, L2-normalised).

    Examples
    --------
    >>> from sentence2simvec import sentence_vector
    >>> v = sentence_vector("Hello")
    >>> v.shape
    (384,)
    """
    return _embed_model(model_name).encode(text, normalize_embeddings=True)


def similarity_score(
    s1: str,
    s2: str,
    *,
    w_ngram: float = 0.3,
    w_embed: float = 0.7,
    model_name: str = _DEFAULT_MODEL,
    return_breakdown: bool = False,
    return_vectors: bool = False,
) -> (
    float
    | Tuple[float, dict]
    | Tuple[float, dict, np.ndarray, np.ndarray]
):
    """
    Hybrid (surface + semantic) similarity in the closed interval [0, 1].

    Parameters
    ----------
    s1, s2 : str
        Sentences to compare.
    w_ngram, w_embed : float
        Weights (should sum to 1.0; not enforced).
    model_name : str, default multilingual MiniLM
        SBERT model identifier.
    return_breakdown : bool, default False
        If True, also return {'jaccard': …, 'cosine': …}.
    return_vectors : bool, default False
        If True, also return the two sentence vectors.

    Returns
    -------
    score : float
        Similarity score.
    score, detail : (float, dict)
        If `return_breakdown=True`.
    score, detail, vec1, vec2 : (float, dict, ndarray, ndarray)
        If `return_vectors=True` (forces breakdown).
    """
    s1n, s2n = _normalize(s1), _normalize(s2)

    jaccard = _ngram_jaccard(s1n, s2n)
    vec1, vec2, cosine = _embed_vectors(s1n, s2n, model_name)

    score = round(w_ngram * jaccard + w_embed * cosine, 4)

    if return_vectors:
        return score, {"jaccard": jaccard, "cosine": cosine}, vec1, vec2
    if return_breakdown:
        return score, {"jaccard": jaccard, "cosine": cosine}
    return score

# ────────────────────────────────
# Quick-test entry point
# ────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick tester for sentence2simvec.core.similarity_score"
    )
    parser.add_argument("text1", help="1st sentence")
    parser.add_argument("text2", help="2nd sentence")
    parser.add_argument("--w-ngram", type=float, default=0.3)
    parser.add_argument("--w-embed", type=float, default=0.7)
    parser.add_argument("--show-vectors", action="store_true", help="print first 8 dims of each vector")
    args = parser.parse_args()

    out = similarity_score(
        args.text1, args.text2,
        w_ngram=args.w_ngram, w_embed=args.w_embed,
        return_vectors=args.show_vectors, return_breakdown=True
    )

    if args.show_vectors:
        score, detail, v1, v2 = out
    else:
        score, detail = out

    print(f"Similarity: {score:.4f}")
    print(f"  • n-gram  = {detail['jaccard']:.4f}")
    print(f"  • cosine  = {detail['cosine']:.4f}")

    if args.show_vectors:
        print("vec1[:8] =", np.round(v1[:8], 4))
        print("vec2[:8] =", np.round(v2[:8], 4))
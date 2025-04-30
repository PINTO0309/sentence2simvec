from sentence2simvec import similarity_score, sentence_vector
import numpy as np

def test_similarity_self():
    assert similarity_score("abc", "abc") == 1.0

def test_vector_dim():
    assert sentence_vector("hello").shape[0] == 384

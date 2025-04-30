# sentence2simvec
Vector-based sentence similarity **(0.0 - 1.0)** for Japanese & multilingual texts.

* 3-gram Jaccard **surface** similarity
* SBERT (MiniLM) **semantic** similarity
* **Python API** + **CLI** (`sentence2simvec`)
* Can output each sentence's embedding vector (numpy, 384-dim)
* Embedding vectors for each sentence can be obtained and saved as a NumPy array.

## Install
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3.10, 3.11, 3.12, 3.13, ...
uv venv -p 3.10 .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate

uv pip install -U sentence2simvec
or
uv pip install -e .
```

## Usage
- CLI
    ```bash
    sentence2simvec "Hello!" "Hello world!" --save-vecs ./vecs

    # Similarity: 0.7135
    #    • n-gram  = 0.2727
    #    • cosine  = 0.9024
    # vecs/vec1.npy, vecs/vec2.npy written
    ```
- Python API
    ```python
    from sentence2simvec import similarity_score, sentence_vector

    # --- Similarity (with vectors) ---------------------------
    score, details, v1, v2 = similarity_score(
        "Hello!",
        "Hello world!",
        return_vectors=True
    )
    print(score)          # 0.87
    print(details)        # {'jaccard': 0.764…, 'cosine': 0.910…}
    print(v1.shape)       # (384,)

    # --- Get a single sentence vector ------------------------
    vec = sentence_vector("Hello")
    # ※ vec is L2 regularized (||vec||₂ = 1)
    ```

## Development
```bash
uv venv -p 3.10 .venv && source .venv/bin/activate
uv pip install build twine pytest ipdb sentence-transformers numpy

# debug run
python sentence2simvec/core.py "Hello!" "Hello world!"

# tests
pytest

# build
python -m build

# upload PyPI
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-..."
twine upload dist/*
unset TWINE_USERNAME TWINE_PASSWORD
```
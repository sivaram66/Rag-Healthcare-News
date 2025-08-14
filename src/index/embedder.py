from __future__ import annotations

import os
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Streamlit Cloud has no GPU; ensure CPU and quiet tokenizers.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

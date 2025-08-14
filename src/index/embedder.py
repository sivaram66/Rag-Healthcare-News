from sentence_transformers import SentenceTransformer  # type: ignore


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

from src.index.embedder import Embedder
from src.index.vector_store import VectorStore


class Retriever:
    def __init__(self, top_k: int = 8) -> None:
        self.embedder = Embedder()
        self.vs = VectorStore()
        self.top_k = top_k

    def search(self, query: str, top_k: int | None = None):
        qemb = self.embedder.encode([query])
        return self.vs.search(qemb, top_k or self.top_k)

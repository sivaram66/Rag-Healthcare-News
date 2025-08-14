from src.index.embedder import Embedder
from src.index.vector_store import VectorStore


class Retriever:
    def __init__(self, top_k=8, persist_dir="data/vectors", collection_name="healthcare_news"):
        self.embedder = Embedder()
        self.store = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
        self.top_k = top_k

    def search(self, query):
        qv = self.embedder.encode([query])
        return self.store.search(qv, top_k=self.top_k)

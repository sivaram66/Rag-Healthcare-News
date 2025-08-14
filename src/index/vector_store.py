import chromadb  # type: ignore
from chromadb.config import Settings  # type: ignore
from pathlib import Path
import os

persist_dir = os.path.join(os.getcwd(), "chroma_data")
os.makedirs(persist_dir, exist_ok=True)
client = chromadb.PersistentClient(path=persist_dir)

class VectorStore:
    def __init__(self, persist_dir: str = "data/vectors", collection_name: str = "healthcare_news") -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.Client(Settings(is_persistent=True, persist_directory=persist_dir))
        self.collection = self.client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, embeddings, metadatas, documents):
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def search(self, query_embedding, top_k: int = 5):
        return self.collection.query(query_embeddings=query_embedding, n_results=top_k)

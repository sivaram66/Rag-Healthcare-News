import chromadb  # type: ignore
from chromadb.config import Settings  # type: ignore
from pathlib import Path
import os


class VectorStore:
    def __init__(self, persist_dir: str = "chroma_data", collection_name: str = "healthcare_news") -> None:
        # Create a local directory to persist DB (Streamlit Cloud allows /mount writes)
        persist_path = os.path.join(os.getcwd(), persist_dir)
        Path(persist_path).mkdir(parents=True, exist_ok=True)

        # Use PersistentClient to avoid runtime errors
        self.client = chromadb.PersistentClient(path=persist_path)

        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, embeddings, metadatas, documents):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def search(self, query_embedding, top_k: int = 5):
        return self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

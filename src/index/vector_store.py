import chromadb  # type: ignore
from pathlib import Path
import os


class VectorStore:
    def __init__(self, collection_name: str = "healthcare_news") -> None:
        # Use an in-memory client (no persistence — works on Streamlit Cloud)
        self.client = chromadb.Client()

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

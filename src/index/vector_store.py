import os
from pathlib import Path

# Detect if running in Streamlit Cloud
RUNNING_IN_CLOUD = os.getenv("STREAMLIT_RUNTIME") is not None

if RUNNING_IN_CLOUD:
    # Import lightweight in-memory client (no persistent server)
    import chromadb.api as chroma_api
else:
    import chromadb
    from chromadb.config import Settings


class VectorStore:
    def __init__(self, persist_dir="chroma_data", collection_name="healthcare_news"):
        if RUNNING_IN_CLOUD:
            # Use in-memory Chroma for cloud (avoids binary/server issues)
            self.client = chroma_api.Client()
        else:
            # Local persistent mode for dev
            persist_path = os.path.join(os.getcwd(), persist_dir)
            Path(persist_path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.Client(
                Settings(is_persistent=True, persist_directory=persist_path)
            )

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

    def search(self, query_embedding, top_k=5):
        return self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

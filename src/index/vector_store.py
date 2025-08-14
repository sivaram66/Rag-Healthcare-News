import os
import sys
from pathlib import Path

# Try to import chromadb; if sqlite3 is too old, shim with pysqlite3


def _import_chromadb():
    try:
        import chromadb  # type: ignore
        from chromadb.config import Settings  # type: ignore
        return chromadb, Settings
    except Exception as e:
        if "sqlite3" in str(e).lower():
            try:
                __import__("pysqlite3")
                sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
                import chromadb  # type: ignore
                from chromadb.config import Settings  # type: ignore
                return chromadb, Settings
            except Exception as inner:
                raise RuntimeError(
                    "ChromaDB import failed due to sqlite3. Ensure pysqlite3-binary is installed.") from inner
        raise


chromadb, Settings = _import_chromadb()


def _persist_path(base_dir: str) -> str:
    # Prefer Streamlit Cloud writable dir if present
    if os.path.isabs(base_dir):
        return base_dir
    cloud_data = "/mount/data"
    if os.path.isdir(cloud_data):
        return os.path.join(cloud_data, base_dir)
    return os.path.join(os.getcwd(), base_dir)


class VectorStore:
    def __init__(self, persist_dir="chroma_data", collection_name="healthcare_news"):
        # Try persistent store first; fall back to in-memory if unavailable
        persist_path = _persist_path(persist_dir)
        try:
            Path(persist_path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.Client(
                Settings(is_persistent=True, persist_directory=persist_path)
            )
        except Exception:
            self.client = chromadb.Client(Settings(is_persistent=False))

        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, embeddings, metadatas, documents):
        self.collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    def search(self, query_embedding, top_k=5):
        return self.collection.query(query_embeddings=query_embedding, n_results=top_k)

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
                    "ChromaDB import failed due to sqlite3. Install pysqlite3-binary."
                ) from inner
        raise


chromadb, Settings = _import_chromadb()


def _prefer_repo_then_cloud(base_dir: str) -> str:
    # Prefer repo path if it exists (bundled index), else use Cloud data dir.
    repo_path = os.path.join(os.getcwd(), base_dir)
    if os.path.isdir(repo_path):
        return repo_path
    cloud_root = "/mount/data"
    if os.path.isdir(cloud_root):
        return os.path.join(cloud_root, base_dir)
    return repo_path


class VectorStore:
    def __init__(self, persist_dir="data/vectors", collection_name="healthcare_news"):
        persist_path = _prefer_repo_then_cloud(persist_dir)
        try:
            Path(persist_path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.Client(
                Settings(is_persistent=True, persist_directory=persist_path)
            )
        except Exception:
            self.client = chromadb.Client(Settings(is_persistent=False))

        self.collection = self.client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, embeddings, metadatas, documents):
        self.collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    def search(self, query_embedding, top_k=5):
        return self.collection.query(query_embeddings=query_embedding, n_results=top_k)

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

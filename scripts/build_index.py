from src.index.db import get_conn
from src.processing.chunking import chunk_text
from src.index.embedder import Embedder
from src.index.vector_store import VectorStore


def build_index() -> None:
    # Read all articles and bulk-index their chunks
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id, url, title, content, published_at, source FROM articles"
        )
        rows = cur.fetchall()

    embedder = Embedder()
    store = VectorStore()

    chunk_texts: list[str] = []
    ids: list[str] = []
    metas: list[dict] = []

    for article_id, url, title, content, published_at, source in rows:
        for idx, piece in enumerate(chunk_text(content)):
            ids.append(f"{article_id}_{idx}")
            chunk_texts.append(piece)
            metas.append({
                "url": url,
                "title": title,
                "published_at": published_at,
                "source": source,
            })

    if chunk_texts:
        embs = embedder.encode(chunk_texts)
        store.add(ids, embs, metas, chunk_texts)
        print(f"Indexed {len(chunk_texts)} chunks from {len(rows)} articles.")


if __name__ == "__main__":
    build_index()

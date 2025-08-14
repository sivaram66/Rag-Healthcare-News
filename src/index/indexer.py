from typing import Iterable
from src.index.db import get_conn
from src.processing.chunking import chunk_text
from src.index.embedder import Embedder
from src.index.vector_store import VectorStore


def iter_unindexed_articles(limit: int = 500) -> Iterable[dict]:
    with get_conn() as con:
        con.execute("CREATE TABLE IF NOT EXISTS indexed(article_id TEXT PRIMARY KEY)")
        rows = con.execute(
            """
            SELECT id, title, source, published_at, content
            FROM articles
            WHERE id NOT IN (SELECT article_id FROM indexed)
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    for id_, title, source, published_at, content in rows:
        yield {
            "id": id_,
            "title": title,
            "source": source,
            "published_at": published_at,
            "content": content,
        }


def index_batch(batch_size: int = 50) -> None:
    embedder = Embedder()
    store = VectorStore()

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    with get_conn() as con:
        for art in iter_unindexed_articles():
            for j, chunk in enumerate(chunk_text(art["content"])):
                ids.append(f"{art['id']}-{j:03d}")
                docs.append(chunk)
                metas.append(
                    {
                        "article_id": art["id"],
                        "title": art["title"],
                        "source": art["source"],
                        "published_at": art["published_at"],
                    }
                )

                if len(ids) >= batch_size:
                    embs = embedder.encode(docs).tolist()
                    store.add(ids=ids, embeddings=embs, metadatas=metas, documents=docs)
                    con.executemany(
                        "INSERT OR IGNORE INTO indexed(article_id) VALUES(?)",
                        [(m["article_id"],) for m in metas],
                    )
                    con.commit()
                    ids, docs, metas = [], [], []

        if docs:
            embs = embedder.encode(docs).tolist()
            store.add(ids=ids, embeddings=embs, metadatas=metas, documents=docs)
            con.executemany(
                "INSERT OR IGNORE INTO indexed(article_id) VALUES(?)",
                [(m["article_id"],) for m in metas],
            )
            con.commit()


if __name__ == "__main__":
    index_batch()

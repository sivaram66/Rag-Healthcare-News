import sqlite3
from pathlib import Path

DB_PATH = Path("data/articles.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            summary TEXT,
            content TEXT,
            published_at TEXT,
            source TEXT
        )
        """
    )
    return con


def upsert_article(article: dict) -> None:
    with get_conn() as con:
        con.execute(
            """
            INSERT INTO articles (id, url, title, summary, content, published_at, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                summary=excluded.summary,
                content=excluded.content,
                published_at=excluded.published_at,
                source=excluded.source
            """,
            (
                article.get("id"),
                article.get("url"),
                article.get("title"),
                article.get("summary"),
                article.get("content"),
                article.get("published_at"),
                article.get("source"),
            ),
        )

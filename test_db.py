from src.ingestion.rss_sources import HEALTHCARE_RSS
from src.ingestion.fetch_rss import fetch_and_store
from src.index.db import get_conn

total = 0
for feed in HEALTHCARE_RSS:
    total += fetch_and_store(feed, max_items=2)

print(f"Fetched & stored {total} articles")

with get_conn() as con:
    rows = con.execute(
        "SELECT id, title, source, published_at FROM articles LIMIT 5"
    ).fetchall()

print("\nSample DB rows:")
for row in rows:
    print(row)

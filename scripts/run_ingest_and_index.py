from src.ingestion.rss_sources import HEALTHCARE_RSS
from src.ingestion.fetch_rss import fetch_and_store
from src.index.indexer import index_batch


if __name__ == "__main__":
    total = 0
    for feed in HEALTHCARE_RSS:
        total += fetch_and_store(feed, max_items=5)

    print(f"Fetched {total} articles")
    index_batch()
    print("Indexing complete.")

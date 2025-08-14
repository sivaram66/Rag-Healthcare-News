from src.ingestion.rss_sources import HEALTHCARE_RSS
from src.ingestion.fetch_rss import fetch_and_store, ARTICLES

total = 0
for feed in HEALTHCARE_RSS:
    count = fetch_and_store(feed, max_items=3)
    print(f"Fetched {count} from {feed}")
    total += count

print(f"\nTotal articles fetched: {total}")
print("\nSample article:\n", ARTICLES[0])

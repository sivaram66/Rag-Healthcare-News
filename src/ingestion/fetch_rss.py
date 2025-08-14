import feedparser  # type: ignore
import hashlib
from dateutil import parser as dtp  # type: ignore
from urllib.parse import urlparse
import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
from src.index.db import upsert_article


# Scratch list for simple inspection in tests
ARTICLES: list[dict] = []


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(" ", strip=True)


def fetch_fulltext(url: str) -> str:
    # Best-effort HTML → text
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception:
        return ""


def parse_entry(entry) -> dict:
    link = entry.get("link", "")
    title = (entry.get("title", "") or "").strip()
    summary = entry.get("summary", "")
    published = entry.get("published", "") or entry.get("updated", "")

    published_iso = None
    if published:
        try:
            published_iso = dtp.parse(published).isoformat()
        except Exception:
            published_iso = None

    source = urlparse(link).netloc if link else "unknown"
    content = fetch_fulltext(link) or clean_html(summary)

    item = {
        "id": _hash(link or title),
        "url": link,
        "title": title,
        "summary": clean_html(summary),
        "content": content,
        "published_at": published_iso,
        "source": source,
    }
    ARTICLES.append(item)
    return item


def fetch_and_store(feed_url: str, max_items: int = 10) -> int:
    parsed = feedparser.parse(feed_url)
    subset = parsed.entries[:max_items]
    for entry in subset:
        upsert_article(parse_entry(entry))
    return len(subset)

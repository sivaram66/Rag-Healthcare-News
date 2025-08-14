import time
from typing import List, Tuple
from src.retriever.hybrid import Retriever


def topk_hit_rate(queries: List[Tuple[str, str]], k: int = 5) -> float:
    r = Retriever(top_k=k)
    hits = 0
    for q, expected_article_id in queries:
        start = time.time()
        res = r.search(q)
        latency = time.time() - start
        metas = res["metadatas"][0]
        if any(m.get("article_id") == expected_article_id for m in metas):
            hits += 1
        print(f"Query '{q}' took {latency:.2f}s")
    return hits / len(queries)

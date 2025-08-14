from src.retriever.hybrid import Retriever


if __name__ == "__main__":
    retriever = Retriever(top_k=5)
    query = "What did EMA announce about medicines or safety recently?"
    res = retriever.search(query)
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        print(f"\nResult {i}: {meta.get('title')} — {meta.get('source')} — {meta.get('published_at')}")
        print(doc[:300], "...")

import os
import sys
from urllib.parse import urlparse

import litellm  # type: ignore
import streamlit as st  # type: ignore
from dotenv import load_dotenv  # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.credibility.scoring import compute_score
from src.detection.flags import suspicion_score
from src.factcheck.pipeline import wiki_search
from src.retriever.hybrid import Retriever
from src.ingestion.rss_sources import HEALTHCARE_RSS
from src.ingestion.fetch_rss import fetch_and_store
from src.index.indexer import index_batch

load_dotenv()


class OpenAILLM:
    def __init__(self) -> None:
        self.api_base = os.getenv("API_BASE", "https://openrouter.ai/api/v1")
        self.api_key = os.getenv("API_KEY")

    def generate(self, claim: str, evidence_list: list[dict]) -> str:
        joined = "\n\n".join(f"{e['title']}: {e['summary']}" for e in evidence_list)
        prompt = (
            f"Claim: {claim}\n\n"
            f"Evidence:\n{joined}\n\n"
            "Classify the claim as True, False, Partly True, or Unclear, and give a short explanation."
        )
        resp = litellm.completion(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=0,
        )
        return resp["choices"][0]["message"]["content"]


st.set_page_config(page_title="Healthcare News RAG", layout="wide")
st.title("Healthcare News RAG — Prototype")

# Sidebar config
top_k = st.sidebar.slider("Top-K", 3, 15, 8)

# Query input
query = st.text_input("Ask a healthcare news question:")

# Sidebar: quick bootstrap for Cloud
init_clicked = st.sidebar.button("Initialize / Refresh index")
if init_clicked:
    with st.spinner("Fetching a few items and building the index..."):
        for feed in HEALTHCARE_RSS:
            fetch_and_store(feed, max_items=5)
        index_batch(batch_size=64)
    st.sidebar.success("Index refreshed")

# Create retriever pointing at bundled index path
retriever = Retriever(top_k=top_k, persist_dir="data/vectors")
st.sidebar.caption(f"Indexed chunks: {retriever.store.count()}")

if st.button("Search") and query:
    r = Retriever(top_k=top_k)
    res = r.search(query)
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    if not docs:
        st.warning("No results yet. Try a different question.")
    else:
        llm = OpenAILLM()
        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            st.markdown(f"### {i}. {meta.get('title','')}")
            st.caption(f"{meta.get('source','')} — {meta.get('published_at','')}")
            st.write(doc[:800] + "…")

            claim = doc[:300]
            evidence = wiki_search(claim)
            verdict_text = llm.generate(claim, evidence)

            domain = urlparse(meta.get("source", "")).netloc
            s_score = suspicion_score(doc)
            if isinstance(s_score, tuple):
                s_score = s_score[0]
            cred = compute_score(domain, evidence, 0.5, s_score)

            with st.expander("Fact-check & credibility"):
                st.write("**Verdict:**", verdict_text)
                st.write("**Credibility:**", f"{cred} / 100")
                st.write("**Evidence:")
                for evi in evidence:
                    st.markdown(f"- **{evi['title']}** — {evi['summary'][:200]}...")

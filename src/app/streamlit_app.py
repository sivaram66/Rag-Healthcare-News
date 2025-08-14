import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Pull keys from Streamlit Secrets (OpenRouter) and standardize env
try:
    if "API_KEY" in st.secrets and st.secrets["API_KEY"]:
        os.environ["OPENAI_API_KEY"] = st.secrets["API_KEY"]           # for OpenAI-compatible clients
        os.environ["OPENROUTER_API_KEY"] = st.secrets["API_KEY"]       # explicit OpenRouter var
    if "API_BASE" in st.secrets and st.secrets["API_BASE"]:
        os.environ["OPENAI_API_BASE"] = st.secrets["API_BASE"]         # used by OpenAI-compatible clients
        os.environ["OPENROUTER_BASE_URL"] = st.secrets["API_BASE"]
except Exception:
    pass

try:
    import litellm
except Exception:
    litellm = None

class SafeLLM:
    def __init__(self, model: str | None = None):
        # OpenRouter model names are namespaced; this one proxies OpenAI's gpt-4o-mini
        self.model = model or os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini")
        self.api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("API_KEY")
        )
        self.api_base = (
            os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENROUTER_BASE_URL")
            or os.getenv("API_BASE")
            or "https://openrouter.ai/api/v1"
        )

    def generate(self, claim: str, evidence: list[dict]) -> str:
        if not litellm or not self.api_key:
            return "LLM not configured. Skipping."
        try:
            blurbs = []
            for ev in evidence or []:
                title = ev.get("title") or ev.get("page") or ""
                text = ev.get("extract") or ev.get("summary") or ev.get("text") or ""
                blurbs.append(f"- {title}: {text[:400] if text else ''}")
            prompt = f"Claim: {claim}\nEvidence:\n" + "\n".join(blurbs) + "\nRespond yes/no with one brief sentence."
            resp = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=20,
                api_base=self.api_base,
                api_key=self.api_key,
            )
            msg = resp.get("choices", [{}])[0].get("message", {}).get("content") or ""
            return msg.strip() or "No verdict."
        except Exception:
            return "LLM call failed. Skipping."

# instantiate once
llm = SafeLLM()

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

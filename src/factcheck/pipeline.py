import requests  # type: ignore
from typing import List

WIKI_API = "https://en.wikipedia.org/w/api.php"


def wiki_search(query: str, max_results: int = 3) -> list[dict]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "utf8": "",
        "format": "json",
    }
    try:
        res = requests.get(WIKI_API, params=params, timeout=10).json()
    except Exception:
        return []

    items = res.get("query", {}).get("search", [])
    out: list[dict] = []
    for it in items[:max_results]:
        pid = it.get("pageid")
        title = it.get("title", "")
        try:
            page = requests.get(
                WIKI_API,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": True,
                    "format": "json",
                    "pageids": pid,
                },
                timeout=10,
            ).json()["query"]["pages"][str(pid)]
            summary = page.get("extract", "").strip()
        except Exception:
            summary = ""
        if summary:
            out.append({"title": title, "summary": summary})
    return out


def fetch_wiki_extract(title: str) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        "titles": title,
    }
    r = requests.get(WIKI_API, params=params, timeout=15)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    return next(iter(pages.values())).get("extract", "")


def build_factcheck_prompt(claim: str, evidence_blurbs: List[str]) -> str:
    joined = "\n\n".join(evidence_blurbs)
    return (
        "You are a healthcare fact-checker.\n"
        f"Claim: \"{claim}\"\n\n"
        f"Evidence from reliable sources:\n{joined}\n\n"
        "Verdict: True / Partly True / False / Unclear.\n"
        "Explain briefly and cite the most relevant sources."
    )


def factcheck_with_llm(llm, claim: str, evidence: list[dict]) -> dict:
    prompt = build_factcheck_prompt(claim, [e["summary"] for e in evidence])
    verdict = llm.generate(prompt)
    return {"verdict": verdict, "evidence_titles": [e["title"] for e in evidence]}

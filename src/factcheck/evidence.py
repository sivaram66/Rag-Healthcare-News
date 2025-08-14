import requests  # type: ignore


def get_wikipedia_evidence(query, max_results=3):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "utf8": "",
        "format": "json",
    }
    res = requests.get(url, params=params).json()
    out = []
    for item in res.get("query", {}).get("search", [])[:max_results]:
        page_id = item.get("pageid")
        fetch = requests.get(
            url,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "format": "json",
                "pageids": page_id,
            },
        ).json()
        page = fetch.get("query", {}).get("pages", {}).get(str(page_id), {})
        out.append({
            "title": page.get("title", ""),
            "summary": (page.get("extract", "") or "")[:800],
        })
    return out

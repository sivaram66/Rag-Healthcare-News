def chunk_text(text: str, max_tokens: int = 800, overlap: int = 120) -> list[str]:
    # Character-based approximation of tokens; simple and predictable
    if not text:
        return []

    chars = max_tokens * 4
    step = chars - (overlap * 4)

    out: list[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + chars])
        i += step
    return out

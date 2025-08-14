import re

SUSPICIOUS_PATTERNS = [
    r"\bmiracle\b",
    r"\bcure(s)?\b",
    r"\bguarantee(d)?\b",
    r"\b100%\b",
    r"\bno side effects\b",
    r"\bin \d+ (days|weeks)\b",
]


def suspicion_score(text):
    # Return (score in [0,1], matched terms)
    terms = [
        "miracle cure",
        "secret remedy",
        "guaranteed",
        "100% safe",
        "no side effects",
        "ancient secret",
        "instant results",
        "cure cancer",
        "risk free",
        "breakthrough discovery",
    ]

    lowered = text.lower()
    flagged = []
    matches = 0

    for t in terms:
        if re.search(rf"\b{re.escape(t)}\b", lowered):
            flagged.append(t)
            matches += 1

    score = min(matches / len(terms), 1.0)
    return score, flagged

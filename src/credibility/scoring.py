def compute_score(source_domain, evidence_list, base_confidence=0.5, suspicion_score_val=0):
    """
    Compute credibility score (0-100) based on:
    - Trusted source domain
    - Quantity and quality of evidence
    - Suspicion score (misleading language)
    - Base confidence (from retrieval quality)
    """

    # Start from base confidence scaled to 100
    score = int(base_confidence * 100)

    # ✅ Boost for trusted domains
    trusted_sources = [
        "medpagetoday.com",
        "bmj.com",
        "ema.europa.eu",
        "medlineplus.gov"
    ]
    if any(src in source_domain for src in trusted_sources):
        score += 15

    # ✅ Boost based on evidence quality
    if evidence_list:
        # More evidence → higher boost
        score += min(len(evidence_list) * 10, 25)

        # Longer summaries → stronger support
        avg_summary_len = sum(len(ev.get("summary", ""))
                              for ev in evidence_list) / len(evidence_list)
        if avg_summary_len > 300:
            score += 10
        elif avg_summary_len > 150:
            score += 5

    # ❌ Penalty for suspicious/misleading language
    score -= int(suspicion_score_val * 20)  # suspicion score between 0 and 1

    # Clamp between 0 and 100
    score = max(0, min(score, 100))

    return score

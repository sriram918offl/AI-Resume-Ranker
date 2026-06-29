"""
Per-candidate reasoning string generator.

Spec rules we honour:
  * Must be specific to each candidate (no copy-paste)
  * Must NOT mention skills not in the candidate's profile (no hallucination)
  * Must NOT contradict the rank
  * Should mention strengths AND concerns concretely

Approach: assemble 2-3 short fact-clauses pulled from the candidate's own data,
plus the dominant strength and dominant concern from the ScoreBreakdown.
Output is one sentence, 100-200 characters.
"""

from __future__ import annotations

from ranker.scoring import ScoreBreakdown


def _top_matched_skill(breakdown: ScoreBreakdown, features: dict) -> str | None:
    """Return the strongest verifiable skill name (matched against JD clusters)."""
    if not breakdown.matched_skills:
        return None
    # Prefer the matched skill that ALSO has the highest trust signal
    trust_by_name = {
        s["raw_name"] or s["name"]: (s["endorsements"], s["duration_months"])
        for s in features["skills"]
    }
    matched_with_trust = sorted(
        breakdown.matched_skills,
        key=lambda n: trust_by_name.get(n, (0, 0)),
        reverse=True,
    )
    return matched_with_trust[0]


def _strength_clause(breakdown: ScoreBreakdown, features: dict) -> str:
    """Pick the single most defensible strength."""
    parts: list[str] = []

    # 1. Title is best signal when it matches
    title = features["current_title_full"]
    if breakdown.title_career >= 0.7:
        parts.append(f"{title}")

    # 2. Years of experience
    yoe = features["years_of_experience"]
    if 5 <= yoe <= 9:
        parts.append(f"{yoe:.1f} yrs")
    elif yoe > 0:
        parts.append(f"{yoe:.1f} yrs (outside ideal 5-9 band)")

    # 3. Top matched skill
    top_skill = _top_matched_skill(breakdown, features)
    if top_skill:
        # Find the proficiency
        match = next((s for s in features["skills"] if (s["raw_name"] or s["name"]) == top_skill), None)
        if match and match["endorsements"] > 0:
            parts.append(f"{top_skill} ({match['proficiency']}, {match['endorsements']} endorsements)")
        else:
            parts.append(top_skill)

    return ", ".join(parts)


def _concern_clause(breakdown: ScoreBreakdown, features: dict) -> str | None:
    """Pick the single most important concern, if any."""
    if breakdown.honeypot_penalty > 0.3:
        return f"honeypot flags: {breakdown.honeypot_reasons[0]}"

    if features["consulting_ratio"] > 0.6:
        return f"{features['consulting_ratio']:.0%} consulting career"

    if features["recruiter_response_rate"] < 0.2 and features["days_since_active"] > 60:
        return (
            f"low engagement (response rate {features['recruiter_response_rate']:.0%}, "
            f"inactive {features['days_since_active']}d)"
        )

    if breakdown.missing_skills and len(breakdown.missing_skills) >= 3:
        return f"missing required clusters: {', '.join(breakdown.missing_skills[:2])}"

    if features["years_of_experience"] < 4:
        return f"only {features['years_of_experience']:.1f} yrs experience"
    if features["years_of_experience"] > 10:
        return f"{features['years_of_experience']:.1f} yrs (above target seniority)"

    if features["short_roles"] >= 3:
        return f"{features['short_roles']} short roles (<18mo) — title-chaser pattern"

    if features["notice_period_days"] > 60:
        return f"long notice period ({features['notice_period_days']}d)"

    # No major concern
    return None


def build_reasoning(breakdown: ScoreBreakdown, features: dict, *, rank: int) -> str:
    """
    Build a 1-2 sentence reasoning string. Examples produced:
      - "AI Engineer, 6.5 yrs, FAISS (advanced, 28 endorsements); active on Redrob (response 78%)."
      - "Backend Engineer with 6.9 yrs and Milvus depth, but missing embedding-retrieval cluster."
      - "Marketing Manager with AI keywords — likely keyword stuffer; ranked low."
    """
    strength = _strength_clause(breakdown, features) or "Adjacent profile"
    concern = _concern_clause(breakdown, features)

    # Tail clause about engagement (positive only — mentioned negatively in concern)
    eng = features["recruiter_response_rate"]
    if eng >= 0.6 and features["days_since_active"] <= 45:
        engagement = f"active on Redrob (response {eng:.0%})"
    elif eng >= 0.4:
        engagement = f"moderate engagement (response {eng:.0%})"
    else:
        engagement = None

    pieces: list[str] = [strength]
    if engagement and not concern:
        pieces.append(engagement)
    if concern:
        pieces.append(f"but {concern}")

    sentence = "; ".join(pieces)
    # Capitalise first letter
    if sentence and not sentence[0].isupper():
        sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith("."):
        sentence += "."

    # Hard cap at ~250 chars to keep CSV clean
    return sentence[:250]

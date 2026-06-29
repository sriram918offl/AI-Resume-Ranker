"""
Multi-component scorer for the Redrob Senior AI Engineer JD.

Components (sum to 1.0 except behavioral which is multiplicative):

  title_career_score   0.28   -- title match + career-history evidence
  skills_score         0.22   -- required-skill coverage with trust multiplier
  experience_score     0.13   -- bell curve around JD's 6-8 ideal
  jd_alignment_score   0.17   -- production signals, anti-keyword-stuffer, anti-research
  location_score       0.05   -- Pune/Noida preferred, India OK, willing-to-relocate boost
  (semantic_score)     0.15   -- cosine(candidate text, JD seed) [optional]

then * behavioral_modifier  (0.4..1.0)
then - honeypot_penalty    (0..1)
then - hard_dq             (e.g. consulting-only 0.4)

Final clamped to [0, 1].

This file is deliberately deterministic and CPU-only. No LLM calls. No network.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from ranker.honeypot import honeypot_score
from ranker.jd_config import (
    BONUS_CITIES,
    CAREER_SIGNAL_PHRASES,
    CONSULTING_FIRMS,
    EXP_ACCEPTABLE,
    EXP_HARD_RANGE,
    EXP_PEAK_RANGE,
    INDIA_OK,
    KEYWORD_STUFFER_TITLES,
    NICE_SKILL_CLUSTERS,
    OTHER_DOMAIN_CLUSTERS,
    PRIMARY_CITIES,
    PRIMARY_TITLE_TERMS,
    REQUIRED_SKILL_CLUSTERS,
    RESEARCH_ONLY_TERMS,
)


@dataclass
class ScoreBreakdown:
    """Full reasoning trail behind a single candidate's score."""

    final: float
    title_career: float
    skills: float
    experience: float
    jd_alignment: float
    location: float
    semantic: float
    behavioral_mult: float
    honeypot_penalty: float
    hard_dq_penalty: float
    matched_skills: list[str]
    missing_skills: list[str]
    honeypot_reasons: list[str]
    jd_alignment_notes: list[str]


# ---------- title + career history ---------------------------------------

def _title_career_score(f: dict) -> tuple[float, list[str]]:
    """Reward title match + career-history evidence; penalise stuffer titles."""
    notes: list[str] = []
    title = f["current_title"]
    headline = f["headline"]
    summary = f["summary"]
    career = f["career_text"]

    # Title match — strict and bounded
    title_hit = any(t in title for t in PRIMARY_TITLE_TERMS)
    headline_hit = any(t in headline for t in PRIMARY_TITLE_TERMS)
    title_score = 0.0
    if title_hit:
        title_score = 1.0
        notes.append("title matches AI/ML engineer family")
    elif headline_hit:
        title_score = 0.6
        notes.append("headline (not title) matches AI/ML family")

    # Keyword-stuffer penalty: title is unrelated but career text uses AI words
    stuffer_hit = any(t in title for t in KEYWORD_STUFFER_TITLES)
    if stuffer_hit:
        # Did they pile AI keywords in skills/summary anyway?
        ai_words = ("rag", "embedding", "transformer", "llm", "fine-tun", "vector")
        skills_blob = " ".join(s["name"] for s in f["skills"])
        if any(w in skills_blob or w in summary for w in ai_words):
            title_score = 0.0
            notes.append(f"keyword-stuffer pattern: '{title}' but AI words in skills")
        else:
            title_score = max(title_score - 0.5, 0.0)
            notes.append(f"non-AI title '{title}'")

    # Career history evidence — count signal phrases in role descriptions
    phrase_hits = sum(1 for p in CAREER_SIGNAL_PHRASES if p in career)
    # Diminishing returns: 0 -> 0, 1 -> 0.25, 3 -> 0.6, 6+ -> ~0.9
    career_evidence = 1 - exp(-phrase_hits / 4.0)
    if phrase_hits >= 3:
        notes.append(f"career text shows {phrase_hits} retrieval/ranking phrases")

    # Combine — 60% title, 40% career evidence
    score = 0.6 * title_score + 0.4 * career_evidence
    return min(score, 1.0), notes


# ---------- skills ------------------------------------------------------

def _skill_trust(skill: dict) -> float:
    """
    Trust multiplier per the JD's anti-keyword-stuffer principle:
    "proficiency × endorsements × duration".
    """
    endorsement_signal = min(skill["endorsements"] / 30.0, 1.0)  # 30+ endorsements saturates
    duration_signal = min(skill["duration_months"] / 24.0, 1.0)  # 24+ months saturates
    # Floor at 0.2 so a low-endorsement skill isn't entirely worthless,
    # but it's clearly down-weighted.
    return skill["weight"] * (0.4 + 0.3 * endorsement_signal + 0.3 * duration_signal)


def _cluster_coverage(skills: list[dict], cluster: list[str]) -> tuple[float, list[str]]:
    """How well do this candidate's skills cover a required cluster?"""
    matched: list[str] = []
    total_trust = 0.0
    for skill_obj in skills:
        sn = skill_obj["name"]
        for term in cluster:
            if term in sn:
                matched.append(skill_obj["raw_name"] or skill_obj["name"])
                total_trust += _skill_trust(skill_obj)
                break
    # 1 strong skill in cluster ~= 0.4, 2+ = saturates fast
    coverage = 1 - exp(-total_trust / 0.8)
    return coverage, matched


def _skills_score(f: dict) -> tuple[float, list[str], list[str]]:
    """Coverage of REQUIRED + bonus from NICE clusters."""
    matched_all: list[str] = []
    missing: list[str] = []

    required_score = 0.0
    for name, cluster in REQUIRED_SKILL_CLUSTERS.items():
        cov, matched = _cluster_coverage(f["skills"], cluster)
        required_score += cov
        if matched:
            matched_all.extend(matched)
        else:
            missing.append(name)
    required_score /= len(REQUIRED_SKILL_CLUSTERS)  # avg

    # Nice-to-have bump (max +0.2)
    nice_bump = 0.0
    for cluster in NICE_SKILL_CLUSTERS.values():
        cov, matched = _cluster_coverage(f["skills"], cluster)
        nice_bump += cov * 0.05
        if matched:
            matched_all.extend(matched)
    nice_bump = min(nice_bump, 0.2)

    return min(required_score + nice_bump, 1.0), matched_all, missing


# ---------- experience ---------------------------------------------------

def _experience_score(f: dict) -> float:
    """Bell curve around JD's 6-8 ideal, zero outside [3, 12]."""
    yoe = f["years_of_experience"]
    if yoe <= EXP_HARD_RANGE[0] or yoe >= EXP_HARD_RANGE[1]:
        return 0.0
    if EXP_PEAK_RANGE[0] <= yoe <= EXP_PEAK_RANGE[1]:
        return 1.0
    if EXP_ACCEPTABLE[0] <= yoe <= EXP_ACCEPTABLE[1]:
        # linear ramp from edge of acceptable to peak
        if yoe < EXP_PEAK_RANGE[0]:
            return (yoe - EXP_ACCEPTABLE[0]) / (EXP_PEAK_RANGE[0] - EXP_ACCEPTABLE[0])
        return (EXP_ACCEPTABLE[1] - yoe) / (EXP_ACCEPTABLE[1] - EXP_PEAK_RANGE[1])
    # Between hard and acceptable: linear ramp to 0
    if yoe < EXP_ACCEPTABLE[0]:
        return 0.3 * (yoe - EXP_HARD_RANGE[0]) / (EXP_ACCEPTABLE[0] - EXP_HARD_RANGE[0])
    return 0.3 * (EXP_HARD_RANGE[1] - yoe) / (EXP_HARD_RANGE[1] - EXP_ACCEPTABLE[1])


# ---------- JD alignment / anti-patterns ---------------------------------

def _jd_alignment(f: dict) -> tuple[float, list[str]]:
    """
    Explicit JD signals that aren't covered by title/skills/experience:
    + production deployment language in career history
    + product-company exposure
    - consulting-firm-only careers
    - pure research patterns
    - vision/speech/robotics dominant without NLP
    - title-chaser pattern (many short stints)
    - "framework enthusiast" — only recent LangChain-style buzzwords, no underlying ML
    """
    notes: list[str] = []
    score = 0.5  # start neutral

    # + production deployment language (already partly in title_career, but reinforce)
    prod_terms = ("production", "shipped", "deployed", "real users", "scale", "millions", "live")
    prod_hits = sum(1 for t in prod_terms if t in f["career_text"])
    if prod_hits >= 2:
        score += 0.15
        notes.append(f"production-deployment language present ({prod_hits} hits)")

    # - consulting ratio
    if f["consulting_ratio"] > 0.8:
        score -= 0.40
        notes.append(f"consulting-firm-only career ({f['consulting_ratio']:.0%})")
    elif f["consulting_ratio"] > 0.5:
        score -= 0.15
        notes.append(f"majority consulting career ({f['consulting_ratio']:.0%})")

    # - title-chaser
    if f["num_roles"] >= 4 and f["short_roles"] / max(f["num_roles"], 1) > 0.5:
        score -= 0.15
        notes.append(f"title-chaser pattern: {f['short_roles']}/{f['num_roles']} roles <18mo")

    # - pure research
    research_text = f["current_title"] + " " + f["headline"]
    is_research = any(t in research_text for t in RESEARCH_ONLY_TERMS)
    if is_research and prod_hits < 1:
        score -= 0.25
        notes.append("research-leaning title with no production language")

    # - vision/speech/robotics dominant without NLP/IR
    skills_blob = " ".join(s["name"] for s in f["skills"])
    other_hits = sum(
        any(term in skills_blob for term in cluster)
        for cluster in OTHER_DOMAIN_CLUSTERS.values()
    )
    nlp_hits = any(
        any(term in skills_blob for term in cluster)
        for cluster in [REQUIRED_SKILL_CLUSTERS["nlp_core"], REQUIRED_SKILL_CLUSTERS["embedding_retrieval"]]
    )
    if other_hits >= 2 and not nlp_hits:
        score -= 0.25
        notes.append("vision/speech/robotics-dominant without NLP/IR signals")

    # - framework-enthusiast: only langchain-style mentions, no underlying ML
    framework_signal = ("langchain" in skills_blob or "langchain" in f["career_text"])
    has_real_ml = any(
        term in skills_blob
        for term in ("pytorch", "tensorflow", "scikit", "transformers", "embedding", "fine-tun")
    )
    if framework_signal and not has_real_ml and f["years_of_experience"] < 4:
        score -= 0.15
        notes.append("framework-enthusiast pattern (LangChain without underlying ML)")

    return max(min(score, 1.0), 0.0), notes


# ---------- location ----------------------------------------------------

def _location_score(f: dict) -> float:
    loc = f["location"]
    country = f["country"]
    relocate = f["willing_to_relocate"]

    if any(c in loc for c in PRIMARY_CITIES):
        return 1.0
    if any(c in loc for c in BONUS_CITIES):
        return 0.85
    if any(c in country for c in INDIA_OK):
        return 0.6 if relocate else 0.45
    # Outside India
    return 0.30 if relocate else 0.10


# ---------- behavioral multiplier ---------------------------------------

def _behavioral_modifier(f: dict) -> float:
    """
    Multiplicative factor 0.4..1.0. The JD explicitly warns that low engagement
    means "for hiring purposes, not actually available." We respect that, but
    don't fully zero out otherwise-strong candidates (hence the 0.4 floor).
    """
    rrr = f["recruiter_response_rate"]
    icr = f["interview_completion_rate"]
    pc = f["profile_completeness"] / 100.0
    days = f["days_since_active"]
    saved = f["saved_by_recruiters_30d"]
    o2w = 1.0 if f["open_to_work_flag"] else 0.6

    # Recency: 1.0 at 0 days, 0.5 at 60 days, ~0.25 at 120, ~0.1 at 180
    recency = exp(-days / 90.0)

    # Combine; weight response_rate heaviest
    raw = (
        0.32 * rrr
        + 0.18 * icr
        + 0.20 * pc
        + 0.18 * recency
        + 0.06 * o2w
        + 0.06 * min(saved / 8.0, 1.0)
    )
    return 0.4 + 0.6 * raw  # clamp into [0.4, 1.0]


# ---------- main entry --------------------------------------------------

# Component weights — must sum to 1.0 for the additive part.
W = {
    "title_career": 0.28,
    "skills": 0.22,
    "experience": 0.13,
    "jd_alignment": 0.17,
    "location": 0.05,
    "semantic": 0.15,
}


def score_candidate(
    features: dict,
    *,
    semantic_score: float = 0.0,
) -> ScoreBreakdown:
    """
    Compute the final score for one candidate.

    semantic_score: optional cosine similarity vs JD embedding (0..1).
    If not provided, that weight is redistributed to title_career + skills.
    """
    title_career, tc_notes = _title_career_score(features)
    skills_s, matched, missing = _skills_score(features)
    exp_s = _experience_score(features)
    jd_s, jd_notes = _jd_alignment(features)
    loc_s = _location_score(features)

    if semantic_score <= 0:
        # Redistribute the 0.15 semantic weight: 60% title_career, 40% skills
        additive = (
            (W["title_career"] + 0.6 * W["semantic"]) * title_career
            + (W["skills"] + 0.4 * W["semantic"]) * skills_s
            + W["experience"] * exp_s
            + W["jd_alignment"] * jd_s
            + W["location"] * loc_s
        )
    else:
        additive = (
            W["title_career"] * title_career
            + W["skills"] * skills_s
            + W["experience"] * exp_s
            + W["jd_alignment"] * jd_s
            + W["location"] * loc_s
            + W["semantic"] * semantic_score
        )

    behavioral = _behavioral_modifier(features)
    hp_penalty, hp_reasons = honeypot_score(features)

    final = additive * behavioral - hp_penalty
    final = max(min(final, 1.0), 0.0)

    return ScoreBreakdown(
        final=final,
        title_career=title_career,
        skills=skills_s,
        experience=exp_s,
        jd_alignment=jd_s,
        location=loc_s,
        semantic=semantic_score,
        behavioral_mult=behavioral,
        honeypot_penalty=hp_penalty,
        hard_dq_penalty=0.0,
        matched_skills=matched,
        missing_skills=missing,
        honeypot_reasons=hp_reasons,
        jd_alignment_notes=tc_notes + jd_notes,
    )

"""
Streaming JSONL loader + feature extractor.

Loads candidates one at a time so the 464MB file doesn't blow memory, and
projects each candidate to the small feature dict the scorer needs.
"""

from __future__ import annotations

import gzip
import json
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

from ranker.jd_config import CONSULTING_FIRMS

PROFICIENCY_WEIGHT = {
    "beginner": 0.4,
    "intermediate": 0.7,
    "advanced": 0.9,
    "expert": 1.0,
}


def _open_candidates(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_candidates(path: str | Path) -> Iterator[dict]:
    """Yield parsed candidate dicts one at a time. Skips blank lines."""
    p = Path(path)
    with _open_candidates(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed rows rather than dying mid-100K
                continue


def load_candidates_from_array(path: str | Path) -> list[dict]:
    """For the sample_candidates.json (an array, not JSONL)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------- feature projection ------------------------------------------

def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except (ValueError, TypeError):
        return None


def project_features(candidate: dict, *, reference_date: date | None = None) -> dict:
    """
    Flatten the rich candidate JSON into a compact feature dict the scorer can
    consume cheaply. Done once per candidate; the scorer doesn't traverse the
    nested structure again.
    """
    ref = reference_date or date(2026, 6, 1)
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", []) or []
    education = candidate.get("education", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    # Normalise skills
    norm_skills = []
    for s in skills:
        norm_skills.append(
            {
                "name": str(s.get("name", "")).lower().strip(),
                "raw_name": s.get("name", ""),
                "proficiency": s.get("proficiency", "intermediate"),
                "endorsements": int(s.get("endorsements", 0) or 0),
                "duration_months": int(s.get("duration_months", 0) or 0),
                "weight": PROFICIENCY_WEIGHT.get(
                    s.get("proficiency", "intermediate"), 0.5
                ),
            }
        )

    # Build a single career-text blob for fast keyword matching
    career_text_parts: list[str] = []
    consulting_months = 0
    total_career_months = 0
    earliest_start: date | None = None
    company_names_lower: list[str] = []

    for role in history:
        title = str(role.get("title", "")).lower()
        company = str(role.get("company", "")).lower()
        desc = str(role.get("description", "")).lower()
        dur = int(role.get("duration_months", 0) or 0)
        start = _parse_date(role.get("start_date"))

        career_text_parts.append(f"{title} at {company}. {desc}")
        total_career_months += dur
        if start and (earliest_start is None or start < earliest_start):
            earliest_start = start
        company_names_lower.append(company)
        if company in CONSULTING_FIRMS or any(c in company for c in CONSULTING_FIRMS):
            consulting_months += dur

    career_text = "\n".join(career_text_parts).lower()
    consulting_ratio = (
        consulting_months / total_career_months if total_career_months else 0.0
    )

    # Last active recency
    last_active = _parse_date(signals.get("last_active_date"))
    days_since_active = (ref - last_active).days if last_active else 365

    # Signup recency (newer accounts get a slight penalty for "we don't know them yet")
    signup = _parse_date(signals.get("signup_date"))
    days_since_signup = (ref - signup).days if signup else 365

    # Title chaser detection: count roles with duration < 18 months
    short_roles = sum(
        1 for r in history if int(r.get("duration_months", 0) or 0) < 18
    )

    profile_summary = (profile.get("summary", "") or "").lower()
    headline = (profile.get("headline", "") or "").lower()
    current_title = (profile.get("current_title", "") or "").lower()

    return {
        "candidate_id": candidate["candidate_id"],
        "name": profile.get("anonymized_name"),
        "current_title": current_title,
        "headline": headline,
        "summary": profile_summary,
        "summary_full": profile.get("summary", "") or "",
        "headline_full": profile.get("headline", "") or "",
        "current_title_full": profile.get("current_title", "") or "",
        "location": (profile.get("location", "") or "").lower(),
        "country": (profile.get("country", "") or "").lower(),
        "years_of_experience": float(profile.get("years_of_experience", 0) or 0),
        "current_company": (profile.get("current_company", "") or "").lower(),
        "career_text": career_text,
        "career_parts": career_text_parts,
        "career_history_raw": history,
        "skills": norm_skills,
        "education": education,
        "signals": signals,
        # Derived
        "total_career_months": total_career_months,
        "consulting_ratio": consulting_ratio,
        "earliest_start_year": earliest_start.year if earliest_start else None,
        "days_since_active": days_since_active,
        "days_since_signup": days_since_signup,
        "short_roles": short_roles,
        "num_roles": len(history),
        "company_names_lower": company_names_lower,
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),
        "open_to_work_flag": bool(signals.get("open_to_work_flag", False)),
        "recruiter_response_rate": float(
            signals.get("recruiter_response_rate", 0) or 0
        ),
        "interview_completion_rate": float(
            signals.get("interview_completion_rate", 0) or 0
        ),
        "offer_acceptance_rate": float(
            signals.get("offer_acceptance_rate", -1) or -1
        ),
        "profile_completeness": float(
            signals.get("profile_completeness_score", 0) or 0
        ),
        "github_activity_score": float(
            signals.get("github_activity_score", -1) or -1
        ),
        "saved_by_recruiters_30d": int(signals.get("saved_by_recruiters_30d", 0) or 0),
        "search_appearance_30d": int(signals.get("search_appearance_30d", 0) or 0),
        "notice_period_days": int(signals.get("notice_period_days", 0) or 0),
        "verified_email": bool(signals.get("verified_email", False)),
        "linkedin_connected": bool(signals.get("linkedin_connected", False)),
    }

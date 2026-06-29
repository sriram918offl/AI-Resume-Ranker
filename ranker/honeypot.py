"""
Honeypot detection — the ~80 traps the organizers planted in the dataset.

The spec says we don't need to special-case them, but ranking >10% in top 100
disqualifies us. So we apply a strong subtractive penalty when a profile looks
internally inconsistent.

Detection heuristics (none of these should alone disqualify — they stack):
1. years_of_experience > sum(career_history.duration_months) / 12
2. years_of_experience > (current_year - earliest_start_year)
3. Skills marked "expert" with 0 endorsements AND 0 duration
4. Multiple skills with identical duration_months (e.g. all == 60)
5. profile_completeness > 90 but skills list < 3
6. Career history shows job at a company "founded after they started there"
   (we can't verify externally, but career_history with start_date earlier
    than what years_of_experience would allow is a flag)
7. duration_months totals wildly exceed years_of_experience * 12
"""

from __future__ import annotations

from datetime import date


def honeypot_score(features: dict, *, ref_year: int = 2026) -> tuple[float, list[str]]:
    """
    Return (penalty 0..1, reasons[]).
    Penalty 0 = clean, 1 = almost certainly a honeypot.
    """
    reasons: list[str] = []
    flags = 0

    yoe = features["years_of_experience"]
    total_months = features["total_career_months"]

    # 1. YoE vs sum of role durations
    if total_months and yoe > 0:
        yoe_months = yoe * 12
        if yoe_months > total_months * 1.5:  # claimed >1.5x actual
            reasons.append(
                f"YoE={yoe} claims {yoe_months:.0f} months but history sums to {total_months}"
            )
            flags += 2
        elif total_months > yoe_months * 1.8:  # actual >1.8x claimed (overlap impossible)
            reasons.append(
                f"history sums to {total_months} months but YoE only {yoe}"
            )
            flags += 2

    # 2. YoE vs earliest start
    earliest = features.get("earliest_start_year")
    if earliest and yoe > 0:
        max_possible = ref_year - earliest + 1
        if yoe > max_possible + 1:
            reasons.append(
                f"YoE={yoe} > years since first role ({earliest} -> {max_possible})"
            )
            flags += 2

    # 3. Expert-no-endorsement-no-duration combos
    expert_with_nothing = sum(
        1
        for s in features["skills"]
        if s["proficiency"] == "expert"
        and s["endorsements"] == 0
        and s["duration_months"] == 0
    )
    if expert_with_nothing >= 2:
        reasons.append(f"{expert_with_nothing} 'expert' skills with no endorsements or duration")
        flags += 2
    elif expert_with_nothing == 1:
        flags += 1

    # 4. Skills all-same-duration (synthetic)
    durations = [s["duration_months"] for s in features["skills"] if s["duration_months"] > 0]
    if len(durations) >= 5:
        if len(set(durations)) == 1:
            reasons.append(f"all {len(durations)} skills have identical duration={durations[0]}")
            flags += 3

    # 5. Profile complete but skills sparse
    if features["profile_completeness"] > 90 and len(features["skills"]) < 3:
        reasons.append(
            f"profile_completeness={features['profile_completeness']} but {len(features['skills'])} skills"
        )
        flags += 2

    # 6. No career history but claims YoE
    if features["num_roles"] == 0 and yoe > 0:
        reasons.append(f"YoE={yoe} but no career history")
        flags += 3

    # Penalty curve: 0 flags=0, 1 flag=0.2, 2=0.4, 3=0.7, 4+=1.0
    penalty = min(1.0, flags / 4.0)
    return penalty, reasons

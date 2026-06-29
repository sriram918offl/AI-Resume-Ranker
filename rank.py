"""
Redrob hackathon — main ranking entry point.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints (per submission_spec.docx Section 3):
  - CPU only
  - No network during ranking
  - ≤ 5 minutes runtime on 16GB RAM
  - Exactly 100 ranked rows, scores non-increasing
  - Deterministic tie-break: score desc, then candidate_id asc

Architecture:
  1. Stream candidates.jsonl one at a time
  2. Project each to a compact feature dict (ranker.features)
  3. Score with deterministic rule-based scorer (ranker.scoring)
  4. Optionally blend with cached semantic similarity (ranker.semantic, if precomputed)
  5. Top-100 by score, build reasoning, write CSV
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback
    def tqdm(x, **kw):
        return x

from ranker.features import load_candidates, project_features
from ranker.reasoning import build_reasoning
from ranker.scoring import score_candidate


def _load_semantic_cache(path: Path | None) -> dict[str, float]:
    """
    Optional: load pre-computed candidate_id -> semantic_score (0..1) from a
    JSON file produced by precompute_embeddings.py. Falls back to {} so the
    ranker is fully self-sufficient if you skip the embedding step.
    """
    if not path or not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def rank(candidates_path: Path, semantic_cache: dict[str, float]) -> list[tuple[float, str, dict, object]]:
    """
    Stream-score every candidate, return list of (score, candidate_id, features, breakdown).
    Memory note: we only keep features for ones we might write out (top 500 by score).
    """
    # Use a min-heap of size 500 — we only need top-100 but keep slack for tiebreaks
    # and post-processing. Each entry is small (~1KB).
    KEEP = 500
    keep: list[tuple[float, str, dict, object]] = []

    n_seen = 0
    for cand in load_candidates(candidates_path):
        n_seen += 1
        feats = project_features(cand)
        sem = float(semantic_cache.get(feats["candidate_id"], 0.0))
        breakdown = score_candidate(feats, semantic_score=sem)
        keep.append((breakdown.final, feats["candidate_id"], feats, breakdown))

        # Periodically trim to keep memory bounded
        if len(keep) > KEEP * 4:
            keep.sort(key=lambda x: (-x[0], x[1]))
            del keep[KEEP:]

    keep.sort(key=lambda x: (-x[0], x[1]))
    print(f"  scored {n_seen} candidates", file=sys.stderr)
    return keep[:KEEP]


def write_submission(ranked, out_path: Path) -> None:
    """
    Emit the top-100 CSV per submission_spec.docx Section 2-3.

    We round scores to 4 decimals, then RE-SORT by (rounded_score desc,
    candidate_id asc) so the displayed scores always satisfy the validator's
    tiebreak rule. Without this, two raw scores like 0.43571 and 0.43578 both
    round to 0.4358 but appear in raw-score order, failing the validator.
    """
    # Take a generous slice in case re-rounding shuffles the top-100 boundary
    top = ranked[: max(200, len(ranked))]
    if len(top) < 100:
        raise SystemExit(
            f"FATAL: only {len(top)} candidates scored — need 100. "
            "Check the input file."
        )

    # Pre-round scores, then deterministically sort by (-rounded, cid asc)
    with_rounded = [
        (round(s, 4), cid, f, b) for (s, cid, f, b) in top
    ]
    with_rounded.sort(key=lambda x: (-x[0], x[1]))

    final = with_rounded[:100]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (score, cid, feats, breakdown) in enumerate(final, start=1):
            reasoning = build_reasoning(breakdown, feats, rank=i)
            w.writerow([cid, i, f"{score:.4f}", reasoning])


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob JD.")
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to candidates.jsonl (or .jsonl.gz, or the sample JSON array).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./submission.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--semantic-cache",
        type=Path,
        default=None,
        help="Optional pre-computed semantic scores JSON (candidate_id -> 0..1).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"Loading semantic cache: {args.semantic_cache or '(none)'}", file=sys.stderr)
    sem_cache = _load_semantic_cache(args.semantic_cache)
    print(f"  {len(sem_cache)} cached scores", file=sys.stderr)

    print(f"Scoring candidates from: {args.candidates}", file=sys.stderr)
    ranked = rank(args.candidates, sem_cache)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(ranked, args.out)
    print(f"Wrote {args.out} in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
One-time pre-computation: embed every candidate's profile text and cache the
cosine similarity to the JD embedding as a JSON dict.

This step is allowed to exceed the 5-minute ranking budget (per spec §3).
At ranking time, rank.py loads the JSON and uses the cached scores — no
sentence-transformers model needs to load, no network needed.

Hardened against crashes:
  - Writes a checkpoint every CHECKPOINT_EVERY candidates
  - Resumes from existing cache if --out already exists

Short-text mode (default):
  Builds a compact ~250 char text per candidate (headline + title + top
  skills). 5-10x faster than the prior long-form version, captures the
  signal that semantic similarity is best at.

Usage:
    python precompute_embeddings.py \\
        --candidates ./candidates.jsonl \\
        --out ./semantic_cache.json

Runs on CPU only. ~10 min for 100K with all-MiniLM-L6-v2 (90 MB).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ranker.features import (
    load_candidates,
    load_candidates_from_array,
    project_features,
)
from ranker.jd_config import JD_SEED_TEXT

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128
CHECKPOINT_EVERY = 5_000


def _iter_candidates(path: Path):
    """Yield candidates from either JSON-array or JSONL files."""
    if path.open("rb").read(2)[:1] == b"[":
        for c in load_candidates_from_array(path):
            yield c
    else:
        for c in load_candidates(path):
            yield c


def _candidate_text(features: dict) -> str:
    """
    Short-text representation: headline + current title + top-8 skill names.
    Total ~250 chars. Catches the signal MiniLM is best at (lexical-semantic
    overlap with JD seed text) without the cost of full-history encoding.
    """
    headline = (features["headline_full"] or "")[:120]
    title = (features["current_title_full"] or "")[:60]
    top_skills = [s["raw_name"] or s["name"] for s in features["skills"][:8]]
    skills_blob = " ".join(top_skills)
    parts = [p for p in (headline, title, skills_blob) if p]
    return " | ".join(parts)


def _atomic_save(cache: dict, out_path: Path) -> None:
    """Save the cache atomically so a crash mid-write can't corrupt it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        dir=str(out_path.parent),
        prefix=out_path.stem + ".",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    )
    json.dump(cache, tmp)
    tmp.close()
    os.replace(tmp.name, out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("./semantic_cache.json"))
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    # Resume: load any existing cache so we skip candidates already embedded.
    cache: dict[str, float] = {}
    if args.out.exists():
        try:
            cache = json.loads(args.out.read_text(encoding="utf-8"))
            print(f"Resumed from {args.out}: {len(cache)} already embedded", file=sys.stderr)
        except Exception:
            cache = {}

    print(f"Loading model: {args.model}", file=sys.stderr)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device="cpu")

    # Embed the JD once
    jd_vec = model.encode([JD_SEED_TEXT], normalize_embeddings=True)[0]
    print(f"JD embedding shape: {jd_vec.shape}", file=sys.stderr)

    t0 = time.perf_counter()
    batch_texts: list[str] = []
    batch_ids: list[str] = []
    n_seen = 0
    n_skipped = 0
    n_since_checkpoint = 0

    def _flush() -> None:
        nonlocal batch_texts, batch_ids
        if not batch_texts:
            return
        vecs = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = vecs @ jd_vec
        scores = np.clip((scores + 1) / 2, 0.0, 1.0)
        for cid, sc in zip(batch_ids, scores):
            cache[cid] = float(sc)
        batch_texts = []
        batch_ids = []

    def _checkpoint() -> None:
        _atomic_save(cache, args.out)
        elapsed = time.perf_counter() - t0
        rate = (n_seen - n_skipped) / max(elapsed, 0.1)
        print(
            f"  checkpoint: {len(cache)} cached, {rate:.0f}/s, {elapsed:.0f}s elapsed",
            file=sys.stderr,
        )

    for cand in _iter_candidates(args.candidates):
        n_seen += 1
        cid = cand.get("candidate_id")
        if cid in cache:
            n_skipped += 1
            continue
        feats = project_features(cand)
        batch_texts.append(_candidate_text(feats))
        batch_ids.append(cid)
        n_since_checkpoint += 1
        if len(batch_texts) >= BATCH_SIZE * 4:
            _flush()
        if n_since_checkpoint >= CHECKPOINT_EVERY:
            _flush()
            _checkpoint()
            n_since_checkpoint = 0

    _flush()
    _atomic_save(cache, args.out)
    elapsed = time.perf_counter() - t0
    print(
        f"Done. {len(cache)} embeddings in {elapsed:.0f}s (skipped {n_skipped} resumed)",
        file=sys.stderr,
    )
    if cache:
        arr = np.array(list(cache.values()))
        print(
            f"Similarity stats: min={arr.min():.3f} mean={arr.mean():.3f} "
            f"max={arr.max():.3f} std={arr.std():.3f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

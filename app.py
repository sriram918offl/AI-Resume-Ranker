"""
HuggingFace Spaces sandbox app — Gradio 5.x.

Satisfies the spec's mandatory sandbox link requirement (submission_spec §10.5).
Accepts a JSON array or JSONL upload, runs the ranker end-to-end, displays the
ranked output and offers the CSV for download. CPU only, no network.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import time
from pathlib import Path

import gradio as gr

from ranker.features import load_candidates, load_candidates_from_array, project_features
from ranker.reasoning import build_reasoning
from ranker.scoring import score_candidate

SEMANTIC_CACHE_PATH = Path(__file__).parent / "semantic_cache.json"


def _load_cache() -> dict[str, float]:
    if SEMANTIC_CACHE_PATH.exists():
        return json.loads(SEMANTIC_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


SEMANTIC_CACHE = _load_cache()


def _candidates_from_upload(filepath: str) -> list[dict]:
    """Gradio 5 with type='filepath' passes the file path as a string."""
    path = Path(filepath)
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    if path.suffix.lower() == ".jsonl" or "\n" in text[:1000]:
        return list(load_candidates(path))
    # Fall back: try array first, then jsonl
    try:
        return load_candidates_from_array(path)
    except json.JSONDecodeError:
        return list(load_candidates(path))


def rank_uploaded(filepath):
    if not filepath:
        return "**Please upload a candidates file (.json or .jsonl).**", "", None

    t0 = time.perf_counter()
    try:
        cands = _candidates_from_upload(filepath)
    except Exception as e:
        return f"**Failed to parse upload:** `{e}`", "", None

    if len(cands) == 0:
        return "**Upload contained no candidates.**", "", None

    scored = []
    for c in cands:
        try:
            feats = project_features(c)
            sem = float(SEMANTIC_CACHE.get(feats["candidate_id"], 0.0))
            breakdown = score_candidate(feats, semantic_score=sem)
            scored.append((breakdown.final, feats["candidate_id"], feats, breakdown))
        except Exception as e:
            return f"**Scoring failed on a candidate:** `{e}`", "", None

    # Round + tiebreak: (-rounded score, candidate_id asc)
    scored = [(round(s, 4), cid, f, b) for (s, cid, f, b) in scored]
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[: min(100, len(scored))]

    # Build CSV
    csv_buf = io.StringIO()
    w = csv.writer(csv_buf, quoting=csv.QUOTE_MINIMAL)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    rows_md: list[str] = []
    for i, (sc, cid, feats, b) in enumerate(top, start=1):
        reasoning = build_reasoning(b, feats, rank=i)
        w.writerow([cid, i, f"{sc:.4f}", reasoning])
        title_short = (feats.get("current_title_full") or "—").replace("|", "/")[:30]
        reasoning_short = reasoning.replace("|", "/").replace("\n", " ")[:120]
        rows_md.append(
            f"| {i} | `{cid}` | {sc:.4f} | {title_short} | {feats['years_of_experience']:.1f} | {reasoning_short} |"
        )

    elapsed = time.perf_counter() - t0
    cache_hits = sum(
        1 for _, cid, _, _ in scored if cid in SEMANTIC_CACHE
    )
    summary = (
        f"**Ranked {len(cands)} candidates in {elapsed:.2f}s** "
        f"({cache_hits}/{len(cands)} semantic-cache hits)."
    )
    table_md = (
        "| Rank | Candidate ID | Score | Title | YoE | Reasoning |\n"
        "|---:|:---|---:|:---|---:|:---|\n"
    ) + "\n".join(rows_md[:30])
    if len(top) > 30:
        table_md += f"\n\n*(showing top 30 of {len(top)})*"

    # Write CSV to a temp file for download
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        encoding="utf-8",
        newline="",
    )
    tmp.write(csv_buf.getvalue())
    tmp.close()
    return summary, table_md, tmp.name


DESCRIPTION = """
# Redrob Ranker — Sandbox

Reproducible CPU-only ranker for the **Intelligent Candidate Discovery & Ranking
Challenge**. No network or LLM calls during ranking.

**Approach:** rule-based multi-component scorer (title+career evidence, skill
clusters with trust multiplier, experience bell-curve, JD-derived alignment,
location, behavioral modifier) + cached sentence-transformers similarity, with
honeypot detection and JD-specific anti-patterns (consulting-only,
framework-enthusiast, title-chaser).

Upload `sample_candidates.json` (50 candidates) from the hackathon bundle, or
any small `.json`/`.jsonl` sample. The ranker will produce the top-100 (or all
of them, whichever is smaller) along with per-candidate reasoning.
"""


with gr.Blocks(title="Redrob Ranker — Sandbox", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        file_input = gr.File(
            label="Upload candidates (.json or .jsonl)",
            file_types=[".json", ".jsonl"],
            type="filepath",
        )
        run_btn = gr.Button("Rank candidates", variant="primary", size="lg")

    summary_out = gr.Markdown()
    table_out = gr.Markdown()
    csv_out = gr.File(label="Download ranked CSV", interactive=False)

    run_btn.click(
        fn=rank_uploaded,
        inputs=[file_input],
        outputs=[summary_out, table_out, csv_out],
        api_name="rank",
    )

if __name__ == "__main__":
    demo.queue().launch()

"""
HuggingFace Spaces sandbox app — satisfies the spec's mandatory sandbox link
(submission_spec.docx §10.5). Accepts a small JSONL sample, runs the ranker
end-to-end, displays the ranked CSV.

Deployed at: https://huggingface.co/spaces/<you>/redrob-ranker

Runs entirely on CPU. No network during ranking.
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


def _candidates_from_upload(file_obj) -> list[dict]:
    """Accept either a JSON array or JSONL upload."""
    path = Path(file_obj.name) if hasattr(file_obj, "name") else Path(str(file_obj))
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return list(load_candidates(path))


def rank_uploaded(file_obj):
    if file_obj is None:
        return "Please upload a candidates file (.json array or .jsonl).", "", ""

    t0 = time.perf_counter()
    try:
        cands = _candidates_from_upload(file_obj)
    except Exception as e:
        return f"Failed to parse upload: {e}", "", ""

    if len(cands) == 0:
        return "Upload contained no candidates.", "", ""

    scored = []
    for c in cands:
        feats = project_features(c)
        sem = float(SEMANTIC_CACHE.get(feats["candidate_id"], 0.0))
        breakdown = score_candidate(feats, semantic_score=sem)
        scored.append((breakdown.final, feats["candidate_id"], feats, breakdown))

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
        rows_md.append(
            f"| {i} | `{cid}` | {sc:.4f} | {feats['current_title_full'] or '—'} | "
            f"{feats['years_of_experience']:.1f} yrs | {reasoning} |"
        )

    elapsed = time.perf_counter() - t0
    summary = (
        f"**Ranked {len(cands)} candidates in {elapsed:.2f}s** "
        f"({'with' if SEMANTIC_CACHE else 'WITHOUT'} semantic-cache hits)."
    )
    table = (
        "| Rank | Candidate ID | Score | Title | YoE | Reasoning |\n"
        "|---:|:---|---:|:---|---:|:---|\n"
    ) + "\n".join(rows_md[:30])
    if len(top) > 30:
        table += f"\n\n*(showing top 30 of {len(top)})*"

    # Write CSV to a temp file for download
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    )
    tmp.write(csv_buf.getvalue())
    tmp.close()
    return summary, table, tmp.name


SAMPLE_QUESTION = (
    "Upload `sample_candidates.json` (50 candidates) from the hackathon bundle, "
    "or any small `.jsonl` sample. The ranker will produce the top-100 (or all "
    "of them, whichever is smaller) along with reasoning."
)


with gr.Blocks(title="Redrob Ranker — Sandbox", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Redrob Ranker — Sandbox\n"
        "Reproducible CPU-only ranker for the Intelligent Candidate Discovery & "
        "Ranking Challenge. **No network or LLM calls during ranking.**\n\n"
        "**Approach in one sentence:** rule-based multi-component scorer "
        "(title+career evidence, skill clusters with trust multiplier, "
        "experience bell-curve, JD-derived alignment, location, behavioral "
        "modifier) + cached sentence-transformers similarity, with honeypot "
        "detection and JD-specific anti-patterns (consulting-only, "
        "framework-enthusiast, title-chaser).\n\n"
        f"{SAMPLE_QUESTION}"
    )
    with gr.Row():
        file_input = gr.File(
            label="Upload candidates (.json or .jsonl)",
            file_types=[".json", ".jsonl"],
        )
        run_btn = gr.Button("Rank", variant="primary")
    summary = gr.Markdown()
    table = gr.Markdown()
    csv_file = gr.File(label="Download ranked CSV")
    run_btn.click(rank_uploaded, inputs=[file_input], outputs=[summary, table, csv_file])

if __name__ == "__main__":
    demo.launch()

---
title: AI Resume Ranker
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: CPU-only ranker for the Redrob Senior AI Engineer JD
---

# Redrob Ranker — Senior AI Engineer JD

A CPU-only, no-network ranker for the [Intelligent Candidate Discovery & Ranking Challenge](https://redrob.ai/hackathon). Ranks 100,000 candidates against the released Senior AI Engineer JD in **~17 seconds** on a 16 GB CPU machine.

## Approach in one paragraph

Multi-component deterministic scorer. Each candidate gets a weighted sum of: **title+career evidence** (28%), **skill cluster coverage with a proficiency × endorsements × duration trust multiplier** (22%), **experience bell curve around the JD's 6-8 ideal** (13%), **JD-derived alignment** including anti-patterns the JD explicitly names — consulting-only careers, framework-enthusiast LangChain demos, title-chasers, pure-research backgrounds (17%), **location** with relocation awareness (5%), and **cached sentence-transformers semantic similarity** (15%). The sum is then multiplied by a **behavioral modifier** (0.4–1.0) combining recruiter-response rate, recency, profile completeness, and interview-completion rate. Honeypot heuristics (impossible YoE vs duration sum, expert-with-no-endorsements clusters, synthetic uniform durations) apply a subtractive penalty.

## Reproduce

```bash
# 1. Install (~2 min)
pip install -r requirements.txt

# 2. Pre-compute candidate embeddings (one-time, ~3-5 min on CPU)
python precompute_embeddings.py \
  --candidates ./candidates.jsonl \
  --out ./semantic_cache.json

# 3. Rank (the deliverable — runs in ~17 seconds)
python rank.py \
  --candidates ./candidates.jsonl \
  --out ./submission.csv \
  --semantic-cache ./semantic_cache.json
```

Or without semantic similarity (slightly weaker, but still valid):

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

## What gets shipped where

| File | What |
|---|---|
| `rank.py` | Main entry. Streams JSONL, scores, writes top-100 CSV. **No network. No LLMs. CPU only.** |
| `precompute_embeddings.py` | One-time embedding generator (sentence-transformers all-MiniLM-L6-v2). Allowed to exceed 5 min per spec §3. |
| `app.py` | Gradio sandbox UI (deployed to HuggingFace Spaces). |
| `ranker/jd_config.py` | The JD compiled into structured features (title terms, skill clusters, consulting-firm names, etc). |
| `ranker/features.py` | JSONL streaming + per-candidate feature projection. |
| `ranker/scoring.py` | The scoring engine. Every weight is named and justified. |
| `ranker/honeypot.py` | Heuristic honeypot detection. |
| `ranker/reasoning.py` | Per-candidate reasoning generator. No templates that just insert names. |
| `submission_metadata.yaml` | Portal metadata. |

## Compute constraints — how we honour them

| Constraint (spec §3) | How we comply |
|---|---|
| ≤ 5 minutes runtime | Pure dict/string scoring + numpy on cached embeddings → **17s for 100K** |
| CPU only | sentence-transformers `device="cpu"`; no GPU code paths |
| No network during ranking | `rank.py` only loads local JSON cache + JSONL. No HTTP. No LLM SDK imports. |
| 16 GB RAM | Streaming JSONL parse; top-K heap with size 500; semantic cache ~5 MB |
| Exactly 100 rows | Hard-coded `top[:100]` after re-sorting on rounded score |
| Score non-increasing, candidate_id asc tiebreak | Final sort is on `(-rounded_score, candidate_id)` |

## What the ranker explicitly looks for (from the JD)

**Positive signals** — title family ("AI Engineer", "ML Engineer", "Applied Scientist", "Recommendation Systems Engineer", etc.); skills cluster coverage with high endorsement count and 24+ months of usage; career descriptions mentioning ranking/retrieval/embeddings/RAG/recommendation systems in production; 5-9 years experience (peak at 6-8); India location (Pune/Noida preferred); active on Redrob.

**Negative signals** — non-AI title with AI-buzzword skills stuffed (keyword-stuffer trap); >50% career at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc); pure-research title with no production deployment language; computer vision/speech/robotics dominant without NLP exposure; <18-month role tenure averaged across many roles (title-chaser); LangChain-only with no underlying ML.

**Honeypots** — caught via internal-consistency checks: years_of_experience exceeding the sum of role durations; "expert" proficiency with 0 endorsements AND 0 duration; multiple skills with identical synthetic durations; profile_completeness=100 with sparse skill list.

## What we explicitly do NOT use

- Hosted LLM APIs (OpenAI, Anthropic, Gemini, etc.) — banned by spec.
- GPU — banned by spec.
- The candidate's name, location-derived demographics — not scored on.
- School prestige — JD explicitly says it doesn't weight this.

## Sample output (top 5 from full 100K)

```
1  CAND_0077337  0.7808  Staff Machine Learning Engineer, 7.0 yrs, LLMs (expert, 53 endorsements); active on Redrob (response 95%).
2  CAND_0079387  0.7444  AI Engineer, 6.9 yrs, Sentence Transformers (expert, 45 endorsements); active on Redrob (response 81%).
3  CAND_0002025  0.7411  Senior AI Engineer, 5.9 yrs, NLP (expert, 51 endorsements); active on Redrob (response 80%).
4  CAND_0081846  0.7268  Lead AI Engineer, 6.7 yrs, Learning to Rank (expert, 54 endorsements); active on Redrob (response 73%).
5  CAND_0008425  0.7229  Senior NLP Engineer, 7.8 yrs, Sentence Transformers (expert, 56 endorsements); but long notice period (90d).
```

## Sandbox

A live sandbox is deployed to HuggingFace Spaces: **[link in `submission_metadata.yaml`]**.
Upload a small `.json` array or `.jsonl` file (≤ 100 candidates), and the app returns the ranked CSV.

## License

MIT — see `LICENSE`.

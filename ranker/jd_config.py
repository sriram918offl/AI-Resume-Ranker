"""
Structured features extracted (by hand) from the hackathon's job_description.docx.

This is the JD compiled into a feature table. We do this *offline*, not at
ranking time — the JD is fixed for the hackathon, so we encode it once.

Reading the JD carefully:
- Role: Senior AI Engineer, founding team, applied ML/retrieval/ranking
- Ideal: 6-8 yrs, 4-5 in applied ML at product cos, shipped a search/ranking system
- Hard 'do not wants': consulting-only career, vision/speech-only, pure research
- Soft penalties: title-chasers, framework-enthusiasts (LangChain demos only)
- Location: Noida/Pune preferred; Hyderabad/Mumbai/Delhi NCR welcome; rest case-by-case
- Behavioral: low engagement = effectively unavailable
"""

from __future__ import annotations

# ---------- Title / career signals ----------------------------------------

PRIMARY_TITLE_TERMS: list[str] = [
    # Direct hits — current_title containing any of these is a strong positive
    "ai engineer",
    "ml engineer",
    "machine learning engineer",
    "applied scientist",
    "applied ai",
    "applied ml",
    "search engineer",
    "ranking engineer",
    "recommendation engineer",
    "nlp engineer",
    "data scientist",  # context-dependent; verified by skills
    "senior software engineer",  # context-dependent; verified by skills
    "research engineer",  # ML research engineer is borderline
    "ml infrastructure",
    "search scientist",
    "recsys",
]

# Career-history descriptions matching these phrases hint at production
# ranking/retrieval work — exactly what the JD wants.
CAREER_SIGNAL_PHRASES: list[str] = [
    "ranking",
    "retrieval",
    "recommendation",
    "search",
    "embedding",
    "vector search",
    "semantic search",
    "bm25",
    "ndcg",
    "mrr",
    "fine-tun",
    "rag",
    "knowledge base",
    "ann ",
    " ann,",
    "hybrid search",
    "personalization",
    "personalisation",
    "matching",
    "deployed",
    "shipped",
    "production",
    "scale",
    "a/b test",
]

# Penalty titles — high keyword match here means likely keyword stuffer.
KEYWORD_STUFFER_TITLES: list[str] = [
    "marketing manager",
    "sales executive",
    "content writer",
    "graphic designer",
    "hr manager",
    "accountant",
    "civil engineer",
    "mechanical engineer",
    "operations manager",
    "project manager",
    "business analyst",
    "customer support",
]


# ---------- Skill clusters ------------------------------------------------

REQUIRED_SKILL_CLUSTERS: dict[str, list[str]] = {
    "embedding_retrieval": [
        "sentence-transformers",
        "sentence transformers",
        "openai embeddings",
        "bge",
        "e5",
        "embedding",
        "embeddings",
        "vector search",
        "bm25",
        "hybrid search",
        "semantic search",
        "dense retrieval",
        "sparse retrieval",
    ],
    "vector_db": [
        "pinecone",
        "weaviate",
        "qdrant",
        "milvus",
        "faiss",
        "opensearch",
        "elasticsearch",
        "pgvector",
        "chroma",
        "chromadb",
        "vespa",
    ],
    "evaluation": [
        "ndcg",
        "mrr",
        "map",
        "mean reciprocal rank",
        "precision at",
        "recall at",
        "a/b testing",
        "ab testing",
        "offline evaluation",
        "online evaluation",
        "learning to rank",
        "ltr",
    ],
    "nlp_core": [
        "nlp",
        "natural language processing",
        "transformers",
        "bert",
        "huggingface",
        "hugging face",
        "fine-tuning llms",
        "fine-tuning",
        "tokenization",
        "tokenisation",
        "language models",
        "llm",
    ],
    "python_engineering": [
        "python",
        "fastapi",
        "flask",
        "django",
        "pytest",
        "asyncio",
        "rest api",
    ],
}

NICE_SKILL_CLUSTERS: dict[str, list[str]] = {
    "fine_tuning_methods": [
        "lora",
        "qlora",
        "peft",
        "rlhf",
        "dpo",
        "instruction tuning",
    ],
    "classical_ml": [
        "xgboost",
        "lightgbm",
        "catboost",
        "random forest",
        "gradient boosting",
        "learning-to-rank",
        "ltr",
    ],
    "distributed_systems": [
        "ray",
        "spark",
        "kafka",
        "kubernetes",
        "k8s",
        "airflow",
        "distributed",
    ],
    "domain_recsys_hrtech": [
        "recsys",
        "recommendation",
        "marketplace",
        "two-sided",
        "matching",
        "talent",
        "recruiting",
    ],
}

# Penalty skill clusters — if the candidate's career is dominated by these
# without NLP/IR signals, JD explicitly does not want them.
OTHER_DOMAIN_CLUSTERS: dict[str, list[str]] = {
    "vision_only": [
        "computer vision",
        "opencv",
        "yolo",
        "object detection",
        "image classification",
        "segmentation",
        "ocr",
        "gans",
    ],
    "speech_only": [
        "speech recognition",
        "tts",
        "asr",
        "wav2vec",
        "whisper",
        "voice",
    ],
    "robotics_only": ["robotics", "ros", "slam"],
}


# ---------- Career penalties --------------------------------------------

CONSULTING_FIRMS: set[str] = {
    # Lowercase strings — match against company name lower
    "tcs", "tata consultancy services",
    "infosys", "infosys ltd", "infosys limited",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl", "hcl technologies",
    "tech mahindra",
    "mindtree",
    "ltimindtree",
    "deloitte", "ey", "pwc", "kpmg",
}

RESEARCH_ONLY_TERMS: list[str] = [
    "research scientist",
    "research engineer",  # only penalty when paired with no production signal
    "phd researcher",
    "post-doctoral",
    "postdoc",
    "academic",
    "professor",
    "lecturer",
]


# ---------- Location ---------------------------------------------------

PRIMARY_CITIES: set[str] = {"pune", "noida"}
BONUS_CITIES: set[str] = {
    "hyderabad",
    "mumbai",
    "delhi",
    "new delhi",
    "gurgaon",
    "gurugram",
    "delhi ncr",
    "bangalore",
    "bengaluru",
}
INDIA_OK: set[str] = {"india"}


# ---------- Experience -------------------------------------------------

# JD says 5-9 years stated, ideal 6-8. Build a soft bell.
EXP_PEAK_RANGE: tuple[float, float] = (6.0, 8.0)
EXP_ACCEPTABLE: tuple[float, float] = (4.0, 10.0)
EXP_HARD_RANGE: tuple[float, float] = (3.0, 12.0)


# ---------- JD seed text for semantic embedding -----------------------

# Used by the optional embedding-similarity layer. Concise on purpose.
JD_SEED_TEXT: str = (
    "Senior AI Engineer for a Series A AI-native talent intelligence platform. "
    "Owns the intelligence layer: ranking, retrieval, matching. "
    "Five to nine years of applied ML at product companies. "
    "Production embedding-based retrieval (sentence-transformers, BGE, E5). "
    "Hybrid search with vector databases (Pinecone, Weaviate, Qdrant, Milvus, FAISS). "
    "Strong Python. Ranking evaluation: NDCG, MRR, MAP, A/B testing. "
    "Has shipped end-to-end ranking, search, or recommendation systems to real users. "
    "Pragmatic shipper-leaning attitude over researcher. "
    "Located in or willing to relocate to Pune or Noida. "
    "NOT a pure researcher, NOT consulting-firm-only, NOT framework enthusiast with only "
    "LangChain demos, NOT vision/speech without NLP exposure."
)

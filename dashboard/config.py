"""Central paths & constants — single source of truth."""
from pathlib import Path

CAPPRO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = CAPPRO_ROOT / "data_collection" / "data"
EXPORT_DIR = Path(__file__).resolve().parent / "exports"     # cached embeddings/citations

DOMAINS = {
    "NLP (ACL/arXiv)": {
        "dir": DATA_ROOT / "nlp",
        "patterns": ["acl_anthology_{year}.json", "arxiv_cscl_{year}.json"],
        "years": range(2018, 2025),
    },
    "COVID-19 (CS-adjacent)": {
        "dir": DATA_ROOT / "covid",
        # prefer sampled files when both exist; fall back to full
        "patterns": ["covid_harvested_{year}_sampled.json",
                     "covid_harvested_{year}.json",
                     "cord19_filtered_{year}.json"],
        "years": range(2019, 2025),
    },
}

YEARS_ALL = list(range(2018, 2025))
T_LATEST = 2024          # scoring time t
T_PREV = 2023            # t−1 for Δsim
VEL_WINDOW = (2022, 2024)  # citation velocity window
ALPHA_CHOICES = [0.3, 0.5, 0.7]
EMB_DIM = 128

# ── Real pipeline outputs (Components 5 & 6) ───────────────────────────
# The dashboard reads ranked gaps from the real pipeline rather than
# re-deriving them from the synthetic demo embeddings in exports/.
# These are produced by:
#   component5/04_fuse_and_rank.py  ->  COMP5_GAPS_DIR
#   component6/03_rerank.py         ->  COMP6_RERANKED_DIR
COMP5_GAPS_DIR = CAPPRO_ROOT / "component5" / "output" / "gaps"
COMP6_RERANKED_DIR = CAPPRO_ROOT / "component6" / "output" / "reranked"
COMP6_VELOCITY_DIR = CAPPRO_ROOT / "component6" / "output"

# Dashboard domain label -> pipeline domain short name used in gap filenames
DOMAIN_SHORT = {
    "NLP (ACL/arXiv)": "NLP",
    "COVID-19 (CS-adjacent)": "COVID",
}

# Years each domain actually has gap files for
DOMAIN_GAP_YEARS = {
    "NLP (ACL/arXiv)": list(range(2018, 2025)),
    "COVID-19 (CS-adjacent)": list(range(2019, 2025)),
}

# Alphas the real pipeline was run with (component5/04 ALPHAS)
REAL_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# ── Component 3: retrospective validation ──────────────────────────────
# Cutoff year: gaps are scored using data from <= this year only,
# then checked against co-occurrences in publications from
# (cutoff+1)..T_LATEST to measure how many predicted gaps "materialize".
VALIDATION_CUTOFF = 2021

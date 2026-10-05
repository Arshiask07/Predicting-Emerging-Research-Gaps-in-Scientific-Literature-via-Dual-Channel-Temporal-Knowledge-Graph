"""Reads the actual data_collection JSON outputs.

Expected paper-record schema (tolerant of missing keys):
{
  "paper_id": "...",        # or "id" / falls back to title hash
  "title": "...",
  "abstract": "...",
  "year": 2021,
  "authors": [...],         # optional
  "citation_count": 42      # optional; else fetched from exports/citations.csv
}
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

import config


def _norm_record(rec: dict, default_year: int) -> dict | None:
    abstract = rec.get("abstract") or rec.get("abstractText") or ""
    if len(str(abstract).strip()) < 40:
        return None                      # skip records without usable abstracts
    pid = rec.get("paper_id") or rec.get("id") or hashlib.md5(
        str(rec.get("title", "")).encode()).hexdigest()[:16]
    return {
        "paper_id": pid,
        "title": rec.get("title", ""),
        "abstract": abstract,
        "year": int(rec.get("year") or default_year),
        "citation_count": int(rec.get("citation_count")
                              or rec.get("citationCount") or 0),
    }


def load_papers(domain_key: str) -> pd.DataFrame:
    """Merge all matching per-year JSON files for a domain into one DataFrame."""
    cfg = config.DOMAINS[domain_key]
    rows = []
    for year in cfg["years"]:
        for pattern in cfg["patterns"]:
            path = cfg["dir"] / pattern.format(year=year)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):                       # {"papers": [...]} wrapper
                data = data.get("papers", [])
            for rec in data:
                r = _norm_record(rec, year)
                if r:
                    rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset="paper_id").reset_index(drop=True)
    return df


def load_citation_history(domain_key: str) -> pd.DataFrame:
    """entity_id, year, citations — produced by Component 6 / S2 dump.
    Falls back to paper-level citation counts mapped through entity links."""
    p = config.EXPORT_DIR / f"citations_{domain_key.split()[0].lower()}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_entity_embeddings(domain_key: str):
    """Returns {channel: {year: DataFrame(entity_id → EMB_DIM)}}.
    Files written by Component 5 (or make_demo_embeddings.py):
        exports/{channel}_{domain}_{year}.csv
    """
    short = domain_key.split()[0].lower()
    out = {}
    for channel in ("node2vec", "specter2"):
        frames = {}
        for y in config.YEARS_ALL:
            p = config.EXPORT_DIR / f"{channel}_{short}_{y}.csv"
            if p.exists():
                frames[y] = pd.read_csv(p).set_index("entity_id")
        out[channel] = frames
    return out


def load_entities(domain_key: str) -> pd.DataFrame | None:
    """Component 4 output: entity_id, label, type, first_year."""
    p = config.EXPORT_DIR / f"entities_{domain_key.split()[0].lower()}.csv"
    return pd.read_csv(p) if p.exists() else None


def load_edges(domain_key: str) -> pd.DataFrame | None:
    """Component 4 output triples: source, relation, target, first_observed."""
    p = config.EXPORT_DIR / f"edges_{domain_key.split()[0].lower()}.csv"
    return pd.read_csv(p) if p.exists() else None


# ── Real pipeline gaps (Components 5 & 6) ──────────────────────────────────
# These are the authoritative ranked gap lists: real Node2Vec + SPECTER
# embeddings, FAISS retrieval, real entity names. The demo embeddings in
# exports/ are synthetic (np.random.randn) and are NOT used here.

GAP_COLUMNS = [
    "entity_a", "entity_b", "surface_form_a", "surface_form_b",
    "structural_score", "semantic_score", "fused_score", "alpha", "year",
]


def _alpha_str(alpha: float) -> str:
    """Match the filename convention in component5/04: 0.5 -> '0.5', 0.0 -> '0'."""
    return f"{alpha:.2f}".rstrip("0").rstrip(".")


def available_gap_years(domain_key: str) -> list[int]:
    """Years that actually have a gap file on disk for this domain + alpha."""
    short = config.DOMAIN_SHORT.get(domain_key)
    if not short:
        return []
    out = []
    for y in config.DOMAIN_GAP_YEARS.get(domain_key, []):
        if _gap_path(short, y, 0.5, reranked=False).exists():
            out.append(y)
    return out


def _gap_path(short: str, year: int, alpha: float, reranked: bool) -> Path:
    base = config.COMP6_RERANKED_DIR if reranked else config.COMP5_GAPS_DIR
    return base / f"{short}_{year}_alpha{_alpha_str(alpha)}.json"


def load_real_gaps(domain_key: str, year: int, alpha: float,
                   reranked: bool = False) -> pd.DataFrame | None:
    """Load ranked candidate gaps from the real pipeline (Component 5 or 6).

    Component 5 files carry the FAISS fused ranking. Component 6 files add
    vel_u / vel_v / priority from mention- or citation-velocity re-ranking.
    Returns None if the file does not exist.
    """
    short = config.DOMAIN_SHORT.get(domain_key)
    if not short:
        return None
    path = _gap_path(short, year, alpha, reranked)
    if not path.exists():
        return None
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not records:
        return pd.DataFrame(columns=GAP_COLUMNS)
    df = pd.DataFrame(records)
    for col in GAP_COLUMNS:
        if col not in df.columns:
            df[col] = None
    # Stable display ordering: fused ranking for C5, priority for C6.
    sort_col = "priority" if (reranked and "priority" in df.columns) else "fused_score"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df


def load_entity_velocity(domain_key: str) -> tuple[pd.DataFrame | None, str]:
    """canonical_id -> velocity, from component6/02_entity_velocity.py.

    The file nests the actual values under a "velocities" key alongside
    metadata (vel_window, path). Returns (DataFrame, path_label) where
    path_label is 'mention' or 'api'; (None, 'unknown') if unavailable.
    """
    short = config.DOMAIN_SHORT.get(domain_key)
    if not short:
        return None, "unknown"
    path = config.COMP6_VELOCITY_DIR / f"entity_velocity_{short}.json"
    if not path.exists():
        return None, "unknown"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, "unknown"
    if not isinstance(data, dict):
        return None, "unknown"

    label = str(data.get("path", "")).lower()
    path_label = "api" if "api" in label and "fallback" not in label else "mention"

    values = data.get("velocities")
    if not isinstance(values, dict):
        # Older/flat shape: {canonical_id: velocity}
        values = {k: v for k, v in data.items() if isinstance(v, (int, float))}
    rows = [{"canonical_id": k, "velocity": v} for k, v in values.items()
            if isinstance(v, (int, float))]
    return (pd.DataFrame(rows) if rows else None), path_label


def citation_path_for(domain_key: str) -> str:
    """Which citation source Component 6 actually used: 'api' or 'mention'.

    component6/01_fetch_citations.py writes a {domain}_FALLBACK.txt flag into
    output/citations/ whenever the Semantic Scholar API is unreachable.
    """
    short = config.DOMAIN_SHORT.get(domain_key)
    if not short:
        return "unknown"
    flag = config.COMP6_VELOCITY_DIR / "citations" / f"{short}_FALLBACK.txt"
    return "mention" if flag.exists() else "api"

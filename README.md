# Temporal Knowledge Graph Embedding for Emerging Research Gap Detection

A 6-component ML/NLP pipeline that discovers emerging research gaps in scientific literature by building temporal knowledge graphs from paper corpora, extracting entities and relations with SciBERT, computing structural + semantic embeddings, and ranking candidate gaps by fused similarity.

**Domains:** NLP (ACL Anthology + arXiv, 2018–2024, 1400 papers) · COVID-19 (multi-source, 2019–2024, 840 papers)

---

## Architecture

```
capstone_sep_15/
├── data_collection/        # Component 1 — data ingestion (6 collector scripts)
│   ├── data/
│   │   ├── nlp/            # Raw + sampled NLP corpus JSONs
│   │   └── covid/          # Raw + sampled COVID corpus JSONs
│   └── scripts/            # 01–06 collector scripts
│
├── component2_entity_relation_extraction/   # Component 2 (original SciBERT pipeline)
│   ├── scripts/            # NER + relation extraction, baseline, validation
│   ├── output/             # Consolidated {NLP,COVID}_entities.json, _relations.json
│   ├── reports/            # Component 2 report
│   └── logs/
│
├── com2_using_3models/     # Component 2 (multi-model evolution: SciBERT + RoBERTa + PubMedBERT)
│   ├── scripts/            # Multi-model extraction pipeline
│   ├── output/             # Per-model subdirectories + consolidated outputs
│   ├── reports/            # Model comparison report
│   └── logs/
│
├── component3/             # Component 3 — retrospective validation
├── component4/             # Component 4 — knowledge graph builder + HTML viewer
├── component5/             # Component 5 — structural + semantic embeddings, gap ranking, evaluation
├── component6/             # Component 6 — citation velocity + reranking
│
├── dashboard/              # Streamlit interactive explorer
│   ├── app.py              # Main Streamlit app
│   ├── config.py           # Shared paths & constants
│   ├── data_loader.py      # Data loading utilities
│   ├── gap_engine.py       # Gap ranking logic
│   ├── make_demo_embeddings.py
│   └── exports/            # Generated CSV exports (gitignored)
│
├── archive/                # Archived earlier outputs
├── emergent_landing.html   # Single-file Japandi landing page
├── timeline.html           # Timeline visualization
└── docs/                   # (future — currently top-level .md files serve this role)
```

## Components

| # | Directory | Purpose | Key outputs |
|---|-----------|---------|-------------|
| 1 | `data_collection/` | Collect papers from ACL Anthology, Semantic Scholar, arXiv, CORD-19, PubMed, Europe PMC, OpenAlex | `data/nlp/*.json`, `data/covid/*.json` (1400 NLP + 840 COVID) |
| 2 | `component2_entity_relation_extraction/` | SciBERT NER + relation extraction over sampled corpus | `{domain}_entities.json` (28k NLP / 12k COVID), `{domain}_relations.json` (23k NLP / 8k COVID) |
| 2b | `com2_using_3models/` | Same task with 3 transformer backbones for comparison | Per-model outputs + comparison report |
| 3 | `component3/` | Retrospective validation: predict gaps pre-cutoff, check post-cutoff co-occurrence | `output/validation_*.json/.md` |
| 4 | `component4/` | Build directed NetworkX KG per year + interactive HTML viewer | `graph_viewer.html`, `graph_summary.json` |
| 5 | `component5/` | Node2Vec structural embeddings + SPECTER2 semantic embeddings → fuse (alpha) → rank gaps via FAISS → evaluate | `output/gaps/`, `output/embeddings/`, ablation reports, `top_gaps_*_latest.json` |
| 6 | `component6/` | Fetch citation counts from Semantic Scholar API → entity velocity → rerank gaps | `output/citations/`, `output/reranked/`, `top_velocity_gaps_*_latest.json` |
| D | `dashboard/` | Streamlit app to explore gaps, graph timeline, validation results | `app.py` + supporting modules |

## Technologies

- **Python 3.12** (Anaconda distribution recommended; see Setup)
- **PyTorch + Transformers** (SciBERT, RoBERTa, PubMedBERT for NER/relation extraction)
- **NetworkX** (temporal graph slicing)
- **Node2Vec** (structural embeddings)
- **Sentence Transformers / SPECTER2** (semantic embeddings)
- **FAISS** (gap ranking via cosine similarity)
- **spaCy + rapidfuzz** (baseline extraction, entity normalization)
- **Streamlit + Pyvis + Plotly** (interactive dashboard)
- **Pandas + NumPy + SciPy** (data processing)

## Setup

### Prerequisites

- Python 3.12 with Conda/Anaconda (the project was developed with `/opt/anaconda3/bin/python3`)
- Internet access for: HuggingFace model downloads, Semantic Scholar API, ACL Anthology XML, arXiv API

### Environment

```bash
# Recommended: use the Anaconda Python 3.12 that the project was developed with
# On macOS with Homebrew Python 3.14, install deps into a venv:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

### Dependencies

See `requirements.txt` (root) for the core pipeline. Component 2's `com2_using_3models/requirements.txt` has additional model-training dependencies.

### Environment Variables

```bash
# Optional but recommended — raises Semantic Scholar API rate limit from
# 100 req/5min to 1 req/sec sustained. Get a free key at:
# https://www.semanticscholar.org/product/api
export S2_API_KEY=your_key_here
```

Without `S2_API_KEY`, the pipeline uses a slower rate-limited fallback (3.5s between requests).

## Running the Pipeline

All scripts use relative paths derived from their own location. Run from the project root or from within each component directory.

### Component 1 — Data Collection

```bash
cd data_collection/scripts

# 1/4: ACL Anthology (NLP primary corpus)
python 01_collect_acl_anthology.py --years 2018 2019 2020 2021 2022 2023 2024

# 2/4: Semantic Scholar enrichment + COVID search
python 02_collect_semantic_scholar.py enrich --year 2023
python 02_collect_semantic_scholar.py covid-search --years 2023 2024

# 3/4: arXiv NLP preprints
python 03_collect_arxiv.py --years 2018 2019 2020 2021 2022 2023 2024

# 4/4: CORD-19 filtering + COVID harvesting + preprocessing
python 04_filter_cord19.py
python 05_collect_covid_papers.py
python 06_preprocess_and_sample.py   # produces nlp_corpus_sampled.json, covid_corpus_sampled.json
```

### Component 2 — Entity & Relation Extraction

```bash
cd component2_entity_relation_extraction/scripts

# Tier 1 baseline (no model downloads needed)
python extract_entities_relations_baseline.py

# Tier 2 SciBERT NER + relation extraction (requires internet for SciERC + model download)
python extract_entities_relations_scibert.py --step full
python extract_relations_scibert.py --step full

# Multi-model comparison (SciBERT + RoBERTa + PubMedBERT)
cd ../../com2_using_3models/scripts
python extract_entities_relations_multimodel.py --model all --step full
```

### Component 3 — Retrospective Validation

```bash
cd component3
python component3_retrospective_validation.py --domain NLP --cutoff 2021
python component3_retrospective_validation.py --domain COVID --cutoff 2021
```

### Component 4 — Knowledge Graph

```bash
cd component4
python build_graph.py
# Opens graph_viewer.html in a browser for interactive exploration
```

### Component 5 — Embeddings + Gap Ranking + Evaluation

```bash
cd component5

# Step 1: Build per-year cumulative NetworkX slices
python 01_build_slices.py

# Step 2: Structural embeddings (Node2Vec + Procrustes alignment)
python 02_structural_embeddings.py

# Step 3: Semantic embeddings (SPECTER2 / MiniLM fallback)
python 03_semantic_embeddings.py

# Step 4: Fuse + rank gaps via FAISS
python 04_fuse_and_rank.py

# Step 5: Evaluate via retrospective validation
python 05_evaluate.py
```

### Component 6 — Citations + Velocity + Rerank

```bash
cd component6

# Step 1: Fetch citation counts from Semantic Scholar API
python 01_fetch_citations.py --domain NLP
python 01_fetch_citations.py --domain COVID

# Step 2: Entity velocity over time
python 02_entity_velocity.py

# Step 3: Rerank gaps with citation velocity
python 03_rerank.py

# Step 4: Evaluate reranked gaps
python 04_evaluate.py
```

### Dashboard

```bash
cd dashboard
streamlit run app.py
```

## Datasets

| File | Description | Size |
|------|-------------|------|
| `data_collection/data/nlp/acl_anthology_{year}.json` | ACL Anthology papers per year (raw) | ~5–9 MB/year |
| `data_collection/data/nlp/arxiv_cscl_{year}.json` | arXiv CS-CL preprints per year | varies |
| `data_collection/data/nlp/nlp_corpus_sampled.json` | Consolidated 1400-paper NLP corpus | ~11 MB |
| `data_collection/data/nlp.zip` | Bundled NLP data archive | 11 MB |
| `data_collection/data/covid/covid_harvested_{year}.json` | COVID papers per year (raw) | ~1–2 MB/year |
| `data_collection/data/covid/covid_corpus_sampled.json` | Consolidated 840-paper COVID corpus | ~2 MB |
| `data_collection/data/covid/cord19_filtered_{year}.json` | CORD-19 filtered subset | varies |

## Important Notes

1. **Two Component 2 trees:** `component2_entity_relation_extraction/` is the original SciBERT-only pipeline. `com2_using_3models/` is the multi-model evolution (SciBERT + RoBERTa + PubMedBERT). Both are preserved as separate trees — check both when modifying Component 2 code.

2. **Absolute paths were fixed:** Scripts in `component5/` and `com2_using_3models/scripts/eval_scierc_relation_f1.py` originally contained hard-coded `/Users/anjan/Desktop/capstone_sep_15` paths. These have been converted to relative paths derived from `__file__`. If you add new scripts, use the `Path(__file__).resolve().parents[N]` pattern.

3. **Large model checkpoints are NOT in this repo:** The fine-tuned NER/relation model weights (~2.5 GB total across both Component 2 trees) are `gitignore`d. They must be re-downloaded/fine-tuned on clone. See `component2_entity_relation_extraction/scripts/checkpoints/README*` or the Component 2 report for download instructions.

4. **Regenerable outputs are NOT in this repo:** Embeddings (.npy), graph slices (.gpickle), and the full sweep of intermediate gap/rerank files are `gitignore`d. Only the final research-result artifacts (ablation reports, top-gap summaries, validation results) are tracked.

5. **Dashboard exports are NOT in this repo:** The 48 CSV files in `dashboard/exports/` are generated by `make_demo_embeddings.py` and are `gitignore`d.

6. **Semantic Scholar API:** Components 1 and 6 call the S2 API. Without an API key, rate limits apply. Set `S2_API_KEY` env var for production use.

## Future Improvements

- Add a top-level `data/processed/` directory for large generated artifacts (currently scattered across component outputs)
- Consolidate the two Component 2 trees into a single parameterized pipeline
- Add unit tests for core data-loading and validation functions
- Add a `pyproject.toml` with proper package structure
- Containerize the pipeline (Docker) for reproducible runs

## License

MIT License — see `LICENSE` file.

# Temporal Knowledge Graph Embedding for Emerging Research Gap Detection

A 7-component ML/NLP pipeline that discovers emerging research gaps in scientific literature by building temporal knowledge graphs from paper corpora, extracting entities and relations with SciBERT, computing structural + semantic embeddings, and ranking candidate gaps by fused similarity.

**Domains:** NLP (ACL Anthology + arXiv, 2018–2024, 1400 papers) · COVID-19 (multi-source, 2019–2024, 840 papers)

**Paper:** `paper_with_equations.tex` (IEEE Access format) · [DOCX version](Temporal%20Knowledge%20Graph%20Embedding%20for%20Emerging%20Research%20Gap%20Detection%20in%20Scientific%20Literature%20(1).docx)

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
├── component2_entity_relation_extraction/   # Component 2 (original SciBERT pipeline) — CANONICAL
│   ├── scripts/            # NER + relation extraction, baseline, validation
│   ├── output/             # Consolidated {NLP,COVID}_entities.json, _relations.json
│   ├── reports/            # Component 2 report
│   └── logs/
│
├── com2_using_3models/     # Component 2 (multi-model evolution: SciBERT + RoBERTa + PubMedBERT) — reference only
│   ├── scripts/            # Multi-model extraction pipeline + extra eval scripts
│   ├── output/             # Per-model subdirectories + consolidated outputs (identical to canonical tree)
│   ├── reports/            # Model comparison report
│   └── logs/
│
├── component3/             # Component 3 — retrospective validation
├── component4/             # Component 4 — knowledge graph builder + HTML viewer
├── component5/             # Component 5 — structural + semantic embeddings, gap ranking, evaluation
├── component6/             # Component 6 — citation velocity + reranking
│
├── dashboard/              # Component 7 — Streamlit interactive explorer
│   ├── app.py              # Main Streamlit app
│   ├── config.py           # Shared paths & constants
│   ├── data_loader.py      # Data loading utilities
│   ├── gap_engine.py       # Gap ranking logic
│   ├── make_demo_embeddings.py
│   ├── export_real_entities_edges.py   # C2 JSON → dashboard CSV (entities + edges)
│   ├── export_real_embeddings.py       # C5 .npy → dashboard CSV (node2vec + specter2)
│   ├── export_real_citations.py        # C6 velocity JSON → dashboard CSV (citations)
│   ├── exports/            # Generated CSV exports (gitignored)
│   └── BUILD.md            # Real-data bridge documentation
│
├── paper_with_equations.tex              # Journal paper (LaTeX, IEEE Access format)
├── equations_mapping.md                  # Equation-to-code line mapping for the paper
├── DATASET_SOURCES_AND_LINKS.md          # Centralized catalog of all dataset sources & download links
└── docs/                   # (future — currently top-level .md files serve this role)
```

## Components

| # | Directory | Purpose | Key outputs |
|---|-----------|---------|-------------|
| 1 | `data_collection/` | Collect papers from ACL Anthology, Semantic Scholar, arXiv, CORD-19, PubMed, Europe PMC, OpenAlex | `data/nlp/*.json`, `data/covid/*.json` (1400 NLP + 840 COVID) |
| 2 | `component2_entity_relation_extraction/` | SciBERT NER + relation extraction over sampled corpus | `{domain}_entities.json` (30,248 NLP / 29,814 COVID), `{domain}_relations.json` (13,122 NLP / 12,162 COVID) |
| 2b | `com2_using_3models/` | Same task with 3 transformer backbones for comparison | Per-model outputs + comparison report |
| 3 | `component3/` | Retrospective validation: predict gaps pre-cutoff, check post-cutoff co-occurrence | `output/validation_*.json/.md` |
| 4 | `component4/` | Build directed NetworkX KG per year + interactive HTML viewer | `graph_viewer.html`, `graph_summary.json` |
| 5 | `component5/` | Node2Vec structural embeddings + SPECTER semantic embeddings → fuse (alpha) → rank gaps via FAISS → evaluate | `output/gaps/`, `output/embeddings/`, ablation reports, `top_gaps_*_latest.json` |
| 6 | `component6/` | Entity velocity (mention counts) → re-rank Component 5 gaps by priority | `output/reranked/` (65 gap files), `output/ablation_velocity_*`, `top_velocity_gaps_*_latest.json` |
| 7 | `dashboard/` | Streamlit app to explore gaps, graph timeline, validation results. Reads **real** Component 2/5/6 data via export scripts | `app.py` + supporting modules + `exports/` CSVs |

## Technologies

- **Python 3.12** (Anaconda distribution recommended; see Setup)
- **PyTorch + Transformers** (SciBERT, RoBERTa, PubMedBERT for NER/relation extraction)
- **NetworkX** (temporal graph slicing)
- **Node2Vec** (structural embeddings)
- **Sentence Transformers / SPECTER** (semantic embeddings)
- **FAISS** (gap ranking via cosine similarity)
- **rapidfuzz** (entity normalization, near-duplicate filtering)
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
python extract_entities_relations_scibert.py --train   # one-time: fine-tune the NER head on SciERC
python extract_entities_relations_scibert.py --run     # inference over the sampled corpora
python extract_relations_scibert.py --train            # one-time: fine-tune the relation classifier
python extract_relations_scibert.py --run              # classify relations over the extracted entity spans

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

# Step 3: Semantic embeddings (SPECTER / MiniLM fallback)
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

### Component 7 — Dashboard

```bash
cd dashboard

# Export real pipeline data to dashboard CSV format (run after Components 2, 5, 6)
python export_real_entities_edges.py   # C2 JSON → entities + edges CSVs
python export_real_embeddings.py        # C5 .npy → node2vec + specter2 CSVs
python export_real_citations.py         # C6 velocity JSON → citations CSVs

# Launch the dashboard
streamlit run app.py
```

## Datasets

| File | Description | Size |
|------|-------------|------|
| `data_collection/data/nlp/acl_anthology_{year}.json` | ACL Anthology papers per year (raw) | ~2.3–9 MB/year |
| `data_collection/data/nlp/arxiv_cscl_{year}.json` | arXiv CS-CL preprints per year | varies |
| `data_collection/data/nlp/nlp_corpus_sampled.json` | Consolidated 1400-paper NLP corpus | ~2 MB |
| `data_collection/data/nlp.zip` | Bundled NLP data archive | 11 MB |
| `data_collection/data/covid/covid_harvested_{year}.json` | COVID papers per year (raw) | ~1–2 MB/year |
| `data_collection/data/covid/covid_corpus_sampled.json` | Consolidated 840-paper COVID corpus | ~2 MB |
| `data_collection/data/covid/cord19_filtered_{year}.json` | CORD-19 filtered subset | varies |

## Documentation

| File | Description |
|------|-------------|
| `paper_with_equations.tex` | Full journal paper in LaTeX (IEEE Access format) with numbered equations |
| `equations_mapping.md` | Maps every equation in the paper to its source code location |
| `DATASET_SOURCES_AND_LINKS.md` | Centralized catalog of all dataset sources, portals, and download links |
| `dashboard/BUILD.md` | Documents the real-data export bridge (Component 7) |
| `component5/working_log.md` | Component 5 development log |
| `component6/working_log.md` | Component 6 development log |
| `component3/COMPONENT3_REPORT.md` | Component 3 detailed report |
| `data_collection/SAMPLING_REPORT.md` | Data collection and sampling report |

## Important Notes

1. **Two Component 2 trees:** `component2_entity_relation_extraction/` is the original SciBERT-only pipeline. `com2_using_3models/` is the multi-model evolution (SciBERT + RoBERTa + PubMedBERT). Both are preserved as separate trees — check both when modifying Component 2 code.

2. **Absolute paths were fixed:** Scripts in `component5/` and `com2_using_3models/scripts/eval_scierc_relation_f1.py` originally contained hard-coded `/Users/anjan/Desktop/capstone_sep_15` paths. These have been converted to relative paths derived from `__file__`. If you add new scripts, use the `Path(__file__).resolve().parents[N]` pattern.

3. **Large model checkpoints are NOT in this repo:** The fine-tuned NER/relation model weights (~2.5 GB total across both Component 2 trees) are `gitignore`d. They must be re-downloaded/fine-tuned on clone — run the `--train` steps under Component 2 above, or see the Component 2 report for details.

4. **Regenerable outputs are NOT in this repo:** Embeddings (.npy), graph slices (.gpickle), and the full sweep of intermediate gap/rerank files are `gitignore`d. Only the final research-result artifacts (ablation reports, top-gap summaries, validation results) are tracked.

5. **Dashboard exports are NOT in this repo:** The CSV files in `dashboard/exports/` are generated by the export scripts and are `gitignore`d. Two types of data coexist:
   - **Real data** (via `export_real_*.py`): Actual Component 2/5/6 outputs converted to CSV. Used by the Top Gaps tab and gap analysis.
   - **Synthetic data** (via `make_demo_embeddings.py`): Random `np.random.randn` vectors for graph-exploration UI responsiveness. Used only by the Graph Timeline tab. **Not** real embeddings — do not use for reported results.

6. **Semantic Scholar API:** Components 1 and 6 can call the S2 API, but it requires institutional access that is not available on this machine. The pipeline therefore runs on **mention velocity** (papers-per-year counts from Component 2's `entities.json`), recorded by the `FALLBACK.txt` flags in `component6/output/citations/`. Mention velocity measures publication volume, not citation impact — do not report it as a citation result. Set `S2_API_KEY` and delete the flags to switch paths.

7. **Dashboard data flow:** The dashboard reads real pipeline data through three export scripts (`export_real_entities_edges.py`, `export_real_embeddings.py`, `export_real_citations.py`). These convert C2 JSON, C5 .npy, and C6 velocity JSON into the CSV format `data_loader.py` expects. Run them after re-running any pipeline component to refresh the dashboard.

## Future Improvements

- Add a top-level `data/processed/` directory for large generated artifacts (currently scattered across component outputs)
- Consolidate the two Component 2 trees into a single parameterized pipeline
- Add unit tests for core data-loading and validation functions
- Add a `pyproject.toml` with proper package structure
- Containerize the pipeline (Docker) for reproducible runs
- Extend the corpus past 2024 once those years have accumulated enough papers
- Add the trajectory term to the gap score for the main ranking path
- Compute real citation velocity with a window ending before the cutoff year
- Build entity vectors from paper text with SPECTER2 for richer semantic signals
- Conduct expert review of top-ranked gaps

## License

MIT License — see `LICENSE` file.

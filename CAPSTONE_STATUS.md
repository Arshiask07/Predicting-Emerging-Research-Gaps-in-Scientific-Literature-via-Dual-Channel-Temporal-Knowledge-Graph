# Capstone Status — Temporal Knowledge Graph Embedding
# Emerging Research Gap Detection in Scientific Literature
# ═══════════════════════════════════════════════════════════════════════════
# This complements the Word document. Quick-reference status only;
# full details are in the .md reports and the Word doc.

## Project
- **Title:** Temporal Knowledge Graph Embedding for Emerging Research Gap Detection in Scientific Literature
- **Author:** Anjan
- **Status:** Complete — 6 components implemented and validated
- **Python:** 3.12 (Anaconda) — scripts use relative paths, should run on any 3.12 install

## Corpus
| Domain | Papers | Years | Sources |
|--------|--------|-------|---------|
| NLP (ACL/arXiv) | 1,400 | 2018–2024 | ACL Anthology, arXiv CS-CL, Semantic Scholar enrichment |
| COVID-19 | 840 | 2019–2024 | PubMed, Europe PMC, OpenAlex, CORD-19, Semantic Scholar |

## Component Status
| # | Component | Status | Key outputs |
|---|-----------|--------|-------------|
| 1 | Data Collection | ✅ Done | 1400 NLP + 840 COVID sampled papers |
| 2a | Entity/Relation Extraction (SciBERT) | ✅ Done | 28,463 NLP entities, 22,690 relations; 11,918 COVID entities, 8,081 relations |
| 2b | Multi-Model Comparison (SciBERT+RoBERTa+PubMedBERT) | ✅ Done | 3-model comparison report |
| 3 | Retrospective Validation | ✅ Done | NLP ~53% hit rate, COVID ~20% (cutoff 2021) |
| 4 | Knowledge Graph Builder | ✅ Done | Per-year NetworkX slices + interactive HTML viewer |
| 5 | Embedding + Gap Ranking + Evaluation | ✅ Done | Structural (Node2Vec) + Semantic (SPECTER2) embeddings; alpha-sweep gap ranking; ablation reports |
| 6 | Citation Velocity + Rerank | ✅ Done | S2 citation fetch, entity velocity, reranked gaps |
| D | Streamlit Dashboard | ✅ Done | 4-tab interactive explorer |

## Key Findings
- NLP domain gap hit rate ~53% at cutoff 2021 (vs COVID ~20%)
- Alpha=0.5 fusion achieves best balance of structural + semantic signals
- Citation velocity improves gap ranking precision in Component 6

## Known Limitations
- Model checkpoints (~2.5 GB) not included in repo — must be re-downloaded on clone
- Large regenerable outputs (.npy, .gpickle, gap sweeps) not included — regenerate via pipeline scripts
- Dashboard exports (48 CSVs) not included — regenerate via make_demo_embeddings.py
- Semantic Scholar API requires key for production rate limits

## Repository Structure
See README.md for full architecture. Top-level docs:
- `COMPREHENSIVE_COMPONENT1_3_REPORT.md` — detailed Component 1–3 documentation
- `PROJECT_INTERNAL_DOCUMENTATION.md` — full internal docs (91 KB)
- `COMPONENT6_CHANGES.md` — Component 6 changelog
- `Component_3.md` — Component 3 specification
- `DATASET_SOURCES_AND_LINKS.md` — dataset provenance
- `emergent_landing.html` — standalone landing page
- `timeline.html` — timeline visualization

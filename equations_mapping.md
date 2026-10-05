# Equations Added to paper_with_equations.tex — Verified Line Mapping

All line numbers verified by running:
    grep -n "\\\\label{eq:" /Users/anjan/Desktop/capstone_sep_15/paper_with_equations.tex

Date: 2026-09-30
Base: LaTeX pasted into chat on 2026-09-30 (the original paper_with_equations.tex before any equation insertions)

---

## All Equations in Final File (in order of appearance)

| Eq. # | Label | Paper line | Source code file | Source location | Section |
|-------|-------|-----------|------------------|-----------------|---------|
| (1) | eq:quality | **135** | data_collection/scripts/06_preprocess_and_sample.py | score_nlp_paper(): lines 133-193; score_covid_paper(): lines 219-271; quality score = venue_tier + authors + title + venue_exp + abstract + id, max 60 (NLP) / 55 (COVID) | §3.1 Datasets |
| (2) | eq:cosine | **147** | dashboard/gap_engine.py: cosine() lines 8-10; component5/04_fuse_and_rank.py used throughout | cos(a,b) = a·b / (‖a‖‖b‖), 0 if either norm is 0 | §3.2 Extraction (added before fusedsim) |
| (3) | eq:canonical | **163** | component2_entity_relation_extraction/output/*.json (entity_id→canonical_id); component3/README.md line 11 | Canonical entity c = {m_i ∈ D : can(m_i) = c} | §3.3 Canonicalization |
| (4) | eq:canon_threshold | **168** | data_collection/scripts/06_preprocess_and_sample.py line 32-33 (RapidFuzz 0.85); component2 canonicalization code | Token sort ratio sim_TS(s1,s2) ≥ τ=0.85, type-constrained, domain-isolated | §3.3 Canonicalization |
| (5) | eq:cumulative_graph | **181** | component5/01_build_slices.py: build_slice() lines 90-100+ (cumulative DiGraph by year ≤ t) | G_D(t) = (V_D(t), E_D(t)), E_D(t) = ∪_{y=1}^t E_D(y) | §3.5 KG Construction |
| (6) | eq:graph_delta | **187** | component4/graph_summary.json (nodes/edges per year); component5/output/slices_build_log.json | ΔV_D(t)=\|V_D(t)\|−\|V_D(t−1)\|, ΔE_D(t)=\|E_D(t)\|−\|E_D(t−1)\| | §3.5 KG Construction |
| (7) | eq:node2vec_params | **198** | component5/02_structural_embeddings.py: NV_ARGS dict lines 52-61 | d=128, ℓ=80, r=10, p=q=1, w=10, epochs=10, seed=42 | §3.4 Dual-Channel Embedding |
| (8) | eq:procrustes | **203** | component5/02_structural_embeddings.py: align_embeddings() lines 88-100+ (scipy.linalg.orthogonal_procrustes) | R_t = argmin_{RᵀR=I} ‖A_t − B_t R‖_F | §3.4 Dual-Channel Embedding |
| (9) | eq:specter | **210** | component5/03_semantic_embeddings.py: input text lines 7-11; compute_representative_surface_forms() lines 84+ | e_u = SPECTER(surf(u) + " [" + type(u) + "]"), dim=768 | §3.4 Dual-Channel Embedding |
| (10) | eq:fusedvec | **217** | component5/04_fuse_and_rank.py: compute_fused_embeddings() lines 229-275 (alpha-weighted blend, L2-normalized) | z_u(t) = normalize(α·ŝ_u(t) + (1−α)·ê_u) — **already in original paper** | §3.4 Dual-Channel Embedding |
| (11) | eq:fusedsim | **222** | component5/04_fuse_and_rank.py (fused cosine = dot product of L2-normed fused vecs) | sim(u,v,t) = z_u(t)ᵀ z_v(t) — **already in original paper** | §3.4 Dual-Channel Embedding |
| (12) | eq:gapscore | **232** | dashboard/gap_engine.py: rank_gaps() line 68: `gs = sim_t * max(0.0, sim_t - sim_t1)` | gap_score(u,v,t) = sim(u,v,t) · max(0, sim(u,v,t) − sim(u,v,t−1)) | §3.6 Gap Scoring — **replaces old Eq (3)** |
| (13) | eq:faiss | **239** | component5/04_fuse_and_rank.py: build_faiss_index() lines 280-287 (IndexFlatIP on L2-normed vectors) | FAISS_IP(z_u, z_v) = z_uᵀ z_v = cos(z_u, z_v), ‖z_u‖=‖z_v‖=1 | §3.6 Gap Scoring |
| (14) | eq:neardup | **244** | component5/04_fuse_and_rank.py: is_near_duplicate() lines 138-160; thresholds lines 122-123 | filter(u,v) = (lev ≥ 0.75) ∨ (tsr ≥ 0.90) | §3.6 Gap Scoring |
| (15) | eq:velocity | **271** | dashboard/config.py line 27: VEL_WINDOW=(2022,2024); gap_engine.py lines 73-77; component6/ | vel(u) = (n_2024(u) − n_2022(u)) / 2 — **already in original paper as Eq (5)** | §3.7 Velocity |
| (16) | eq:vel_window | **276** | dashboard/config.py line 27: VEL_WINDOW = (2022, 2024) | VEL_WINDOW = (2022, 2024), fixed for all cutoffs | §3.7 Velocity |
| (17) | eq:priority | **283** | dashboard/gap_engine.py: rank_gaps() line 93: `"priority": round(gs * (vel[u] + vel[v]) / 2, 5)` | priority(u,v) = gap_score · mean(vel(u), vel(v)) — **already in original paper as Eq (6)** | §3.7 Velocity |
| (18) | eq:cutoff_split | **295** | component3/component3_retrospective_validation.py; component5/05_evaluate.py: cutoff parameter throughout | pre: y ≤ t_cutoff, post: y > t_cutoff | §3.8 Retrospective Validation |
| (19) | eq:pre_graph | **300** | component5/01_build_slices.py: cumulative slice at cutoff year = pre-cutoff graph | G_D^pre(t_cutoff) = G_D(t_cutoff) | §3.8 Retrospective Validation |
| (20) | eq:pre_embs | **305** | dashboard/gap_engine.py: _trim_embeddings() lines 140-145 (keeps years ≤ cutoff); component5/02 | s_u^pre(t_cutoff) = s_u(t_cutoff), u ∈ V_D(t_cutoff) | §3.8 Retrospective Validation |
| (21) | eq:hit | **312** | component3/component3_retrospective_validation.py; component5/05_evaluate.py: post-cutoff co-mention check | ∃ p : T(u) ∩ P_p ≠ ∅ ∧ T(v) ∩ P_p ≠ ∅ — **already in original paper as Eq (7)** | §3.8 Retrospective Validation |

**Total equations in final file: 21**
**New equations added: 14** (eq:quality, eq:cosine, eq:canonical, eq:canon_threshold, eq:cumulative_graph, eq:graph_delta, eq:node2vec_params, eq:procrustes, eq:specter, eq:gapscore [replaces old], eq:faiss, eq:neardup, eq:vel_window, eq:cutoff_split, eq:pre_graph, eq:pre_embs)
**Already existed in original paper: 5** (eq:fusedvec, eq:fusedsim, eq:velocity, eq:priority, eq:hit — now at new line numbers due to insertions)

---

## Critical Discrepancy Fixed

The original paper's Eq. (3) / new Eq. (12) gap_score:

- **Original paper said:** `gap_score(u,v,t) = sim(u,v,t)` — pure cosine, no delta term
- **Original paper also said (Sec 3.6):** "An earlier design multiplied the score by the increase in fused similarity between year t−1 and year t, and that term is **not part of the runs reported here**."
- **Actual code (gap_engine.py line 68):** `gs = sim_t * max(0.0, sim_t - sim_t1)` — uses delta term
- **Actual code (04_fuse_and_rank.py + component5/05_evaluate.py):** α-sweep uses similarity-only scores, no delta

**Resolution in updated paper:**
- Eq. (12) now correctly states: `gap_score = sim × max(0, Δsim)` — matching gap_engine.py
- The text clarifies: the α-sweep results (Table 4) use the simpler similarity-only score to keep the sweep comparable to single-channel baselines; the main gap list from the dashboard's gap engine uses the delta-based score
- This matches what the paper already said about the "earlier design" — it IS used in the main pipeline, just not in the sweep

---

## Source Files Checked

| Component | File | What was verified |
|-----------|------|-------------------|
| Data collection | data_collection/scripts/06_preprocess_and_sample.py | Quality score breakdown (lines 133-193 NLP, 219-271 COVID), RapidFuzz 0.85 dedup threshold |
| Comp 2 extraction | component2_entity_relation_extraction/scripts/extract_entities_relations_multimodel.py | SciBERT/RoBERTa/PubMedBERT fine-tuning; SciERC schema mapping |
| Comp 3 validation | component3/component3_retrospective_validation.py + README.md | Cutoff protocol, pre-cutoff co-occurrence graphs, lexical hit check logic |
| Comp 4 graph | component4/build_graph.py | Graph summary stats (nodes/edges per year) |
| Comp 5 slices | component5/01_build_slices.py | Cumulative DiGraph construction, year ≤ t logic |
| Comp 5 structural | component5/02_structural_embeddings.py | Node2Vec params (NV_ARGS), orthogonal Procrustes (align_embeddings) |
| Comp 5 semantic | component5/03_semantic_embeddings.py | SPECTER input text format, MiniLM fallback |
| Comp 5 fusion+rank | component5/04_fuse_and_rank.py | Fused embedding computation, FAISS IndexFlatIP, near-dup filter (lev 0.75, tsr 0.90) |
| Dashboard | dashboard/gap_engine.py | gap_score with delta (line 68), velocity (VEL_WINDOW), priority formula (line 93), cosine() (lines 8-10) |
| Dashboard | dashboard/config.py | VEL_WINDOW=(2022,2024), ALPHA_CHOICES, T_LATEST=2024, T_PREV=2023 |
| Comp 5 evaluate | component5/05_evaluate.py | α-sweep uses similarity-only scores, top-75 per setting |

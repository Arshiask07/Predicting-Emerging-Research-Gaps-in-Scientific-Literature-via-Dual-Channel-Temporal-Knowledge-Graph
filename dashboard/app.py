import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from pyvis.network import Network
from pathlib import Path
import sys

# Make component3/ discoverable for real-data validation import
SCAM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCAM_ROOT / "component3"))

import config
import data_loader as dl
import gap_engine as ge  # type: ignore[import-not-found]

st.set_page_config(page_title="Emerging Research Gap Explorer",
                   page_icon="🔬", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
domain = st.sidebar.selectbox("Domain", list(config.DOMAINS))
alpha = st.sidebar.select_slider(
    "Fusion weight α",
    options=config.REAL_ALPHAS, value=0.5,
    help="α = 0.0 is pure semantic (SPECTER), α = 1.0 is pure structural "
         "(Node2Vec). These are the five values the real pipeline was run with.")
min_vel = st.sidebar.number_input(
    "Min mention velocity", min_value=0.0, max_value=50.0, value=0.0, step=0.5,
    help="Drop pairs whose mean entity velocity falls below this. Default 0.0 "
         "keeps everything, including entities that are declining (negative velocity).")
top_k = st.sidebar.slider("Show top-K gaps", 5, 100, 15)

# Cache version key — increment when user explicitly requests recompute
if "version" not in st.session_state:
    st.session_state["version"] = 0

@st.cache_resource(show_spinner="Loading frozen corpus snapshot…")
def load_domain(dom, version):
    papers = dl.load_papers(dom)
    ents = dl.load_entities(dom)
    edges = dl.load_edges(dom)
    embs = dl.load_entity_embeddings(dom)
    cits = dl.load_citation_history(dom)
    graphs = ge.build_graphs(ents, edges)
    return papers, ents, edges, embs, cits, graphs

papers, ents, edges, embs, cits, graphs = load_domain(domain, st.session_state["version"])

if papers.empty:
    st.error(f"No papers found under `{config.DATA_ROOT}` — check "
             "`data_collection/scripts/06_preprocess_and_sample.py` output.")
    st.stop()

years_avail = [y for y, g in graphs.items() if len(g.nodes)]
if not years_avail:
    st.info("No graph snapshots yet — run the extraction pipeline "
            "(`make_demo_embeddings.py` fills placeholders).")
    st.stop()
YR_MIN, YR_MAX = min(years_avail), max(years_avail)

# ── Header ────────────────────────────────────────────────────────────
st.title("🔬 Temporal Research Gap Explorer")
st.caption(f"{domain} · {len(papers):,} papers · snapshots {YR_MIN}–{YR_MAX} · α={alpha}")

tab_overview, tab_gaps, tab_graph, tab_validate = st.tabs(
    ["📊 Corpus Overview", "🏆 Top Gaps", "🕸️ Graph Timeline", "✅ Validation"])

# ── Tab 1: Corpus overview ────────────────────────────────────────────
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    n_ents = len(ents) if ents is not None else 0
    n_edges = len(edges) if edges is not None else 0
    years_present = sorted(papers.year.unique())
    c1.metric("Papers indexed", f"{len(papers):,}")
    c2.metric("Concepts extracted", f"{n_ents:,}")
    c3.metric("Typed relations", f"{n_edges:,}")
    c4.metric("Coverage", f"{years_present[0]}–{years_present[-1]}")

    per_year = papers.groupby("year").size().reindex(years_present, fill_value=0)
    fig = go.Figure(go.Bar(x=per_year.index.astype(str), y=per_year.values,
                           marker_color="#6366f1"))
    fig.update_layout(title="Papers per year (frozen snapshot)", height=320,
                      margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Graph growth overlay (entities/relations appearing per snapshot year)
    if ents is not None and not ents.empty and edges is not None and not edges.empty:
        ent_growth = ents.groupby("first_year").size().reindex(
            range(YR_MIN, YR_MAX + 1), fill_value=0)
        edge_growth = edges.groupby("first_observed").size().reindex(
            range(YR_MIN, YR_MAX + 1), fill_value=0)
        figg = go.Figure()
        figg.add_bar(name="New concepts", x=list(ent_growth.index.astype(str)),
                     y=ent_growth.values, marker_color="#34d399", opacity=0.75)
        figg.add_bar(name="New relations", x=list(edge_growth.index.astype(str)),
                     y=edge_growth.values, marker_color="#60a5fa", opacity=0.75)
        figg.update_layout(barmode="overlay",
                           title="Knowledge graph growth per year", height=300,
                           margin=dict(t=40, b=10))
        st.plotly_chart(figg, use_container_width=True)

# ── Tab 2: Ranked gaps (real pipeline: Component 5 + Component 6) ────────
# Reads the FAISS-ranked gap files written by component5/04_fuse_and_rank.py
# and the velocity re-ranked files from component6/03_rerank.py. These use real
# Node2Vec + SPECTER embeddings; the synthetic demo embeddings in exports/ are
# NOT used for this tab.
with tab_gaps:
    gap_years = dl.available_gap_years(domain)
    if not gap_years:
        st.error(
            f"No ranked gap files found for **{domain}**. Run the real pipeline:\n\n"
            "```\n"
            f"python component5/04_fuse_and_rank.py --domain {config.DOMAIN_SHORT.get(domain)}\n"
            "```"
        )
    else:
        c_year, c_alpha, c_source = st.columns([1, 1, 1.4])
        with c_year:
            gap_year = st.selectbox("Snapshot year", gap_years,
                                    index=len(gap_years) - 1)
        with c_alpha:
            gap_alpha = st.select_slider("α", options=config.REAL_ALPHAS,
                                         value=alpha)
        with c_source:
            source = st.radio(
                "Ranking", ["Fused (Comp 5)", "Velocity re-ranked (Comp 6)"],
                horizontal=True,
                help="Comp 5 ranks by fused cosine similarity. Comp 6 re-ranks "
                     "those same pairs by priority = fused × mean(velocity).")

        use_reranked = source.startswith("Velocity")
        gaps = dl.load_real_gaps(domain, gap_year, gap_alpha, reranked=use_reranked)
        _, vel_path = dl.load_entity_velocity(domain)

        if gaps is None:
            st.warning(
                f"No `{source}` file for {config.DOMAIN_SHORT.get(domain)} "
                f"{gap_year} α={gap_alpha}. "
                + ("Run `component6/03_rerank.py` to produce it."
                   if use_reranked else
                   "Run `component5/04_fuse_and_rank.py` to produce it.")
            )
        elif gaps.empty:
            st.warning("That gap file is empty — try a different year or α.")
        else:
            # Optional velocity floor, applied client-side.
            if use_reranked and "vel_u" in gaps.columns and min_vel > 0:
                mean_vel = (gaps.vel_u + gaps.vel_v) / 2
                n_before = len(gaps)
                gaps = gaps[mean_vel >= min_vel].reset_index(drop=True)
                st.info(f"Velocity floor ≥ {min_vel}: kept {len(gaps):,} of "
                        f"{n_before:,} candidate pairs.")
                gaps_empty = gaps.empty
            else:
                gaps_empty = False

            if gaps_empty:
                st.warning(f"No pairs survive the velocity floor of {min_vel}.")
            else:
                shown = gaps.head(top_k)
                n_pool = len(gaps)
                n_checked = 0  # real pipeline pre-filters; see note below

                src = "Component 6 (velocity re-ranked)" if use_reranked \
                    else "Component 5 (FAISS fused ranking)"
                st.caption(
                    f"**{src}** · {config.DOMAIN_SHORT.get(domain)} · "
                    f"{gap_year} · α={gap_alpha} · showing {len(shown)} of "
                    f"{n_pool:,} ranked candidates (top-500 pool). "
                    f"Velocity signal: "
                    + ("mention counts (S2 API unreachable)" if vel_path == "mention"
                       else "Semantic Scholar citation counts")
                )

                for i, row in shown.iterrows():
                    with st.container(border=True):
                        col_r, col_pair, col_m = st.columns([0.06, 0.44, 0.50])
                        col_r.markdown(f"### {i+1}")
                        name_a = row.surface_form_a or row.entity_a
                        name_b = row.surface_form_b or row.entity_b
                        col_pair.markdown(
                            f"**{name_a}** ⟷ **{name_b}**\n\n"
                            f"<span style='color: #64748b; font-size: 0.82em;'>"
                            f"`{row.entity_a}` ⟷ `{row.entity_b}`</span>",
                            unsafe_allow_html=True,
                        )
                        m = col_m.columns(3)
                        if use_reranked:
                            m[0].metric("Priority", row.priority)
                            m[1].metric("Fused sim", row.fused_score)
                            m[2].metric("Mean vel",
                                        round((row.vel_u + row.vel_v) / 2, 2))
                        else:
                            m[0].metric("Fused sim", row.fused_score)
                            m[1].metric("Structural",
                                        round(row.structural_score, 3)
                                        if pd.notna(row.structural_score) else "—")
                            m[2].metric("Semantic",
                                        round(row.semantic_score, 3)
                                        if pd.notna(row.semantic_score) else "—")

                st.caption(
                    "Pairs have no direct edge of any relation type in the "
                    "cumulative graph for this year. The pool was built by FAISS "
                    "top-20 neighbour retrieval over fused vectors, then "
                    "near-duplicate surface forms were filtered "
                    "(Levenshtein ≥ 0.75 or token-set ≥ 0.90)."
                )

# ── Tab 3: Graph timeline ─────────────────────────────────────────────
with tab_graph:
    yr_lo, yr_hi = st.slider("Snapshot year range", YR_MIN, YR_MAX, (YR_MIN, YR_MAX))
    focus = st.text_input("Focus concept (optional)", "")
    show_edges = st.checkbox("Show typed relations", True)
    max_nodes = st.slider("Max nodes to render", 50, 2000, 500, step=50,
                          help="Larger graphs render slower. Lower this to keep the "
                               "visualization responsive. Nodes beyond the limit are omitted.")

    # Cache the merged graph by year range so slider changes don't rebuild from scratch
    # Version bumped on each server restart so stale caches don't survive
    if "graph_cache_version" not in st.session_state:
        st.session_state["graph_cache_version"] = hash(Path(__file__).resolve().stat().st_mtime)
    graph_cache_key = (f"merged_graph_{domain.replace(' ', '_')}_{yr_lo}_{yr_hi}"
                       f"_{focus}_{max_nodes}_{st.session_state['graph_cache_version']}")
    if graph_cache_key not in st.session_state:
        st.session_state[graph_cache_key] = None

    merged = None
    if st.session_state[graph_cache_key] is None and ents is not None and edges is not None:
        # Build debounced — only when slider settles
        merged = nx.Graph()
        for row in ents[(ents.first_year >= yr_lo) & (ents.first_year <= yr_hi)].itertuples():
            merged.add_node(row.entity_id, type=row.type, label=row.label)
        for row in edges[(edges.first_observed >= yr_lo)
                          & (edges.first_observed <= yr_hi)].itertuples():
            if merged.has_node(row.source) and merged.has_node(row.target):
                merged.add_edge(row.source, row.target, rel=row.relation)
        if focus and focus in merged:
            merged = nx.ego_graph(merged, focus, radius=2)
        # Truncate to max_nodes if needed — prefer nodes with highest degree centrality
        if len(merged.nodes) > max_nodes:
            deg = nx.degree_centrality(merged)
            top_nodes = sorted(deg, key=lambda n: deg[n], reverse=True)[:max_nodes]
            merged = merged.subgraph(top_nodes).copy()
        st.session_state[graph_cache_key] = merged

    merged = st.session_state.get(graph_cache_key)

    if merged is None:
        st.info("Enter a focus concept or adjust the year range to build the graph.")
    elif len(merged.nodes) == 0:
        st.info("No nodes in this selection — widen the year range or clear the focus.")
    elif len(merged.nodes) > 400:
        st.warning(f"{len(merged.nodes)} nodes selected — rendering may be slow. "
                   "Narrow the year range or set a focus concept.")
        _render = st.checkbox("Render anyway", value=False)
    else:
        _render = True

    if len(merged.nodes) > 0 and _render:
        # ── Render the merged graph with PyVis ──────────────────────────────
        # Human-readable node labels from entity data (not canonical IDs)
        # Human-readable edge labels from relation-type mapping
        net = Network(height="600px", bgcolor="#0e1117", font_color="white",
                      heading="", notebook=False)
        palette = {"Method": "#60a5fa", "Task": "#34d399", "Metric": "#fbbf24",
                   "Material": "#f472b6", "Dataset": "#a78bfa", "Model": "#fb923c"}

        def _good_label(node_id, attr):
            """Return a display label: prefer the entity label field, with
            quality checks. Short/empty labels fall back to the canonical ID;
            very long labels are truncated."""
            raw = attr.get("label", "")
            if not raw or len(raw) <= 2:
                return node_id       # e.g. "LL", "ja" → show "canon_Other_00129"
            if len(raw) > 30:
                return raw[:27] + "…"  # e.g. "German↔English and Chinese→English translation tasks" → truncated
            return raw

        for n, d in merged.nodes(data=True):
            net.add_node(n,
                         label=_good_label(n, d),
                         color=palette.get(d.get("type"), "#94a3b8"),
                         size=12)

        # ── Human-readable edge labels ──────────────────────────────────────
        REL_LABELS = {
            "METHOD_APPLIED_TO":      "Method applied to",
            "METHOD_EVALUATED_BY":    "Method evaluated by",
            "USED_FOR":               "Used for",
            "ENTITY_ASSOCIATED_WITH_ENTITY": "Associated with",
        }
        if show_edges:
            for s, t_, d in merged.edges(data=True):
                net.add_edge(s, t_,
                             title=REL_LABELS.get(d.get("rel", ""),
                                                  d.get("rel", "")),
                             color="#475569")

        html = net.generate_html(notebook=False)
        # Disable long stabilization — 1000 default iterations = 30-60s of
        # browser-side settling for 500 nodes. 10 iterations settles in <1s.
        html = html.replace('"iterations": 1000', '"iterations": 10')
        html = html.replace('"fit": true', '"fit": false')
        components.html(html, height=620, scrolling=False)

# ── Tab 4: Retrospective validation ───────────────────────────────────
with tab_validate:
    st.subheader("Retrospective hit rate (Contribution 3)")
    st.caption("Gaps predicted using data ≤2021 only, then checked against "
               "real co-mentions in publications from 2022–2024.")

    val_mode = st.radio(
        "Validation mode",
        ["Demo embeddings (from make_demo_embeddings.py)",
         "Real extraction data (Component 2 JSON)"],
        horizontal=True,
        help="Demo mode scores gaps from synthetic embeddings. "
             "Real mode loads the actual SciBERT-extracted entities/relations "
             "and checks surface-form co-mentions in post-cutoff papers.",
    )

    if val_mode.startswith("Demo"):
        st.info("Using frozen demo embeddings from `dashboard/exports/`.")
        demo_key = f"demo_val_{domain.replace(' ', '_')}_{alpha}_{top_k}"
        if st.button("▶ Run demo validation", type="primary"):
            with st.spinner(f"Re-scoring gaps at t={config.VALIDATION_CUTOFF}…"):
                try:
                    hits, total, top_hits_df = ge.retrospective_validate(
                        embs, graphs, papers, ents=ents, cutoff=config.VALIDATION_CUTOFF,
                        top_k=top_k, alpha=alpha, sample_n=400, return_df=True)
                    st.session_state[demo_key] = (hits, total, top_hits_df)
                except Exception as e:
                    st.error(f"Demo validation failed: {e}")
                    st.session_state[demo_key] = None

        if demo_key in st.session_state and st.session_state[demo_key] is not None:
            hits, total, top_hits_df = st.session_state[demo_key]
            if total == 0:
                st.error("No candidate gaps could be scored at the cutoff year — "
                         "check that pre-cutoff embeddings exist.")
            elif hits == 0:
                st.warning(f"0/{total} predicted gaps realized post-cutoff.")
            else:
                rate = hits / total
                st.success(f"Demo hit rate @top-{top_k}: **{rate:.1%}** "
                           f"({hits}/{total})")
                if top_hits_df is not None and not top_hits_df.empty:
                    st.subheader("Top materialized gaps (post-cutoff)")
                    ent_map = {}
                    if ents is not None and not ents.empty and "label" in ents.columns:
                        ent_map = dict(zip(ents.entity_id, ents.label))
                    for idx, h in top_hits_df.head(10).reset_index(drop=True).iterrows():
                        u_name = ent_map.get(h["u"], h["u"])
                        v_name = ent_map.get(h["v"], h["v"])
                        st.markdown(
                            f"{idx + 1}. **{u_name}** ⟷ **{v_name}**  \n"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;<small style='color: #64748b;'>`{h['u']}` ⟷ `{h['v']}` · gap_score={h.get('gap_score', 0):.4f}</small>",
                            unsafe_allow_html=True,
                        )

    else:
        st.info("Loading real Component 2 extraction outputs from "
                "`component2_entity_relation_extraction/output/`.")
        real_key = f"real_val_{domain.replace(' ', '_')}_{alpha}_{top_k}"
        if st.button("▶ Run real-data validation", type="primary"):
            try:
                import component3_retrospective_validation as c3
            except ImportError:
                st.error("Cannot import `component3_retrospective_validation` — "
                         "ensure `component3/` is on the Python path.")
            else:
                with st.spinner("Loading extraction data & scoring gaps…"):
                    try:
                        report = c3.validate_with_real_data(
                            domain, cutoff=config.VALIDATION_CUTOFF,
                            top_k=top_k, alpha=alpha,
                            min_papers_per_entity=3,
                            out_dir=Path(__file__).resolve().parent
                            / "exports" / "component3_real_validation",
                        )
                        st.session_state[real_key] = report
                    except Exception as e:
                        st.error(f"Real-data validation failed: {e}")
                        st.session_state[real_key] = None

        if real_key in st.session_state and st.session_state[real_key] is not None:
            report = st.session_state[real_key]
            if report.get("status") == "ok":
                rate = report["hit_rate"]
                st.success(
                    f"Real-data hit rate @top-{top_k} "
                    f"(cutoff {config.VALIDATION_CUTOFF}): "
                    f"**{rate:.1%}** ({report['hits']}/{report['candidate_gaps_scored']})"
                )
                st.info(
                    f"{report['entities_considered']:,} entities · "
                    f"{report['relations_considered']:,} relations · "
                    f"{report['papers_pre_cutoff']:,} pre-cutoff papers · "
                    f"{report['papers_post_cutoff']:,} post-cutoff papers · "
                    f"{report['elapsed_seconds']}s"
                )
                if report.get("top_hits"):
                    st.subheader("Top materialized gaps (post-cutoff)")
                    ent_map = {}
                    if ents is not None and not ents.empty and "label" in ents.columns:
                        ent_map = dict(zip(ents.entity_id, ents.label))
                    for i, h in enumerate(report["top_hits"][:10], 1):
                        u_name = ent_map.get(h["u"], h["u"])
                        v_name = ent_map.get(h["v"], h["v"])
                        extra = f"gap_score={h['gap_score']:.4f}" if "gap_score" in h else f"{h.get('pre_cutoff_cooccurrences', 0)} pre-cutoff co-occurrences"
                        st.markdown(
                            f"{i}. **{u_name}** ⟷ **{v_name}**  \n"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;<small style='color: #64748b;'>`{h['u']}` ⟷ `{h['v']}` · {extra}</small>",
                            unsafe_allow_html=True,
                        )
            else:
                st.warning(f"Validation skipped: {report.get('reason', 'unknown')}")

st.divider()
st.caption("Frozen snapshot — no live API calls. Data source: "
           f"`{config.DATA_ROOT.relative_to(config.CAPPRO_ROOT)}`")

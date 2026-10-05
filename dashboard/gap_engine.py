import numpy as np
import pandas as pd
import networkx as nx
import config
import re


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def fused_sim(un, vn, us, vs, alpha):
    return alpha * cosine(un, vn) + (1 - alpha) * cosine(us, vs)


def build_graphs(entities_df, edges_df):
    """One nx.Graph snapshot per year from Component 4 CSV outputs."""
    graphs = {}
    for y in config.YEARS_ALL:
        g = nx.Graph()
        if entities_df is not None:
            sub = entities_df[entities_df.first_year <= y]
            g.add_nodes_from(sub.entity_id)
        if edges_df is not None:
            for _, r in edges_df[edges_df.first_observed <= y].iterrows():
                if g.has_node(r.source) and g.has_node(r.target):
                    g.add_edge(r.source, r.target)
        graphs[y] = g
    return graphs


def rank_gaps(
    embeddings,
    graphs,
    cit_df,
    alpha=config.ALPHA_CHOICES[1],
    t=config.T_LATEST,
    t1=config.T_PREV,
    min_vel=0.0,
    top_k=None,
    sample_n=None,       # if set, randomly sample this many nodes before pairing
    seed=42,
):
    ch = embeddings.get("node2vec", {})
    sp = embeddings.get("specter2", {})
    if t not in ch or t1 not in ch or t not in sp or t1 not in sp:
        return pd.DataFrame(), 0

    common = sorted(set(ch[t].index) & set(ch[t1].index) & set(sp[t].index) & set(sp[t1].index))
    if not common:
        return pd.DataFrame(), 0

    # 1. Precompute citation velocities once for all entities
    vel_dict = {}
    if cit_df is not None and not cit_df.empty and "entity_id" in cit_df.columns:
        piv = cit_df.pivot(index="entity_id", columns="year", values="citations").fillna(0)
        vel_end = min(t, config.VEL_WINDOW[1])
        cols_avail = [c for c in piv.columns if c < vel_end]
        vel_start = max(cols_avail) if cols_avail else (vel_end - 2)
        if vel_end in piv.columns and vel_start in piv.columns and vel_end > vel_start:
            vel_dict = ((piv[vel_end] - piv[vel_start]) / (vel_end - vel_start)).to_dict()

    # 2. Node sampling: if min_vel > 0, prioritize nodes that meet min_vel
    if min_vel > 0:
        active_nodes = [n for n in common if vel_dict.get(n, 0.0) >= min_vel]
        if active_nodes:
            if sample_n and sample_n < len(common):
                rng = np.random.default_rng(seed)
                needed = min(sample_n, len(common))
                n_active = min(len(active_nodes), max(1, needed // 2))
                sel_active = rng.choice(active_nodes, size=n_active, replace=False).tolist()
                rem = list(set(common) - set(sel_active))
                n_rem = needed - n_active
                sel_rem = rng.choice(rem, size=n_rem, replace=False).tolist() if n_rem > 0 and rem else []
                common = sorted(sel_active + sel_rem)
        else:
            if sample_n and sample_n < len(common):
                rng = np.random.default_rng(seed)
                common = sorted(rng.choice(common, size=sample_n, replace=False).tolist())
    elif sample_n and sample_n < len(common):
        rng = np.random.default_rng(seed)
        common = sorted(rng.choice(common, size=sample_n, replace=False).tolist())

    # 3. Vectorized cosine similarities using normalized NumPy matrices
    sub_n_t = ch[t].loc[common].to_numpy(dtype=np.float32)
    sub_n_t /= np.maximum(np.linalg.norm(sub_n_t, axis=1, keepdims=True), 1e-9)
    sub_n_t1 = ch[t1].loc[common].to_numpy(dtype=np.float32)
    sub_n_t1 /= np.maximum(np.linalg.norm(sub_n_t1, axis=1, keepdims=True), 1e-9)

    sub_s_t = sp[t].loc[common].to_numpy(dtype=np.float32)
    sub_s_t /= np.maximum(np.linalg.norm(sub_s_t, axis=1, keepdims=True), 1e-9)
    sub_s_t1 = sp[t1].loc[common].to_numpy(dtype=np.float32)
    sub_s_t1 /= np.maximum(np.linalg.norm(sub_s_t1, axis=1, keepdims=True), 1e-9)

    sim_t_mat = alpha * (sub_n_t @ sub_n_t.T) + (1.0 - alpha) * (sub_s_t @ sub_s_t.T)
    sim_t1_mat = alpha * (sub_n_t1 @ sub_n_t1.T) + (1.0 - alpha) * (sub_s_t1 @ sub_s_t1.T)
    delta_mat = np.maximum(0.0, sim_t_mat - sim_t1_mat)
    gs_mat = sim_t_mat * delta_mat

    g = graphs.get(t) if graphs else None
    rows = []
    checked = 0
    N = len(common)

    for i in range(N):
        u = common[i]
        vel_u = max(0.0, vel_dict.get(u, 0.0))
        for j in range(i + 1, N):
            v = common[j]
            if g is not None and g.has_edge(u, v):
                continue
            checked += 1
            gs = float(gs_mat[i, j])
            if gs <= 0:
                continue
            vel_v = max(0.0, vel_dict.get(v, 0.0))
            if min_vel > 0 and max(vel_u, vel_v) < min_vel:
                continue
            sim_t_val = float(sim_t_mat[i, j])
            delta_val = float(delta_mat[i, j])
            mean_vel = (vel_u + vel_v) / 2.0
            priority = gs * mean_vel if mean_vel > 0 else gs
            rows.append({
                "u": u, "v": v,
                "sim_t": round(sim_t_val, 4),
                "delta_sim": round(delta_val, 4),
                "gap_score": round(gs, 5),
                "vel_u": round(vel_u, 1),
                "vel_v": round(vel_v, 1),
                "priority": round(priority, 5),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, checked
    df = df.sort_values("priority", ascending=False)
    if top_k:
        df = df.head(top_k).reset_index(drop=True)

    # 4. Compute history only for final top-k gaps
    hist_list = []
    all_years = sorted(ch.keys())
    for _, r in df.iterrows():
        u, v = r.u, r.v
        hist = {}
        for y in all_years:
            if (u in ch[y].index and v in ch[y].index and
                y in sp and u in sp[y].index and v in sp[y].index):
                vu_n = ch[y].loc[u].to_numpy()
                vv_n = ch[y].loc[v].to_numpy()
                vu_s = sp[y].loc[u].to_numpy()
                vv_s = sp[y].loc[v].to_numpy()
                cos_n = float(np.dot(vu_n, vv_n) / (np.linalg.norm(vu_n) * np.linalg.norm(vv_n) + 1e-9))
                cos_s = float(np.dot(vu_s, vv_s) / (np.linalg.norm(vu_s) * np.linalg.norm(vv_s) + 1e-9))
                hist[y] = round(alpha * cos_n + (1.0 - alpha) * cos_s, 4)
        hist_list.append(hist)
    df["sim_history"] = hist_list

    return df, checked


def retrospective_validate(embeddings, graphs, papers, ents=None, cutoff=config.VALIDATION_CUTOFF,
                           top_k=None, alpha=config.ALPHA_CHOICES[1],
                           sample_n=400, seed=42, return_df=False):
    """Contribution 3: score gaps using only data <= cutoff, check post-cutoff.

    Returns (hits, total) or (hits, total, top_hits_df) if return_df is True.
    """
    pre_embs = _trim_embeddings(embeddings, cutoff)
    pre_graphs = {y: g for y, g in graphs.items() if y <= cutoff}
    pre_cits = _trim_citations(papers, cutoff) if not papers.empty else pd.DataFrame()
    ranked, checked = rank_gaps(pre_embs, pre_graphs, pre_cits, alpha=alpha,
                                t=cutoff, t1=cutoff - 1, top_k=top_k,
                                sample_n=sample_n, seed=seed)
    if checked == 0 or ranked.empty:
        return (0, 0, pd.DataFrame()) if return_df else (0, 0)

    # Build surface tokens for entities
    from collections import defaultdict
    ent_surfaces = defaultdict(set)
    if ents is not None and not ents.empty and "label" in ents.columns:
        for _, r in ents.iterrows():
            lbl = str(r.label or r.entity_id).lower()
            for tok in re.findall(r"[a-z]+", lbl):
                if len(tok) >= 4:
                    ent_surfaces[r.entity_id].add(tok)

    # Build post-cutoff paper text token index for co-mention check
    post = papers[papers.year > cutoff]
    if post.empty:
        return (0, len(ranked), pd.DataFrame()) if return_df else (0, len(ranked))

    post_paper_idx = []
    for _, row in post.iterrows():
        txt = (str(row.title) + " " + str(row.abstract)).lower()
        toks = {t for t in re.findall(r"[a-z]+", txt) if len(t) >= 4}
        if toks:
            post_paper_idx.append(toks)

    # Check if predicted concept pairs co-occur in any post-cutoff paper
    hits = 0
    hit_rows = []
    for _, row in ranked.iterrows():
        u_surf = ent_surfaces.get(row.u, set())
        v_surf = ent_surfaces.get(row.v, set())
        if not u_surf:
            u_surf = {t for t in re.findall(r"[a-z]+", str(row.u).lower()) if len(t) >= 4}
        if not v_surf:
            v_surf = {t for t in re.findall(r"[a-z]+", str(row.v).lower()) if len(t) >= 4}

        materialized = False
        if u_surf and v_surf:
            for p_toks in post_paper_idx:
                if (u_surf & p_toks) and (v_surf & p_toks):
                    materialized = True
                    break
        if materialized:
            hits += 1
            hit_rows.append(row)

    total_candidates = len(ranked)
    hit_df = pd.DataFrame(hit_rows)
    if return_df:
        return hits, total_candidates, hit_df
    return hits, total_candidates


def _trim_embeddings(embeddings, cutoff):
    """Keep only embedding years <= cutoff."""
    out = {}
    for channel, frames in embeddings.items():
        out[channel] = {y: f for y, f in frames.items() if y <= cutoff}
    return out


def _trim_citations(papers, cutoff):
    """Build entity→citation history from papers <= cutoff only."""
    if papers.empty:
        return pd.DataFrame()
    from collections import defaultdict
    ent_cites = defaultdict(lambda: defaultdict(int))
    for _, row in papers[papers.year <= cutoff].iterrows():
        txt = (str(row.title) + " " + str(row.abstract)).lower()
        for tok in set(re.findall(r"[a-z]+", txt)):
            if len(tok) > 3:
                ent_cites[tok][int(row.year)] += 1
    rows = []
    for ent, yr_map in ent_cites.items():
        for yr, cnt in yr_map.items():
            rows.append({"entity_id": ent, "year": yr, "citations": cnt})
    return pd.DataFrame(rows)

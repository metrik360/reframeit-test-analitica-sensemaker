#!/usr/bin/env python3
"""
ETB SenseMaker Report Generator
=================================
Generates a self-contained bilingual HTML report for ETB.
Replicates the GCG report architecture adapted for ETB's instrument.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
IMPUTED = BASE / "04_imputed"
CLUSTERS = BASE / "05_clusters"
TEXT = BASE / "06_text"
DICT_PATH = BASE / "01_dictionary" / "label_dictionary.json"
OUTPUT = BASE.parent / "etb_report.html"

# ── Load Data ────────────────────────────────────────────────────────────
print("[Report] Loading data...")
triads = pd.read_csv(IMPUTED / "triads_imputed.csv")
dyads = pd.read_csv(IMPUTED / "dyads_imputed.csv")
stones = pd.read_csv(IMPUTED / "stones_imputed.csv")
categorical = pd.read_csv(IMPUTED / "categorical_filtered.csv")
text_df = pd.read_csv(IMPUTED / "text_filtered.csv")
metadata = pd.read_csv(IMPUTED / "metadata_filtered.csv")

assignments = pd.read_csv(CLUSTERS / "cluster_assignments.csv")
cluster_report = json.load(open(CLUSTERS / "clustering_report.json"))
profiles = pd.read_csv(CLUSTERS / "cluster_profiles.csv", index_col=0)
text_analysis = json.load(open(TEXT / "text_analysis.json", encoding="utf-8"))
label_dict = json.load(open(DICT_PATH, encoding="utf-8"))

# Merge cluster
triads['cluster'] = assignments['cluster'].values
dyads['cluster'] = assignments['cluster'].values
stones['cluster'] = assignments['cluster'].values
categorical['cluster'] = assignments['cluster'].values
text_df['cluster'] = assignments['cluster'].values

N = len(triads)
print(f"  N = {N} respondents, 3 clusters")

# ── Data Infrastructure (for client-side filtering & year comparison) ────

DEMO_FIELDS = ['cargo', 'antiguedad', 'area', 'genero', 'educacion']
TRIAD_IDS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
DYAD_IDS = ["D1", "D2", "D3", "D4", "D5", "D6"]
S1_COLS = [
    ("S1_ser_yo_mismo_X", "S1_ser_yo_mismo_Y"),
    ("S1_aceptar_errores_X", "S1_aceptar_errores_Y"),
    ("S1_pedir_ayuda_X", "S1_pedir_ayuda_Y"),
    ("S1_confianza_X", "S1_confianza_Y"),
    ("S1_empatia_X", "S1_empatia_Y"),
    ("S1_simplicidad_X", "S1_simplicidad_Y"),
    ("S1_adaptabilidad_X", "S1_adaptabilidad_Y"),
    ("S1_curiosidad_X", "S1_curiosidad_Y"),
]
S2_COLS = [
    ("S2_mi_experiencia_X", "S2_mi_experiencia_Y"),
    ("S2_percepcion_mercado_X", "S2_percepcion_mercado_Y"),
    ("S2_como_nos_vendemos_X", "S2_como_nos_vendemos_Y"),
    ("S2_como_quisiera_X", "S2_como_quisiera_Y"),
]

def build_embedded_data(triads_df, dyads_df, stones_df, cat_df, cluster_col):
    """Export row-level data as compact integer arrays for client-side filtering."""
    n = len(triads_df)
    # Triad columns: 9 triads × 3 apexes = 27 cols
    triad_cols = []
    for tid in TRIAD_IDS:
        triad_cols.extend([f"{tid}_a", f"{tid}_b", f"{tid}_c"])
    # Dyad columns: 6
    dyad_cols = DYAD_IDS
    # Stone columns: S1 (8×2=16) + S2 (4×2=8) = 24
    stone_cols = []
    for xc, yc in S1_COLS + S2_COLS:
        stone_cols.extend([xc, yc])

    # Encode as int ×10000 for precision (4 decimals)
    def to_int_array(df, cols):
        result = []
        for _, row in df.iterrows():
            result.append([int(round(float(row[c]) * 10000)) for c in cols])
        return result

    triads_arr = to_int_array(triads_df, triad_cols)
    dyads_arr = to_int_array(dyads_df, dyad_cols)
    stones_arr = to_int_array(stones_df, stone_cols)

    # Demographics: encode as integer indices
    demo_maps = {}
    demos_arr = []
    for col in DEMO_FIELDS:
        if col in cat_df.columns:
            unique_vals = sorted(cat_df[col].dropna().unique().tolist())
            demo_maps[col] = unique_vals
        else:
            demo_maps[col] = []

    for _, row in cat_df.iterrows():
        row_demos = []
        for col in DEMO_FIELDS:
            if col in cat_df.columns:
                val = row[col]
                try:
                    idx = demo_maps[col].index(val)
                except (ValueError, KeyError):
                    idx = -1
                row_demos.append(idx)
            else:
                row_demos.append(-1)
        demos_arr.append(row_demos)

    clusters = [int(cluster_col.iloc[i]) for i in range(n)]

    return {
        'n': n,
        'triadCols': triad_cols,
        'dyadCols': dyad_cols,
        'stoneCols': stone_cols,
        'demoFields': DEMO_FIELDS,
        'demoMaps': demo_maps,
        'cluster': clusters,
        'triads': triads_arr,
        'dyads': dyads_arr,
        'stones': stones_arr,
        'demos': demos_arr
    }


def generate_synthetic_2026(triads_df, dyads_df, stones_df, cat_df, cluster_col):
    """Generate synthetic 2026 data with realistic moderate improvements."""
    t26 = triads_df.copy()
    d26 = dyads_df.copy()
    s26 = stones_df.copy()
    c26 = cat_df.copy()
    cl26 = cluster_col.copy()

    np.random.seed(2026)

    # --- Cluster migration: ~5% of C1 → C2, ~5% of C2 → C3 ---
    c1_mask = cl26 == 1
    c1_indices = cl26[c1_mask].index.tolist()
    migrate_c1_to_c2 = np.random.choice(c1_indices, size=min(52, len(c1_indices)), replace=False)
    cl26.loc[migrate_c1_to_c2] = 2

    c2_mask = cl26 == 2
    c2_indices = cl26[c2_mask].index.tolist()
    migrate_c2_to_c3 = np.random.choice(c2_indices, size=min(18, len(c2_indices)), replace=False)
    cl26.loc[migrate_c2_to_c3] = 3

    # --- Dyad shifts toward positive (lower values = more positive pole) ---
    dyad_shifts = {'D1': -0.04, 'D4': -0.05, 'D5': -0.02, 'D6': -0.06}
    for did, shift in dyad_shifts.items():
        noise = np.random.normal(0, 0.02, len(d26))
        d26[did] = (d26[did] + shift + noise).clip(0, 1)

    # --- Stone improvements: X=0 means "frequent", so DECREASE = improvement ---
    # Y=0 means "easy", so DECREASE = improvement
    s26['S1_ser_yo_mismo_X'] = (s26['S1_ser_yo_mismo_X'] - 0.08 + np.random.normal(0, 0.02, len(s26))).clip(0, 1)
    s26['S1_ser_yo_mismo_Y'] = (s26['S1_ser_yo_mismo_Y'] - 0.04 + np.random.normal(0, 0.02, len(s26))).clip(0, 1)
    # General small improvement on other S1 items (more frequent + easier)
    for xc, yc in S1_COLS[1:]:
        s26[xc] = (s26[xc] - 0.03 + np.random.normal(0, 0.015, len(s26))).clip(0, 1)
        s26[yc] = (s26[yc] - 0.02 + np.random.normal(0, 0.015, len(s26))).clip(0, 1)

    # --- Triad shifts: positive apex +0.03, others re-normalized ---
    for tid in TRIAD_IDS:
        a_col, b_col, c_col = f"{tid}_a", f"{tid}_b", f"{tid}_c"
        t26[a_col] = t26[a_col] + 0.03 + np.random.normal(0, 0.01, len(t26))
        t26[b_col] = t26[b_col] - 0.015 + np.random.normal(0, 0.01, len(t26))
        t26[c_col] = t26[c_col] - 0.015 + np.random.normal(0, 0.01, len(t26))
        # Re-normalize to sum=1
        row_sums = t26[a_col] + t26[b_col] + t26[c_col]
        row_sums = row_sums.replace(0, 1)
        t26[a_col] = (t26[a_col] / row_sums).clip(0, 1)
        t26[b_col] = (t26[b_col] / row_sums).clip(0, 1)
        t26[c_col] = (t26[c_col] / row_sums).clip(0, 1)

    return t26, d26, s26, c26, cl26


print("[Report] Building embedded data for interactive filtering...")
data_2025 = build_embedded_data(triads, dyads, stones, categorical, assignments['cluster'])

t26, d26, s26, c26, cl26 = generate_synthetic_2026(triads, dyads, stones, categorical, assignments['cluster'])
# Add cluster column to 2026 DataFrames for component generation
s26['cluster'] = cl26.values
d26['cluster'] = cl26.values
t26['cluster'] = cl26.values
c26['cluster'] = cl26.values
data_2026 = build_embedded_data(t26, d26, s26, c26, cl26)

# Build 2026 profiles (triad means per cluster)
triad_cols_all = [c for c in t26.columns if c.startswith('T') and c != 'cluster']
profiles_26 = t26.groupby('cluster')[triad_cols_all].mean()

embedded_json = json.dumps({'2025': data_2025, '2026': data_2026}, separators=(',', ':'))
print(f"  Embedded data size: {len(embedded_json) / 1024:.0f} KB")

# ── Constants ────────────────────────────────────────────────────────────
COLORS = {1: '#EF4444', 2: '#F59E0B', 3: '#10B981'}
NAMES_ES = {
    1: 'Los Escépticos Prudentes',
    2: 'Los Constructores Pragmáticos',
    3: 'Los Visionarios Comprometidos'
}
NAMES_EN = {
    1: 'The Cautious Skeptics',
    2: 'The Pragmatic Builders',
    3: 'The Committed Visionaries'
}
ICONS = {1: '&#128269;', 2: '&#9889;', 3: '&#127793;'}
SIZES = {cl: int((triads['cluster'] == cl).sum()) for cl in [1, 2, 3]}
PCTS = {cl: round(SIZES[cl] / N * 100, 1) for cl in [1, 2, 3]}

# ── 2026 Aggregate Statistics (for year-toggled text) ────────────────────
SIZES_26 = {cl: int((cl26 == cl).sum()) for cl in [1, 2, 3]}
PCTS_26 = {cl: round(SIZES_26[cl] / N * 100, 1) for cl in [1, 2, 3]}
C2C3_pct_25 = round(PCTS[2] + PCTS[3], 1)
C2C3_pct_26 = round(PCTS_26[2] + PCTS_26[3], 1)

# Key dyad medians
cl_25 = assignments['cluster']
d6_c1_25 = round(dyads.loc[cl_25 == 1, 'D6'].median(), 2)
d6_c3_25 = round(dyads.loc[cl_25 == 3, 'D6'].median(), 2)
d6_c1_26 = round(d26.loc[cl26 == 1, 'D6'].median(), 2)
d6_c3_26 = round(d26.loc[cl26 == 3, 'D6'].median(), 2)
d6_range_25 = round(d6_c1_25 - d6_c3_25, 2)
d6_range_26 = round(d6_c1_26 - d6_c3_26, 2)

d4_c1_25 = round(dyads.loc[cl_25 == 1, 'D4'].median(), 2)
d4_c1_26 = round(d26.loc[cl26 == 1, 'D4'].median(), 2)
d4_c3_25 = round(dyads.loc[cl_25 == 3, 'D4'].median(), 2)
d4_c3_26 = round(d26.loc[cl26 == 3, 'D4'].median(), 2)

d5_overall_25 = round(dyads['D5'].median(), 2)
d5_overall_26 = round(d26['D5'].median(), 2)
d5_c1_25 = round(dyads.loc[cl_25 == 1, 'D5'].median(), 2)
d5_c1_26 = round(d26.loc[cl26 == 1, 'D5'].median(), 2)

d1_fear_pct_25 = round((dyads.loc[cl_25 == 1, 'D1'] > 0.6).mean() * 100, 1)
d1_fear_pct_26 = round((d26.loc[cl26 == 1, 'D1'] > 0.6).mean() * 100, 1)
d1_fear_c3_25 = round((dyads.loc[cl_25 == 3, 'D1'] > 0.6).mean() * 100, 1)
d1_fear_c3_26 = round((d26.loc[cl26 == 3, 'D1'] > 0.6).mean() * 100, 1)

# Risk thresholds (2026)
fear_pct_26 = round((d26['D1'] > 0.6).mean() * 100, 0)
imposed_pct_26 = round((d26['D4'] > 0.6).mean() * 100, 0)
belong_pct_26 = round((d26['D5'] > 0.5).mean() * 100, 0)

# Key triad values
t4a_c1_25 = round(triads.loc[cl_25 == 1, 'T4_a'].mean() * 100, 1)
t4a_c1_26 = round(t26.loc[cl26 == 1, 'T4_a'].mean() * 100, 1)
t4a_c3_25 = round(triads.loc[cl_25 == 3, 'T4_a'].mean() * 100, 1)
t4a_c3_26 = round(t26.loc[cl26 == 3, 'T4_a'].mean() * 100, 1)
t4a_gap_25 = round(t4a_c3_25 - t4a_c1_25, 1)
t4a_gap_26 = round(t4a_c3_26 - t4a_c1_26, 1)

t1_c1_resign_25 = round(triads.loc[cl_25 == 1, 'T1_a'].mean() * 100, 0)
t1_c1_resign_26 = round(t26.loc[cl26 == 1, 'T1_a'].mean() * 100, 0)

# Stone KPIs
s1_x_25 = round(stones['S1_ser_yo_mismo_X'].mean(), 2)
s1_x_26 = round(s26['S1_ser_yo_mismo_X'].mean(), 2)
s1_y_25 = round(stones['S1_ser_yo_mismo_Y'].mean(), 2)
s1_y_26 = round(s26['S1_ser_yo_mismo_Y'].mean(), 2)

# Compound risk (2025 + 2026)
compound_risk = int(((cl_25 == 1) & (dyads['D4'] > 0.5) & (dyads['D5'] > 0.4)).sum())
compound_pct = compound_risk / N
compound_mask_26 = (cl26 == 1) & (d26['D4'] > 0.5) & (d26['D5'] > 0.4)
compound_risk_26 = int(compound_mask_26.sum())
compound_pct_26 = compound_risk_26 / N

# D6 disconnection (>0.6) pcts
d6_disc_c1_25 = round((dyads.loc[cl_25 == 1, 'D6'] > 0.6).mean() * 100, 0)
d6_disc_c1_26 = round((d26.loc[cl26 == 1, 'D6'] > 0.6).mean() * 100, 0)
d6_disc_c3_25 = round((dyads.loc[cl_25 == 3, 'D6'] > 0.6).mean() * 100, 0)
d6_disc_c3_26 = round((d26.loc[cl26 == 3, 'D6'] > 0.6).mean() * 100, 0)

print(f"  2025: C1={SIZES[1]}({PCTS[1]}%), C2={SIZES[2]}({PCTS[2]}%), C3={SIZES[3]}({PCTS[3]}%)")
print(f"  2026: C1={SIZES_26[1]}({PCTS_26[1]}%), C2={SIZES_26[2]}({PCTS_26[2]}%), C3={SIZES_26[3]}({PCTS_26[3]}%)")
print(f"  Stone fix: S1_X 2025={s1_x_25} → 2026={s1_x_26} (lower=more frequent=better)")
print(f"  Compound risk: 2025={compound_risk} → 2026={compound_risk_26}")

# ── Helper Functions ─────────────────────────────────────────────────────

def bi(es, en):
    """Generate bilingual span."""
    return f'<span class="lang-es">{es}</span><span class="lang-en">{en}</span>'

def yr(text_2025, text_2026):
    """Generate year-toggle spans (mirrors bilingual pattern)."""
    return f'<span class="y25">{text_2025}</span><span class="y26">{text_2026}</span>'

def yr_block(html_2025, html_2026):
    """Generate year-toggle DIVs for block-level content (charts, tables, profiles)."""
    return f'<div class="y25">{html_2025}</div><div class="y26">{html_2026}</div>'

def make_ternary(tid):
    """Create a Plotly ternary scatter for a triad."""
    td = label_dict['triads'][tid]
    a_col = f"{tid}_a"
    b_col = f"{tid}_b"
    c_col = f"{tid}_c"

    a_es = td['es']['apex_a']
    b_es = td['es']['apex_b']
    c_es = td['es']['apex_c']

    fig = go.Figure()
    for cl in [1, 2, 3]:
        mask = triads['cluster'] == cl
        fig.add_trace(go.Scatterternary(
            a=triads.loc[mask, a_col],
            b=triads.loc[mask, b_col],
            c=triads.loc[mask, c_col],
            mode='markers',
            marker=dict(size=5, color=COLORS[cl], opacity=0.45),
            name=f'C{cl} {NAMES_ES[cl]}',
            hovertemplate=f'{a_es}: %{{a:.2f}}<br>{b_es}: %{{b:.2f}}<br>{c_es}: %{{c:.2f}}<extra>C{cl}</extra>'
        ))
        # Centroid
        ca = triads.loc[mask, a_col].mean()
        cb = triads.loc[mask, b_col].mean()
        cc = triads.loc[mask, c_col].mean()
        fig.add_trace(go.Scatterternary(
            a=[ca], b=[cb], c=[cc],
            mode='markers',
            marker=dict(size=14, symbol='diamond', color=COLORS[cl],
                       line=dict(width=2, color='white')),
            name=f'C{cl} centroide',
            showlegend=False,
            hovertemplate=f'Centroide C{cl}<br>{a_es}: {ca:.3f}<br>{b_es}: {cb:.3f}<br>{c_es}: {cc:.3f}<extra></extra>'
        ))

    fig.update_layout(
        ternary=dict(
            aaxis=dict(title=a_es, min=0),
            baxis=dict(title=b_es, min=0),
            caxis=dict(title=c_es, min=0),
        ),
        height=380, margin=dict(l=40, r=40, t=30, b=30),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Montserrat', size=11)
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=f'ternary_{tid}')

def make_dyad_html(did):
    """Create CSS gauge-style dyad visualization."""
    dd = label_dict['dyads'][did]
    pole_l_es = dd['es']['pole_left']
    pole_r_es = dd['es']['pole_right']
    pole_l_en = dd['en']['pole_left']
    pole_r_en = dd['en']['pole_right']
    label_es = dd['es']['label']
    label_en = dd['en']['label']

    col = did
    overall_median = float(dyads[col].median())

    rows_html = ""
    for cl in [1, 2, 3]:
        mask = dyads['cluster'] == cl
        val = float(dyads.loc[mask, col].median())
        pct = val * 100
        color = COLORS[cl]
        name_es = NAMES_ES[cl]

        rows_html += f"""
        <div class="dyad-cluster-row">
            <div class="dyad-cluster-label">
                <span class="dyad-cluster-dot" style="background:{color}"></span>
                <span>C{cl} · {bi(name_es, NAMES_EN[cl])}</span>
            </div>
            <div class="dyad-gauge-track">
                <div class="dyad-gauge-fill" style="width:{pct:.1f}%; background: linear-gradient(90deg, {color}26, {color}73);">
                    <div class="dyad-gauge-marker" style="background:{color};"></div>
                    <div class="dyad-gauge-value">{val:.2f}</div>
                </div>
            </div>
        </div>"""

    return f"""
    <div class="dyad-card">
        <h4>{bi(f'{did}: {label_es}', f'{did}: {label_en}')}</h4>
        <div class="dyad-poles">
            <span class="pole-left">{bi(f'← {pole_l_es}', f'← {pole_l_en}')}</span>
            <span class="pole-right">{bi(f'{pole_r_es} →', f'{pole_r_en} →')}</span>
        </div>
        {rows_html}
        <div class="dyad-scale">
            <span>0</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1.0</span>
        </div>
        <div class="dyad-overall">
            <span>{bi('Mediana general:', 'Overall median:')}</span> <strong>{overall_median:.2f}</strong>
        </div>
    </div>"""

def make_demo_table(col_name, title_es, title_en, cat_data=None):
    """Create a demographic crosstab HTML table."""
    if cat_data is None:
        cat_data = categorical
    if col_name not in cat_data.columns:
        return ""

    ct = pd.crosstab(cat_data[col_name], cat_data['cluster'], normalize='columns') * 100
    ct = ct.round(1)

    rows = ""
    for val in ct.index:
        cells = "".join([f"<td>{ct.loc[val, cl]:.1f}%</td>" for cl in sorted(ct.columns)])
        rows += f"<tr><td>{val}</td>{cells}</tr>"

    headers = "".join([f"<th style='color:{COLORS[cl]}'>C{cl}</th>" for cl in sorted(ct.columns)])

    return f"""
    <div class="demo-table-container">
        <h4>{bi(title_es, title_en)}</h4>
        <table class="data-table">
            <thead><tr><th>{bi('Categoría', 'Category')}</th>{headers}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>"""

def make_stone_scatter(stone_set, items, x_label_es, y_label_es, x_label_en, y_label_en,
                       quadrant_labels=None, quadrant_colors=None,
                       x_endpoints=None, y_endpoints=None,
                       stone_data=None, div_suffix=''):
    """Create a clean scatter plot for stone items per cluster using numbered markers.

    quadrant_labels: dict with keys 'bl','br','tl','tr' -> bilingual label strings
    quadrant_colors: dict with keys 'bl','br','tl','tr' -> rgba color strings
    x_endpoints: tuple (label_at_0, label_at_1) for axis extreme annotations
    y_endpoints: tuple (label_at_0, label_at_1) for axis extreme annotations
    stone_data: DataFrame to use (default: global `stones`)
    div_suffix: suffix for div_id to avoid duplicate IDs (e.g. '_26')
    """
    if stone_data is None:
        stone_data = stones
    fig = go.Figure()

    # Default quadrant colors
    default_colors = {
        'tl': "rgba(16,185,129,0.04)", 'tr': "rgba(245,158,11,0.04)",
        'bl': "rgba(107,114,128,0.04)", 'br': "rgba(239,68,68,0.04)"
    }
    qc = quadrant_colors if quadrant_colors else default_colors

    # Add quadrant shading
    fig.add_shape(type="rect", x0=0, y0=0.5, x1=0.5, y1=1, fillcolor=qc['tl'], line_width=0)
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=1, y1=1, fillcolor=qc['tr'], line_width=0)
    fig.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=0.5, fillcolor=qc['bl'], line_width=0)
    fig.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=0.5, fillcolor=qc['br'], line_width=0)

    # Crosshair lines at 0.5
    fig.add_hline(y=0.5, line_dash="dot", line_color="#D1D5DB", line_width=1)
    fig.add_vline(x=0.5, line_dash="dot", line_color="#D1D5DB", line_width=1)

    # Quadrant labels
    if quadrant_labels:
        q_positions = [
            (0.25, 0.25, 'bl'), (0.75, 0.25, 'br'),
            (0.25, 0.75, 'tl'), (0.75, 0.75, 'tr'),
        ]
        for qx, qy, qkey in q_positions:
            if qkey in quadrant_labels:
                fig.add_annotation(
                    x=qx, y=qy, text=quadrant_labels[qkey],
                    showarrow=False, xanchor='center', yanchor='middle',
                    font=dict(size=8.5, color='#9CA3AF', family='Montserrat'),
                    opacity=0.55
                )

    # Axis endpoint annotations (original questionnaire words)
    if x_endpoints:
        fig.add_annotation(x=0.04, y=-0.06, text=x_endpoints[0], showarrow=False,
                          xanchor='left', yanchor='top',
                          font=dict(size=7, color='#9CA3AF', family='Montserrat'),
                          opacity=0.7, xref='x', yref='paper')
        fig.add_annotation(x=0.96, y=-0.06, text=x_endpoints[1], showarrow=False,
                          xanchor='right', yanchor='top',
                          font=dict(size=7, color='#9CA3AF', family='Montserrat'),
                          opacity=0.7, xref='x', yref='paper')
    if y_endpoints:
        fig.add_annotation(x=-0.06, y=0.04, text=y_endpoints[0], showarrow=False,
                          yanchor='bottom', textangle=-90,
                          font=dict(size=7, color='#9CA3AF', family='Montserrat'),
                          opacity=0.7, xref='paper', yref='y')
        fig.add_annotation(x=-0.06, y=0.96, text=y_endpoints[1], showarrow=False,
                          yanchor='top', textangle=-90,
                          font=dict(size=7, color='#9CA3AF', family='Montserrat'),
                          opacity=0.7, xref='paper', yref='y')

    cluster_symbols = {1: 'circle', 2: 'square', 3: 'diamond'}

    for cl in [1, 2, 3]:
        mask = stone_data['cluster'] == cl
        x_vals = []
        y_vals = []
        labels = []
        hovers = []

        for i, (item_name, x_col, y_col) in enumerate(items):
            x_mean = float(stone_data.loc[mask, x_col].mean())
            y_mean = float(stone_data.loc[mask, y_col].mean())
            x_vals.append(x_mean)
            y_vals.append(y_mean)
            labels.append(str(i + 1))
            hovers.append(f'{item_name}<br>X: {x_mean:.3f}<br>Y: {y_mean:.3f}')

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers+text',
            marker=dict(size=22, color=COLORS[cl], opacity=0.85,
                       symbol=cluster_symbols[cl],
                       line=dict(width=1.5, color='white')),
            text=labels,
            textfont=dict(size=9, color='white', family='Montserrat'),
            textposition='middle center',
            name=f'C{cl} · {NAMES_ES[cl]}',
            hovertemplate='%{customdata}<extra>C' + str(cl) + '</extra>',
            customdata=hovers
        ))

    fig.update_layout(
        xaxis=dict(title=x_label_es, range=[-0.02, 1.02], dtick=0.25,
                  gridcolor='#F3F4F6', zeroline=False),
        yaxis=dict(title=y_label_es, range=[-0.02, 1.02], dtick=0.25,
                  gridcolor='#F3F4F6', zeroline=False),
        height=420, margin=dict(l=55, r=20, t=25, b=55),
        showlegend=True,
        legend=dict(x=0.01, y=-0.18, orientation='h',
                   bgcolor='rgba(255,255,255,0)', font=dict(size=10)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,1)',
        font=dict(family='Montserrat', size=11)
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=f'stone_{stone_set}{div_suffix}')

def make_stone_legend_table(items, stone_data=None):
    """Create an HTML legend table for stone numbered markers."""
    if stone_data is None:
        stone_data = stones
    rows = ""
    for i, (item_name, x_col, y_col) in enumerate(items):
        # Per-cluster means
        cells = ""
        for cl in [1, 2, 3]:
            mask = stone_data['cluster'] == cl
            xm = float(stone_data.loc[mask, x_col].mean())
            ym = float(stone_data.loc[mask, y_col].mean())
            cells += f'<td style="font-size:0.72rem;">{xm:.2f}, {ym:.2f}</td>'
        rows += f'<tr><td><strong>{i+1}</strong></td><td>{item_name}</td>{cells}</tr>'

    return f"""
    <table class="data-table stone-legend">
        <thead>
            <tr>
                <th>#</th>
                <th>{bi('Ítem', 'Item')}</th>
                <th style="color:var(--c1)">C1 (X,Y)</th>
                <th style="color:var(--c2)">C2 (X,Y)</th>
                <th style="color:var(--c3)">C3 (X,Y)</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>"""

# ── Generate Components ──────────────────────────────────────────────────
print("[Report] Generating ternary plots...")
triad_sections = ""
triad_ids = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]

for tid in triad_ids:
    td = label_dict['triads'][tid]
    chart = make_ternary(tid)

    # Compute zone data (which apex dominates)
    a_col, b_col, c_col = f"{tid}_a", f"{tid}_b", f"{tid}_c"
    zone_data = ""
    for cl in [1, 2, 3]:
        mask = triads['cluster'] == cl
        ma = triads.loc[mask, a_col].mean()
        mb = triads.loc[mask, b_col].mean()
        mc = triads.loc[mask, c_col].mean()
        zone_data += f"""
        <div class="zone-row">
            <span class="zone-label" style="color:{COLORS[cl]}">C{cl}</span>
            <div class="zone-bars">
                <div class="zone-bar" style="width:{ma*100:.0f}%; background:{COLORS[cl]}88;" title="{td['es']['apex_a']}: {ma:.1%}"></div>
            </div>
            <span class="zone-val">{ma:.0%} / {mb:.0%} / {mc:.0%}</span>
        </div>"""

    triad_sections += f"""
    <div class="triad-section">
        <h3>{bi(f"{tid}: {td['es']['label']}", f"{tid}: {td['en']['label']}")}</h3>
        <p class="triad-question">{bi(td['es']['question_stem'], td['en'].get('question_stem', ''))}</p>
        <div class="triad-grid">
            <div class="chart-container">{chart}</div>
            <div class="triad-data">
                <h4>{bi('Distribución por cluster', 'Distribution by cluster')}</h4>
                <div class="triad-apexes">
                    <span class="apex-tag">A: {td['es']['apex_a']}</span>
                    <span class="apex-tag">B: {td['es']['apex_b']}</span>
                    <span class="apex-tag">C: {td['es']['apex_c']}</span>
                </div>
                {zone_data}
            </div>
        </div>
    </div>"""

print("[Report] Generating dyad sections...")
dyad_sections = ""
for did in ["D1", "D2", "D3", "D4", "D5", "D6"]:
    dyad_sections += make_dyad_html(did)

print("[Report] Generating stone visualizations...")
# S1: Psychological Safety (8 items)
s1_items = [
    ("Ser yo mismo", "S1_ser_yo_mismo_X", "S1_ser_yo_mismo_Y"),
    ("Aceptar errores", "S1_aceptar_errores_X", "S1_aceptar_errores_Y"),
    ("Pedir ayuda", "S1_pedir_ayuda_X", "S1_pedir_ayuda_Y"),
    ("Confianza", "S1_confianza_X", "S1_confianza_Y"),
    ("Empatía", "S1_empatia_X", "S1_empatia_Y"),
    ("Simplicidad", "S1_simplicidad_X", "S1_simplicidad_Y"),
    ("Adaptabilidad", "S1_adaptabilidad_X", "S1_adaptabilidad_Y"),
    ("Curiosidad", "S1_curiosidad_X", "S1_curiosidad_Y"),
]
s1_quadrant_labels = {
    'bl': 'Integrado\nIntegrated',               # frecuente + fácil = zona saludable
    'br': 'Posible no vivido\nPossible not lived', # raro + fácil = potencial sin activar
    'tl': 'Vivido con esfuerzo\nLived with effort', # frecuente + difícil = tensión
    'tr': 'Bloqueado\nBlocked'                     # raro + difícil = zona de riesgo
}
s1_quadrant_colors = {
    'bl': "rgba(16,185,129,0.06)",   # verde — zona saludable
    'br': "rgba(245,158,11,0.06)",   # ámbar — potencial sin activar
    'tl': "rgba(59,130,246,0.05)",   # azul — tensión/esfuerzo
    'tr': "rgba(107,114,128,0.06)"   # gris — zona de riesgo
}
s1_chart_25 = make_stone_scatter("S1", s1_items,
    "Pasa todo el tiempo ← Frecuencia → Es muy raro",
    "Muy fácil ← Dificultad → Imposible",
    "Happens all the time ← Frequency → Very rare",
    "Very easy ← Difficulty → Impossible",
    quadrant_labels=s1_quadrant_labels, quadrant_colors=s1_quadrant_colors,
    x_endpoints=("Pasa todo el tiempo", "Es muy raro"),
    y_endpoints=("Muy fácil", "Imposible"))
s1_chart_26 = make_stone_scatter("S1", s1_items,
    "Pasa todo el tiempo ← Frecuencia → Es muy raro",
    "Muy fácil ← Dificultad → Imposible",
    "Happens all the time ← Frequency → Very rare",
    "Very easy ← Difficulty → Impossible",
    quadrant_labels=s1_quadrant_labels, quadrant_colors=s1_quadrant_colors,
    x_endpoints=("Pasa todo el tiempo", "Es muy raro"),
    y_endpoints=("Muy fácil", "Imposible"),
    stone_data=s26, div_suffix='_26')
s1_chart = yr_block(s1_chart_25, s1_chart_26)
s1_legend_25 = make_stone_legend_table(s1_items)
s1_legend_26 = make_stone_legend_table(s1_items, stone_data=s26)
s1_legend = yr_block(s1_legend_25, s1_legend_26)

# S2: Brand & Identity (4 items)
s2_items = [
    ("Mi experiencia", "S2_mi_experiencia_X", "S2_mi_experiencia_Y"),
    ("Percepción mercado", "S2_percepcion_mercado_X", "S2_percepcion_mercado_Y"),
    ("Cómo nos vendemos", "S2_como_nos_vendemos_X", "S2_como_nos_vendemos_Y"),
    ("Como quisiera", "S2_como_quisiera_X", "S2_como_quisiera_Y"),
]
s2_quadrant_labels = {
    'bl': 'Alineado\nAligned',                # frecuente + fácil = coherente
    'br': 'Aspiracional\nAspirational',        # raro + fácil = deseado no vivido
    'tl': 'Brecha vivida\nLived gap',          # frecuente + difícil = tensión
    'tr': 'Desconectado\nDisconnected'         # raro + difícil = sin alineación
}
s2_quadrant_colors = {
    'bl': "rgba(16,185,129,0.06)",   # verde — coherencia
    'br': "rgba(245,158,11,0.06)",   # ámbar — aspiración
    'tl': "rgba(239,68,68,0.05)",    # rojo suave — tensión
    'tr': "rgba(107,114,128,0.06)"   # gris — desapego
}
s2_chart_25 = make_stone_scatter("S2", s2_items,
    "Pasa todo el tiempo ← Frecuencia → Es muy raro",
    "Muy fácil ← Dificultad → Imposible",
    "Happens all the time ← Frequency → Very rare",
    "Very easy ← Difficulty → Impossible",
    quadrant_labels=s2_quadrant_labels, quadrant_colors=s2_quadrant_colors,
    x_endpoints=("Pasa todo el tiempo", "Es muy raro"),
    y_endpoints=("Muy fácil", "Imposible"))
s2_chart_26 = make_stone_scatter("S2", s2_items,
    "Pasa todo el tiempo ← Frecuencia → Es muy raro",
    "Muy fácil ← Dificultad → Imposible",
    "Happens all the time ← Frequency → Very rare",
    "Very easy ← Difficulty → Impossible",
    quadrant_labels=s2_quadrant_labels, quadrant_colors=s2_quadrant_colors,
    x_endpoints=("Pasa todo el tiempo", "Es muy raro"),
    y_endpoints=("Muy fácil", "Imposible"),
    stone_data=s26, div_suffix='_26')
s2_chart = yr_block(s2_chart_25, s2_chart_26)
s2_legend_25 = make_stone_legend_table(s2_items)
s2_legend_26 = make_stone_legend_table(s2_items, stone_data=s26)
s2_legend = yr_block(s2_legend_25, s2_legend_26)

print("[Report] Building cluster profiles...")

def build_profiles_html(prof_data, sizes_dict, pcts_dict):
    """Build cluster profile HTML using given profile data and sizes."""
    html = ""
    for cl in [1, 2, 3]:
        # Get text analysis data (same for both years — qualitative)
        exp_uni = text_analysis.get("unigrams", {}).get("experiencia", {}).get(f"cluster_{cl}", {})
        distinctive = exp_uni.get("distinctive", {})
        top_words = list(distinctive.keys())[:5]
        top_words_str = ", ".join(top_words) if top_words else "N/A"

        # Get quotes
        quotes_list = text_analysis.get("quotes", {}).get(str(cl), [])
        quotes_html = ""
        for q in quotes_list[:4]:
            quotes_html += f'<div class="quote-card" style="border-left-color:{COLORS[cl]}"><em>"{q["text"][:300]}"</em></div>'

        # Presidente quotes
        pres_quotes = text_analysis.get("quotes_presidente", {}).get(str(cl), [])
        pres_html = ""
        for q in pres_quotes[:2]:
            pres_html += f'<div class="quote-card" style="border-left-color:{COLORS[cl]}"><em>"{q["text"][:300]}"</em></div>'

        # Metaphors
        metaphor_html = ""
        for mf_key in ["cultura_metafora", "trabajo_metafora", "cambio_metafora"]:
            mf_data = text_analysis.get("metaphor_summary", {}).get(mf_key, {}).get(f"cluster_{cl}", {})
            top_m = mf_data.get("top_10", [])[:3]
            if top_m:
                mf_label = {"cultura_metafora": "Cultura", "trabajo_metafora": "Trabajo", "cambio_metafora": "Cambio"}[mf_key]
                mf_vals = ", ".join([f'<em>"{m}"</em> ({c})' for m, c in top_m])
                metaphor_html += f'<div class="metaphor-row"><strong>{mf_label}:</strong> {mf_vals}</div>'

        # Key traits from profiles
        key_triads = []
        for tid in triad_ids:
            td = label_dict['triads'][tid]
            a, b, c = f"{tid}_a", f"{tid}_b", f"{tid}_c"
            vals = [prof_data.loc[cl, a], prof_data.loc[cl, b], prof_data.loc[cl, c]]
            max_idx = np.argmax(vals)
            max_val = vals[max_idx]
            apex_names = [td['es']['apex_a'], td['es']['apex_b'], td['es']['apex_c']]
            if max_val > 0.5:
                key_triads.append(f"<li>{tid}: <strong>{apex_names[max_idx]}</strong> ({max_val:.0%})</li>")
        traits_es = "\n".join(key_triads[:6])

        html += f"""
        <div class="cluster-profile" style="border-left: 5px solid {COLORS[cl]}">
            <div class="cluster-header">
                <span class="cluster-icon">{ICONS[cl]}</span>
                <div>
                    <h3 style="color:{COLORS[cl]}">C{cl}: {bi(NAMES_ES[cl], NAMES_EN[cl])}</h3>
                    <div class="cluster-meta">n = {sizes_dict[cl]} ({pcts_dict[cl]}%)</div>
                </div>
            </div>
            <div class="cluster-body">
                <div class="cluster-desc">
                    <h4>{bi('Rasgos dominantes', 'Dominant traits')}</h4>
                    <ul>{traits_es}</ul>
                    <h4>{bi('Vocabulario distintivo', 'Distinctive vocabulary')}</h4>
                    <p class="distinctive-words">{top_words_str}</p>
                </div>
                <div class="cluster-metaphors">
                    <h4>{bi('Metáforas', 'Metaphors')}</h4>
                    {metaphor_html}
                </div>
            </div>
            <div class="cluster-quotes">
                <h4>{bi('Voces representativas', 'Representative voices')}</h4>
                {quotes_html}
            </div>
            <div class="cluster-presidente">
                <h4>{bi('Carta al presidente', 'Letter to the president')}</h4>
                {pres_html}
            </div>
        </div>"""
    return html

profiles_html_25 = build_profiles_html(profiles, SIZES, PCTS)
profiles_html_26 = build_profiles_html(profiles_26, SIZES_26, PCTS_26)
cluster_profiles_html = yr_block(profiles_html_25, profiles_html_26)

print("[Report] Building demographics...")
demo_cols = {
    "cargo": ("Cargo / Rol", "Role / Position"),
    "antiguedad": ("Antigüedad", "Seniority"),
    "area": ("Área", "Department"),
    "genero": ("Género", "Gender"),
    "educacion": ("Nivel educativo", "Education level"),
}
demo_tables_25 = ""
demo_tables_26 = ""
for col, (es, en) in demo_cols.items():
    demo_tables_25 += make_demo_table(col, es, en)
    demo_tables_26 += make_demo_table(col, es, en, cat_data=c26)
demo_tables = yr_block(demo_tables_25, demo_tables_26)

# Likert analysis
likert_cols = {}
for lk_id in ["L1", "L2", "L3", "L4"]:
    lk_data = label_dict.get("likert", {}).get(lk_id, {})
    if isinstance(lk_data, dict):
        lk_label = lk_data.get("es", lk_id)
        if isinstance(lk_label, dict):
            lk_label = lk_label.get("label", lk_id)
        # Truncate long labels
        if len(str(lk_label)) > 80:
            lk_label = str(lk_label)[:77] + "..."
        likert_cols[lk_id] = lk_label
    else:
        likert_cols[lk_id] = str(lk_data)[:80] if lk_data else lk_id
likert_html_25 = ""
likert_html_26 = ""
for lk, lk_label in likert_cols.items():
    if lk in categorical.columns:
        likert_html_25 += make_demo_table(lk, lk_label, lk_label)
        likert_html_26 += make_demo_table(lk, lk_label, lk_label, cat_data=c26)
likert_html = yr_block(likert_html_25, likert_html_26)

# ── Compute Risk Factors ─────────────────────────────────────────────────
print("[Report] Computing risk factors...")

def build_risk_chart(cat_data, div_suffix=''):
    """Build a diverging risk bar chart from categorical data with cluster column."""
    c1_pct_global = (cat_data['cluster'] == 1).mean()
    r_data = []
    for col, (title_es, title_en) in demo_cols.items():
        if col not in cat_data.columns:
            continue
        ct = pd.crosstab(cat_data[col], cat_data['cluster'], normalize='index')
        ct_n = pd.crosstab(cat_data[col], cat_data['cluster']).sum(axis=1)
        for val in ct.index:
            n_val = int(ct_n[val])
            if n_val < 20:
                continue
            c1_p = float(ct.loc[val, 1]) if 1 in ct.columns else 0
            rr = c1_p / c1_pct_global if c1_pct_global > 0 else 0
            r_data.append({
                'category': col, 'value': val, 'n': n_val,
                'c1_pct': c1_p, 'risk_ratio': rr,
                'title_es': title_es, 'title_en': title_en
            })

    chart_data = sorted(r_data, key=lambda x: x['risk_ratio'])
    fig_r = go.Figure()
    for r in chart_data:
        if r['n'] < 25:
            continue
        offset = r['risk_ratio'] - 1.0
        color = '#EF4444' if offset > 0.1 else ('#F59E0B' if offset > 0 else '#10B981')
        label = f"{r['value']} ({r['title_es']})"
        if len(label) > 45:
            label = label[:42] + "..."
        fig_r.add_trace(go.Bar(
            y=[label], x=[offset],
            orientation='h',
            marker_color=color,
            hovertemplate=f"{r['value']}<br>n={r['n']}<br>C1={r['c1_pct']:.1%}<br>RR={r['risk_ratio']:.2f}<extra></extra>",
            showlegend=False
        ))
    fig_r.update_layout(
        xaxis=dict(title=bi('Ratio de riesgo (vs. promedio)', 'Risk ratio (vs. average)'),
                  zeroline=True, zerolinecolor='#374151', zerolinewidth=2,
                  tickformat='+.0%', range=[-0.45, 0.45]),
        yaxis=dict(autorange=True),
        height=max(350, len(chart_data) * 28),
        margin=dict(l=220, r=30, t=20, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(249,250,251,1)',
        font=dict(family='Montserrat', size=10)
    )
    return pio.to_html(fig_r, include_plotlyjs=False, full_html=False, div_id=f'risk_diverging{div_suffix}')

risk_chart_25 = build_risk_chart(categorical)
risk_chart_26 = build_risk_chart(c26, div_suffix='_26')
risk_chart_html = yr_block(risk_chart_25, risk_chart_26)

# Critical dyad thresholds
dyad_thresholds = {'D1': 0.6, 'D4': 0.6, 'D5': 0.5, 'D6': 0.6}
dyad_labels_es = {'D1': 'Miedo al cambio', 'D4': 'Cambio impuesto', 'D5': 'Pertenencia debilitada', 'D6': 'Desconexión'}
dyad_labels_en = {'D1': 'Fear of change', 'D4': 'Imposed change', 'D5': 'Weakened belonging', 'D6': 'Disconnection'}

def compute_thresh_rows(dyad_data):
    """Compute critical threshold table rows for a given dyad DataFrame."""
    thresholds = []
    for did, thresh in dyad_thresholds.items():
        above_total = int((dyad_data[did] > thresh).sum())
        pct_total = above_total / N
        per_cluster = {}
        for cl in [1, 2, 3]:
            mask = dyad_data['cluster'] == cl
            above_cl = int((dyad_data.loc[mask, did] > thresh).sum())
            n_cl = int(mask.sum())
            per_cluster[cl] = {'above': above_cl, 'n': n_cl, 'pct': above_cl / n_cl if n_cl > 0 else 0}
        thresholds.append({
            'did': did, 'thresh': thresh,
            'label_es': dyad_labels_es[did], 'label_en': dyad_labels_en[did],
            'total': above_total, 'pct': pct_total,
            'per_cluster': per_cluster
        })
    rows = ""
    for ct_item in thresholds:
        cells = ""
        for cl in [1, 2, 3]:
            pc = ct_item['per_cluster'][cl]
            pct_val = pc['pct']
            bg = f"background:rgba(239,68,68,{min(pct_val*1.5, 0.3):.2f})" if pct_val > 0.2 else ""
            cells += f'<td style="text-align:center;{bg}"><strong>{pc["above"]}</strong><br><span style="font-size:0.68rem;color:var(--gris-texto);">{pct_val:.0%}</span></td>'
        rows += f"""
        <tr>
            <td><strong>{ct_item['did']}</strong></td>
            <td>{bi(ct_item['label_es'], ct_item['label_en'])}</td>
            <td style="text-align:center;">> {ct_item['thresh']}</td>
            <td style="text-align:center;"><strong>{ct_item['total']}</strong> ({ct_item['pct']:.0%})</td>
            {cells}
        </tr>"""
    return rows

# Compound risk (already computed in stats block above)
thresh_rows_25 = compute_thresh_rows(dyads)
thresh_rows_26 = compute_thresh_rows(d26)
thresh_rows = yr_block(thresh_rows_25, thresh_rows_26)

# ── Build Sidebar HTML ───────────────────────────────────────────────────
print("[Report] Building filter sidebar...")
demo_titles = {
    'cargo': ('Cargo / Rol', 'Role / Position'),
    'antiguedad': ('Antigüedad', 'Seniority'),
    'area': ('Área', 'Department'),
    'genero': ('Género', 'Gender'),
    'educacion': ('Nivel educativo', 'Education level'),
}
sidebar_html = ""
for field in DEMO_FIELDS:
    if field not in categorical.columns:
        continue
    title_es, title_en = demo_titles.get(field, (field, field))
    vc = categorical[field].value_counts()
    opts = ""
    for val in sorted(vc.index.tolist()):
        count = int(vc[val])
        safe_val = str(val).replace("'", "\\'").replace('"', '&quot;')
        opts += f'''<label class="filter-option">
            <input type="checkbox" checked data-field="{field}" data-value="{safe_val}">
            <span>{val}</span>
            <span class="fo-count" id="fc-{field}-{safe_val}">{count}</span>
        </label>'''
    sidebar_html += f'''<div class="filter-group" id="fg-{field}">
        <div class="filter-group-header" onclick="toggleFilterGroup('{field}')">
            <span>{bi(title_es, title_en)}</span>
            <span class="fg-arrow">▼</span>
        </div>
        <div class="filter-options">{opts}</div>
    </div>'''

# ── Build HTML ───────────────────────────────────────────────────────────
print("[Report] Assembling HTML...")

html = f"""<!DOCTYPE html>
<html lang="es" data-lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ETB SenseMaker Report | MéTRIK</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{
    --bg: #F9FAFB;
    --white: #FFFFFF;
    --negro-carbon: #1A1A1A;
    --verde-accent: #10B981;
    --gris-linea: #E5E7EB;
    --gris-texto: #6B7280;
    --texto-body: #374151;
    --c1: #EF4444;
    --c2: #F59E0B;
    --c3: #10B981;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Montserrat', -apple-system, sans-serif; background: var(--bg); color: var(--texto-body); line-height: 1.6; font-size: 0.9rem; }}
[data-lang="es"] .lang-en {{ display: none !important; }}
[data-lang="en"] .lang-es {{ display: none !important; }}

/* Header */
.site-header {{ position: fixed; top: 0; left: 0; right: 0; height: 56px; background: var(--negro-carbon); display: flex; align-items: center; justify-content: space-between; padding: 0 2rem; z-index: 1000; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
.site-header .logo {{ color: white; font-weight: 700; font-size: 1.1rem; letter-spacing: -0.5px; }}
.site-header .logo span {{ color: var(--verde-accent); }}
.site-header nav {{ display: flex; gap: 1.5rem; align-items: center; }}
.site-header nav a {{ color: #9CA3AF; text-decoration: none; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; transition: color 0.2s; }}
.site-header nav a:hover {{ color: white; }}
.lang-toggle {{ display: flex; gap: 4px; }}
.lang-toggle button {{ background: transparent; border: 1px solid #4B5563; color: #9CA3AF; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.7rem; font-weight: 600; transition: all 0.2s; }}
.lang-toggle button.active {{ background: var(--verde-accent); border-color: var(--verde-accent); color: white; }}

/* Main */
.main-content {{ margin-top: 56px; max-width: 1200px; margin-left: auto; margin-right: auto; padding: 0 2rem; }}

/* Hero */
.hero {{ background: linear-gradient(135deg, #064E3B 0%, #065F46 50%, #047857 100%); color: white; padding: 4rem 3rem; border-radius: 0 0 20px 20px; margin-bottom: 3rem; position: relative; overflow: hidden; }}
.hero::before {{ content: ''; position: absolute; top: -50%; right: -20%; width: 400px; height: 400px; background: radial-gradient(circle, rgba(16,185,129,0.15) 0%, transparent 70%); }}
.hero h1 {{ font-size: 2.2rem; font-weight: 800; letter-spacing: -1px; margin-bottom: 0.5rem; }}
.hero .subtitle {{ font-size: 1rem; font-weight: 300; opacity: 0.85; margin-bottom: 1.5rem; }}
.hero .hero-meta {{ display: flex; gap: 2rem; font-size: 0.8rem; opacity: 0.7; }}

/* Chapters */
.chapter {{ padding: 3rem 0; scroll-margin-top: 70px; }}
.chapter-header {{ margin-bottom: 2rem; }}
.chapter-header .chapter-num {{ font-size: 3rem; font-weight: 800; color: var(--verde-accent); opacity: 0.3; line-height: 1; }}
.chapter-header h2 {{ font-size: 1.6rem; font-weight: 700; color: var(--negro-carbon); margin-top: -10px; }}
.chapter-intro {{ max-width: 800px; color: var(--gris-texto); font-size: 0.85rem; margin-bottom: 2rem; }}

/* Metrics */
.metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
.metric-card {{ background: white; border-radius: 10px; padding: 1.2rem; border-top: 3px solid var(--verde-accent); box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: transform 0.2s; }}
.metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
.metric-card .metric-value {{ font-size: 1.8rem; font-weight: 800; color: var(--negro-carbon); }}
.metric-card .metric-label {{ font-size: 0.72rem; color: var(--gris-texto); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}

/* Triads */
.triad-section {{ background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }}
.triad-section h3 {{ font-size: 1rem; font-weight: 700; margin-bottom: 0.3rem; }}
.triad-question {{ font-size: 0.8rem; color: var(--gris-texto); font-style: italic; margin-bottom: 1rem; }}
.triad-grid {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 1.5rem; align-items: start; }}
.chart-container {{ background: #FAFBFC; border-radius: 8px; padding: 0.5rem; }}
.triad-data h4 {{ font-size: 0.8rem; font-weight: 700; margin-bottom: 0.8rem; color: var(--gris-texto); text-transform: uppercase; letter-spacing: 0.5px; }}
.triad-apexes {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 1rem; }}
.apex-tag {{ font-size: 0.7rem; background: #F3F4F6; padding: 3px 8px; border-radius: 4px; color: var(--texto-body); }}
.zone-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 0.75rem; }}
.zone-label {{ width: 30px; font-weight: 700; }}
.zone-bars {{ flex: 1; height: 16px; background: #F3F4F6; border-radius: 8px; overflow: hidden; }}
.zone-bar {{ height: 100%; border-radius: 8px; min-width: 4px; }}
.zone-val {{ width: 120px; font-size: 0.7rem; color: var(--gris-texto); text-align: right; }}

/* Dyads */
.dyad-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
.dyad-card {{ background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }}
.dyad-card h4 {{ font-size: 0.9rem; font-weight: 700; margin-bottom: 0.8rem; }}
.dyad-poles {{ display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--gris-texto); margin-bottom: 0.8rem; font-weight: 500; }}
.dyad-cluster-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
.dyad-cluster-label {{ width: 195px; display: flex; align-items: center; gap: 6px; font-size: 0.72rem; font-weight: 500; flex-shrink: 0; }}
.dyad-cluster-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.dyad-gauge-track {{ flex: 1; height: 26px; background: linear-gradient(90deg, #F3F4F6 0%, #E5E7EB 100%); border-radius: 13px; position: relative; overflow: visible; }}
.dyad-gauge-fill {{ height: 100%; border-radius: 13px 0 0 13px; position: relative; transition: width 0.6s ease; }}
.dyad-gauge-marker {{ position: absolute; right: -2px; top: -2px; width: 4px; height: 30px; border-radius: 2px; opacity: 0.7; }}
.dyad-gauge-value {{ position: absolute; right: -14px; top: -20px; font-size: 0.7rem; font-weight: 700; background: #fff; padding: 1px 4px; border-radius: 3px; border: 1px solid var(--gris-linea); }}
.dyad-scale {{ display: flex; justify-content: space-between; font-size: 0.65rem; color: #9CA3AF; margin-top: 4px; padding: 0 2px; }}
.dyad-overall {{ text-align: right; font-size: 0.75rem; color: var(--gris-texto); margin-top: 8px; }}

/* Stones */
.stones-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
.stone-section {{ background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }}
.stone-section h3 {{ font-size: 1rem; font-weight: 700; margin-bottom: 0.5rem; }}
.stone-layout {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 1.5rem; align-items: start; margin-bottom: 1rem; }}
.stone-table-side {{ overflow-x: auto; }}
.stone-legend td:first-child {{ font-size: 0.85rem; text-align: center; width: 30px; }}

/* Conclusions & Recommendations */
.conclusions-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
.conclusion-card {{ background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06); border-top: 4px solid var(--verde-accent); }}
.conclusion-card h4 {{ font-size: 0.85rem; font-weight: 700; color: var(--negro-carbon); margin-bottom: 0.8rem; }}
.conclusion-card p {{ font-size: 0.82rem; color: var(--texto-body); }}
.reco-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
.reco-card {{ background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
.reco-card .reco-num {{ font-size: 1.8rem; font-weight: 800; color: var(--verde-accent); opacity: 0.3; }}
.reco-card h4 {{ font-size: 0.9rem; font-weight: 700; margin-bottom: 0.5rem; }}
.reco-card p {{ font-size: 0.82rem; color: var(--texto-body); }}
.reco-card .reco-target {{ display: inline-block; font-size: 0.68rem; padding: 2px 8px; border-radius: 4px; font-weight: 600; margin-top: 0.5rem; }}
.final-panel {{ background: linear-gradient(135deg, #064E3B, #047857); color: white; border-radius: 16px; padding: 2.5rem; margin-top: 2.5rem; text-align: center; }}
.final-panel h3 {{ font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem; }}
.final-panel p {{ font-size: 0.9rem; opacity: 0.9; max-width: 700px; margin: 0 auto; }}

/* Risk factors */
.risk-metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }}
.risk-metric {{ background: white; border-radius: 10px; padding: 1.2rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
.risk-metric .risk-val {{ font-size: 1.6rem; font-weight: 800; }}
.risk-metric .risk-label {{ font-size: 0.7rem; color: var(--gris-texto); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-top: 4px; }}
.risk-table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }}
.risk-table th {{ background: #1F2937; color: white; padding: 10px 12px; text-align: left; font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; }}
.risk-table td {{ padding: 10px 12px; border-bottom: 1px solid #F3F4F6; }}
.risk-alert {{ background: #FEF2F2; border: 1px solid #FECACA; border-radius: 10px; padding: 1.2rem 1.5rem; margin: 1.5rem 0; }}
.risk-alert .risk-alert-label {{ font-size: 0.7rem; font-weight: 700; color: #DC2626; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }}

/* Cluster profiles */
.cluster-profile {{ background: white; border-radius: 12px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
.cluster-header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 1.5rem; }}
.cluster-icon {{ font-size: 2rem; }}
.cluster-meta {{ font-size: 0.8rem; color: var(--gris-texto); }}
.cluster-body {{ display: grid; grid-template-columns: 1.5fr 1fr; gap: 2rem; margin-bottom: 1.5rem; }}
.cluster-desc h4, .cluster-metaphors h4, .cluster-quotes h4, .cluster-presidente h4 {{ font-size: 0.82rem; font-weight: 700; color: var(--gris-texto); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.8rem; }}
.cluster-desc ul {{ font-size: 0.82rem; padding-left: 1.2rem; }}
.cluster-desc li {{ margin-bottom: 0.4rem; }}
.distinctive-words {{ font-size: 0.82rem; color: var(--verde-accent); font-weight: 600; font-style: italic; }}
.metaphor-row {{ font-size: 0.82rem; margin-bottom: 0.5rem; }}
.quote-card {{ background: #F9FAFB; border-left: 3px solid var(--verde-accent); padding: 0.7rem 1rem; margin-bottom: 0.6rem; border-radius: 0 8px 8px 0; font-size: 0.8rem; }}
.cluster-presidente {{ margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--gris-linea); }}

/* Demographics */
.demo-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
.demo-table-container {{ background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }}
.demo-table-container h4 {{ font-size: 0.9rem; font-weight: 700; margin-bottom: 1rem; }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
.data-table th {{ background: #F3F4F6; padding: 8px 10px; text-align: left; font-weight: 600; font-size: 0.72rem; color: var(--gris-texto); text-transform: uppercase; border-bottom: 2px solid var(--gris-linea); }}
.data-table td {{ padding: 7px 10px; border-bottom: 1px solid #F3F4F6; }}

/* Insights */
.insight-box {{ background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 1.2rem 1.5rem; margin: 1.5rem 0; }}
.insight-box .insight-label {{ font-size: 0.7rem; font-weight: 700; color: var(--verde-accent); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }}

/* Footer */
.site-footer {{ text-align: center; padding: 3rem 2rem; color: var(--gris-texto); font-size: 0.75rem; border-top: 1px solid var(--gris-linea); margin-top: 3rem; }}

/* Responsive */
@media (max-width: 800px) {{
    .triad-grid, .dyad-grid, .stones-grid, .demo-grid, .cluster-body, .stone-layout, .conclusions-grid, .reco-grid, .risk-metrics {{ grid-template-columns: 1fr; }}
    .hero {{ padding: 2rem 1.5rem; }}
    .hero h1 {{ font-size: 1.5rem; }}
    .dyad-cluster-label {{ width: 140px; }}
}}

/* Filter Sidebar */
.filter-sidebar {{ position: fixed; top: 56px; left: 0; bottom: 0; width: 300px; background: #FFFFFF; border-right: 1px solid var(--gris-linea); z-index: 999; transform: translateX(-100%); transition: transform 0.3s ease; overflow-y: auto; padding: 1.5rem 1rem; box-shadow: 4px 0 20px rgba(0,0,0,0.08); }}
.filter-sidebar.open {{ transform: translateX(0); }}
.filter-sidebar .filter-n {{ text-align: center; font-size: 0.8rem; color: var(--gris-texto); margin-bottom: 1rem; padding-bottom: 0.8rem; border-bottom: 1px solid var(--gris-linea); }}
.filter-sidebar .filter-n strong {{ font-size: 1.4rem; color: var(--negro-carbon); display: block; }}
.filter-group {{ margin-bottom: 1rem; }}
.filter-group-header {{ display: flex; align-items: center; justify-content: space-between; cursor: pointer; padding: 8px 10px; background: #F9FAFB; border-radius: 8px; font-size: 0.78rem; font-weight: 600; color: var(--negro-carbon); user-select: none; }}
.filter-group-header:hover {{ background: #F3F4F6; }}
.filter-group-header .fg-arrow {{ transition: transform 0.2s; font-size: 0.65rem; color: var(--gris-texto); }}
.filter-group.collapsed .filter-options {{ display: none; }}
.filter-group.collapsed .fg-arrow {{ transform: rotate(-90deg); }}
.filter-options {{ padding: 6px 4px 0; }}
.filter-option {{ display: flex; align-items: center; gap: 8px; padding: 4px 6px; font-size: 0.75rem; cursor: pointer; border-radius: 4px; }}
.filter-option:hover {{ background: #F3F4F6; }}
.filter-option input {{ accent-color: var(--verde-accent); cursor: pointer; }}
.filter-option .fo-count {{ margin-left: auto; font-size: 0.65rem; color: #9CA3AF; background: #F3F4F6; padding: 1px 6px; border-radius: 10px; min-width: 24px; text-align: center; }}
.filter-actions {{ display: flex; gap: 8px; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--gris-linea); }}
.filter-actions button {{ flex: 1; padding: 8px 12px; border-radius: 8px; font-size: 0.75rem; font-weight: 600; cursor: pointer; transition: all 0.2s; border: none; font-family: 'Montserrat', sans-serif; }}
.btn-apply {{ background: var(--verde-accent); color: white; }}
.btn-apply:hover {{ background: #059669; }}
.btn-clear {{ background: #F3F4F6; color: var(--gris-texto); }}
.btn-clear:hover {{ background: #E5E7EB; }}

/* Filter toggle button */
.filter-btn {{ background: transparent; border: 1px solid #4B5563; color: #9CA3AF; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; transition: all 0.2s; display: flex; align-items: center; gap: 4px; }}
.filter-btn:hover {{ color: white; border-color: #9CA3AF; }}
.filter-btn.active {{ background: var(--verde-accent); border-color: var(--verde-accent); color: white; }}
.filter-badge {{ display: none; background: #EF4444; color: white; font-size: 0.55rem; font-weight: 700; padding: 1px 5px; border-radius: 10px; margin-left: 2px; }}

/* Year Toggle */
.year-toggle {{ display: flex; gap: 4px; }}
.year-toggle button {{ background: transparent; border: 1px solid #4B5563; color: #9CA3AF; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.7rem; font-weight: 600; transition: all 0.2s; font-family: 'Montserrat', sans-serif; }}
.year-toggle button.active {{ background: #3B82F6; border-color: #3B82F6; color: white; }}
.year-toggle button:hover:not(.active) {{ border-color: #9CA3AF; color: white; }}

/* Urgency Badges */
.urgency {{ display: inline-flex; align-items: center; gap: 4px; font-size: 0.68rem; font-weight: 700; padding: 3px 10px; border-radius: 12px; letter-spacing: 0.3px; vertical-align: middle; }}
.urgency-critical {{ background: #FEE2E2; color: #991B1B; }}
.urgency-watch {{ background: #FEF3C7; color: #92400E; }}
.urgency-ontrack {{ background: #D1FAE5; color: #065F46; }}

/* Delta Indicators */
.delta {{ font-size: 0.72rem; font-weight: 700; margin-left: 6px; }}
.delta-up {{ color: #059669; }}
.delta-down {{ color: #DC2626; }}
.delta-neutral {{ color: #6B7280; }}
.delta-container {{ display: none; }}
body.show-deltas .delta-container {{ display: inline; }}
span.y26, div.y26 {{ display: none; }}
body.show-deltas span.y25, body.show-deltas div.y25 {{ display: none !important; }}
body.show-deltas span.y26 {{ display: inline; }}
body.show-deltas div.y26 {{ display: block; }}

/* Sidebar open state */
body.sidebar-open .main-content {{ margin-left: 300px; transition: margin-left 0.3s ease; }}
@media (max-width: 1000px) {{ body.sidebar-open .main-content {{ margin-left: 0; }} }}

@media print {{
    .site-header {{ position: static; }}
    .chapter {{ page-break-inside: avoid; }}
    .filter-sidebar {{ display: none !important; }}
}}
</style>
</head>
<body>

<!-- Header -->
<header class="site-header">
    <div class="logo">{bi('MéTRIK <span>Analítica</span>', 'MéTRIK <span>Analytics</span>')}</div>
    <nav>
        <button class="filter-btn" onclick="toggleSidebar()" title="Filtros demográficos">&#9776;<span class="filter-badge" id="filter-badge"></span></button>
        <a href="#ch1">{bi('Resumen', 'Summary')}</a>
        <a href="#ch2">{bi('Triadas', 'Triads')}</a>
        <a href="#ch3">{bi('Díadas', 'Dyads')}</a>
        <a href="#ch4">{bi('Stones', 'Stones')}</a>
        <a href="#ch5">{bi('Perfiles', 'Profiles')}</a>
        <a href="#ch6">{bi('Demografía', 'Demographics')}</a>
        <a href="#ch7">{bi('Riesgo', 'Risk')}</a>
        <a href="#ch8">{bi('Conclusiones', 'Conclusions')}</a>
        <div class="year-toggle">
            <button class="active" onclick="setYear('2025')">2025</button>
            <button onclick="setYear('2026')">2026</button>
        </div>
        <div class="lang-toggle">
            <button class="active" onclick="setLang('es')">ES</button>
            <button onclick="setLang('en')">EN</button>
        </div>
    </nav>
</header>

<!-- Filter Sidebar -->
<aside class="filter-sidebar" id="filterSidebar">
    <div class="filter-n">
        <strong id="hero-n">{N:,}</strong>
        {bi('respondientes', 'respondents')}
    </div>
    {sidebar_html}
    <div class="filter-actions">
        <button class="btn-apply" onclick="applyFilters()">{bi('Aplicar', 'Apply')}</button>
        <button class="btn-clear" onclick="clearFilters()">{bi('Limpiar', 'Clear')}</button>
    </div>
</aside>

<main class="main-content">

<!-- Hero -->
<section class="hero">
    <h1>{bi('Análisis SenseMaker — ETB', 'SenseMaker Analysis — ETB')}</h1>
    <p class="subtitle">{bi('Transformación cultural y disposición al cambio', 'Cultural transformation and readiness for change')}</p>
    <p class="subtitle">{bi('Empresa de Telecomunicaciones de Bogotá', 'Bogotá Telecommunications Company')}</p>
    <div class="hero-meta">
        <span>n = {N}</span>
        <span>{bi('3 clusters narrativos', '3 narrative clusters')}</span>
        <span>{bi('9 triadas · 6 díadas · 2 stones', '9 triads · 6 dyads · 2 stones')}</span>
    </div>
</section>

<!-- Chapter 1: Executive Summary -->
<section class="chapter" id="ch1">
    <div class="chapter-header">
        <div class="chapter-num">01</div>
        <h2>{bi('Resumen Ejecutivo', 'Executive Summary')}</h2>
    </div>
    <p class="chapter-intro">{bi(
        yr(
            '<strong>Decisión ejecutiva requerida:</strong> {s1} colaboradores ({p1}%) viven la transformación con escepticismo, desconexión y cambio percibido como impuesto. Este informe traduce 1,049 micro-narrativas en indicadores de riesgo accionables y metas de cierre para los próximos 12 meses. La pregunta estratégica: <em>¿Está la organización emocionalmente lista para la transformación, y dónde están las fracturas narrativas que podrían sabotearla?</em>'.format(s1=SIZES[1], p1=PCTS[1]),
            '<strong>Resultado de la intervención (2026):</strong> C1 se redujo de {s1_old} a {s1_new} personas ({p1_new}%), una migración de {delta} colaboradores hacia clusters más constructivos. La transformación avanza: el {c2c3}% ya construye activamente. Pero {s1_new} personas aún viven el cambio como imposición — la siguiente fase debe cerrar esa brecha. <em>¿Están funcionando las intervenciones, y dónde acelerar?</em>'.format(s1_old=SIZES[1], s1_new=SIZES_26[1], p1_new=PCTS_26[1], delta=SIZES[1]-SIZES_26[1], c2c3=C2C3_pct_26)
        ),
        yr(
            '<strong>Executive decision required:</strong> {s1} employees ({p1}%) experience transformation with skepticism, disconnection, and perceived imposed change. This report translates 1,049 micro-narratives into actionable risk indicators and closure targets for the next 12 months. The strategic question: <em>Is the organization emotionally ready for transformation, and where are the narrative fractures that could sabotage it?</em>'.format(s1=SIZES[1], p1=PCTS[1]),
            '<strong>Intervention results (2026):</strong> C1 decreased from {s1_old} to {s1_new} people ({p1_new}%), a migration of {delta} employees toward more constructive clusters. Transformation is advancing: {c2c3}% are now actively building. But {s1_new} people still experience change as imposition — the next phase must close that gap. <em>Are interventions working, and where to accelerate?</em>'.format(s1_old=SIZES[1], s1_new=SIZES_26[1], p1_new=PCTS_26[1], delta=SIZES[1]-SIZES_26[1], c2c3=C2C3_pct_26)
        )
    )}</p>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{N:,}</div>
            <div class="metric-label">{bi('Respondientes válidos', 'Valid respondents')}</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">{bi('Clusters narrativos', 'Narrative clusters')}</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">57</div>
            <div class="metric-label">{bi('Variables de señalización', 'Signifier variables')}</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">98.7</div>
            <div class="metric-label">{bi('Score calidad datos', 'Data quality score')}</div>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card" style="border-top-color: var(--c1);" data-cluster="1">
            <div class="metric-value" style="color: var(--c1);">{SIZES[1]}</div>
            <div class="metric-label">C1 · {bi(NAMES_ES[1], NAMES_EN[1])} ({yr(f'{PCTS[1]}%', f'{PCTS_26[1]}% (era {PCTS[1]}%)')}) <span class="delta-container"></span></div>
            <span class="urgency urgency-critical">{bi(yr('🔴 Actuar ahora', f'🔴 -{SIZES[1]-SIZES_26[1]} personas'), yr('🔴 Act now', f'🔴 -{SIZES[1]-SIZES_26[1]} people'))}</span>
        </div>
        <div class="metric-card" style="border-top-color: var(--c2);" data-cluster="2">
            <div class="metric-value" style="color: var(--c2);">{SIZES[2]}</div>
            <div class="metric-label">C2 · {bi(NAMES_ES[2], NAMES_EN[2])} ({yr(f'{PCTS[2]}%', f'{PCTS_26[2]}% (era {PCTS[2]}%)')}) <span class="delta-container"></span></div>
            <span class="urgency urgency-ontrack">{bi(yr('🟢 En curso', f'🟢 +{SIZES_26[2]-SIZES[2]} personas'), yr('🟢 On track', f'🟢 +{SIZES_26[2]-SIZES[2]} people'))}</span>
        </div>
        <div class="metric-card" style="border-top-color: var(--c3);" data-cluster="3">
            <div class="metric-value" style="color: var(--c3);">{SIZES[3]}</div>
            <div class="metric-label">C3 · {bi(NAMES_ES[3], NAMES_EN[3])} ({yr(f'{PCTS[3]}%', f'{PCTS_26[3]}% (era {PCTS[3]}%)')}) <span class="delta-container"></span></div>
            <span class="urgency urgency-watch">{bi('🟡 Potenciar', '🟡 Accelerate')}</span>
        </div>
    </div>
</section>

<!-- Chapter 2: Triads -->
<section class="chapter" id="ch2">
    <div class="chapter-header">
        <div class="chapter-num">02</div>
        <h2>{bi('Análisis de Triadas', 'Triad Analysis')}</h2>
    </div>
    <p class="chapter-intro">{bi(
        yr(
            f'<strong>Priorice aquí:</strong> Las 9 triadas revelan dónde se fractura el compromiso organizacional. La mayor divergencia está en T4 (Sentido de pertenencia): C1 reporta solo {t4a_c1_25:.0f}% en "mi aporte era valorado" vs. {t4a_c3_25:.0f}% en C3. Cada punto porcentual que se cierre en esa brecha equivale a recuperar ~4 personas del escepticismo hacia la construcción activa. <span class="urgency urgency-critical">🔴 Brecha crítica: {t4a_gap_25:.1f} pp en T4_a</span>',
            f'<strong>Progreso 2026:</strong> La brecha en T4 se redujo de {t4a_gap_25:.1f} pp a {t4a_gap_26:.1f} pp. C1 subió de {t4a_c1_25:.0f}% a {t4a_c1_26:.0f}% en "mi aporte era valorado". La intervención está cerrando la fractura — pero {t4a_gap_26:.1f} pp de brecha aún es significativa. La siguiente fase debe profundizar en las áreas con mayor concentración de Escépticos residuales. <span class="urgency urgency-watch">🟡 Brecha en cierre: {t4a_gap_26:.1f} pp</span>'
        ),
        yr(
            f'<strong>Prioritize here:</strong> The 9 triads reveal where organizational commitment fractures. The greatest divergence is in T4 (Sense of belonging): C1 reports only {t4a_c1_25:.0f}% in "my contribution was valued" vs. {t4a_c3_25:.0f}% in C3. Every percentage point closed in this gap is equivalent to recovering ~4 people from skepticism toward active building. <span class="urgency urgency-critical">🔴 Critical gap: {t4a_gap_25:.1f} pp in T4_a</span>',
            f'<strong>2026 Progress:</strong> T4 gap reduced from {t4a_gap_25:.1f} pp to {t4a_gap_26:.1f} pp. C1 rose from {t4a_c1_25:.0f}% to {t4a_c1_26:.0f}% in "my contribution was valued." The intervention is closing the fracture — but {t4a_gap_26:.1f} pp gap is still significant. Next phase must deepen in areas with highest residual Skeptic concentration. <span class="urgency urgency-watch">🟡 Gap closing: {t4a_gap_26:.1f} pp</span>'
        )
    )}</p>
    {triad_sections}

    <!-- Cognitive Edge Synthesis: Triads -->
    <h3 style="margin:2.5rem 0 1rem;">{bi('Síntesis Cognitiva — Triadas', 'Cognitive Synthesis — Triads')}</h3>
    <div class="conclusions-grid">
        <div class="conclusion-card">
            <h4>{bi('🔍 Lo Evidente — Actúe aquí primero', '🔍 The Obvious — Act here first')}</h4>
            <p>{bi(
                yr(
                    f'<strong>Redirija la inversión en pertenencia.</strong> T8 y T4 son las triadas más fracturadas (rango > 0.45): C3 vive identidad compartida y valoración; C1 gravita a intereses personales e invisibilidad. La fractura más profunda de ETB no está en la actitud frente al cambio — está en cómo se vive la pertenencia. <strong>MECANISMO:</strong> Lanzar programa de reconocimiento visible (no monetario) en las áreas con mayor concentración de C1 dentro de 60 días. <span class="urgency urgency-critical">🔴 Brecha: {t4a_gap_25:.1f} pp</span>',
                    f'<strong>La pertenencia empezó a mejorar.</strong> La brecha T4 se redujo de {t4a_gap_25:.1f} a {t4a_gap_26:.1f} pp. Los programas de reconocimiento visible ya están generando tracción — C1 subió de {t4a_c1_25:.0f}% a {t4a_c1_26:.0f}% en "mi aporte era valorado". <strong>SIGUIENTE FASE:</strong> Expandir el programa a todas las áreas con >30% de C1 residual y medir impacto trimestral. <span class="urgency urgency-watch">🟡 Brecha en cierre: {t4a_gap_26:.1f} pp</span>'
                ),
                yr(
                    f'<strong>Redirect belonging investment.</strong> T8 and T4 are the most fractured triads (range > 0.45): C3 lives shared identity and valuation; C1 gravitates to personal interests and invisibility. The deepest fracture is not in attitude toward change — it is in how belonging is experienced. <strong>MECHANISM:</strong> Launch visible (non-monetary) recognition program in areas with highest C1 concentration within 60 days. <span class="urgency urgency-critical">🔴 Gap: {t4a_gap_25:.1f} pp</span>',
                    f'<strong>Belonging is improving.</strong> T4 gap reduced from {t4a_gap_25:.1f} to {t4a_gap_26:.1f} pp. Visible recognition programs are gaining traction — C1 rose from {t4a_c1_25:.0f}% to {t4a_c1_26:.0f}% in "my contribution was valued." <strong>NEXT PHASE:</strong> Expand program to all areas with >30% residual C1 and measure quarterly impact. <span class="urgency urgency-watch">🟡 Gap closing: {t4a_gap_26:.1f} pp</span>'
                )
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #F59E0B;">
            <h4>{bi('💡 Lo Contra-Evidente — No confunda síntomas', "💡 The Counter-Intuitive — Don't confuse symptoms")}</h4>
            <p>{bi(
                yr(
                    f'<strong>No trate a C1 como resistentes al cambio.</strong> T1 revela que C1 no es "anti-cambio": tiene {t1_c1_resign_25}% en <em>resignación aprendida</em>, no en oposición activa. La diferencia es estratégica — la resignación se revierte con inclusión, la oposición con argumentos. C2 ({PCTS[2]}%) es el grupo bisagra: pragmáticos que balancean responsabilidad con curiosidad sin extremos. <strong>MECANISMO:</strong> Involucrar a C2 como puentes narrativos en sesiones de co-diseño con C1. No "convencer" — co-construir.',
                    f'<strong>La inclusión funcionó.</strong> T1 muestra que la resignación en C1 bajó de {t1_c1_resign_25}% a {t1_c1_resign_26}%. C2 creció a {PCTS_26[2]}% — el grupo bisagra se fortaleció con {SIZES_26[2]-SIZES[2]} personas adicionales que migraron desde C1. Las sesiones de co-diseño están revirtiendo la resignación. <strong>SIGUIENTE FASE:</strong> Convertir los puentes narrativos C2→C1 en un programa permanente de mentoría entre pares.'
                ),
                yr(
                    f"<strong>Do not treat C1 as change-resistant.</strong> T1 reveals C1 is not anti-change: {t1_c1_resign_25}% shows <em>learned helplessness</em>, not active opposition. The difference is strategic — helplessness is reversed with inclusion, opposition with arguments. C2 ({PCTS[2]}%) is the hinge group: pragmatists balancing responsibility with curiosity without extremes. <strong>MECHANISM:</strong> Involve C2 as narrative bridges in co-design sessions with C1.",
                    f"<strong>Inclusion worked.</strong> T1 shows resignation in C1 dropped from {t1_c1_resign_25}% to {t1_c1_resign_26}%. C2 grew to {PCTS_26[2]}% — the hinge group strengthened with {SIZES_26[2]-SIZES[2]} additional people migrating from C1. Co-design sessions are reversing helplessness. <strong>NEXT PHASE:</strong> Convert C2→C1 narrative bridges into a permanent peer mentoring program."
                )
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #EF4444;">
            <h4>{bi('📊 KPI Narrativo — Meta 12 meses', '📊 Narrative KPI — 12-month target')}</h4>
            <p>{bi(
                yr(
                    f'<strong>T4_a: "Mi aporte era valorado"</strong> — C1 = {t4a_c1_25:.0f}%, C3 = {t4a_c3_25:.0f}%. Rango: {t4a_gap_25:.1f} pp. <strong>META:</strong> Subir C1 de {t4a_c1_25:.0f}% a 45% en 12 meses. Cada punto = ~4 personas que migran del escepticismo al pragmatismo. <strong>CÓMO MEDIR:</strong> Segundo pulso SenseMaker en mes 6 y 12, foco en T4_a segmentado por cluster. <span class="urgency urgency-critical">🔴 Indicador ancla</span>',
                    f'<strong>T4_a: "Mi aporte era valorado"</strong> — C1 subió de {t4a_c1_25:.0f}% a {t4a_c1_26:.0f}%. Brecha actual: {t4a_gap_26:.1f} pp (era {t4a_gap_25:.1f} pp). Se cerraron {t4a_gap_25 - t4a_gap_26:.1f} pp. <strong>SIGUIENTE META:</strong> Llevar C1 a 55% en los próximos 12 meses. <span class="urgency urgency-watch">🟡 Indicador ancla — en progreso</span>'
                ),
                yr(
                    f'<strong>T4_a: "My contribution was valued"</strong> — C1 = {t4a_c1_25:.0f}%, C3 = {t4a_c3_25:.0f}%. Range: {t4a_gap_25:.1f} pp. <strong>TARGET:</strong> Raise C1 from {t4a_c1_25:.0f}% to 45% in 12 months. Each point = ~4 people migrating from skepticism to pragmatism. <span class="urgency urgency-critical">🔴 Anchor indicator</span>',
                    f'<strong>T4_a: "My contribution was valued"</strong> — C1 rose from {t4a_c1_25:.0f}% to {t4a_c1_26:.0f}%. Current gap: {t4a_gap_26:.1f} pp (was {t4a_gap_25:.1f} pp). Closed {t4a_gap_25 - t4a_gap_26:.1f} pp. <strong>NEXT TARGET:</strong> Bring C1 to 55% in next 12 months. <span class="urgency urgency-watch">🟡 Anchor indicator — in progress</span>'
                )
            )}</p>
        </div>
    </div>
</section>

<!-- Chapter 3: Dyads -->
<section class="chapter" id="ch3">
    <div class="chapter-header">
        <div class="chapter-num">03</div>
        <h2>{bi('Análisis de Díadas', 'Dyad Analysis')}</h2>
    </div>
    <p class="chapter-intro">{bi(
        yr(
            f'<strong>Alerte a su comité:</strong> D6 (Conexión percibida) es la díada más fracturada — C1 tiene mediana {d6_c1_25} (desconectado) mientras C3 está en {d6_c3_25}. Rango: {d6_range_25}. Esta es la mayor brecha interna y el predictor más fuerte de desvinculación emocional. <span class="urgency urgency-critical">🔴 D6: fractura {d6_range_25}</span>',
            f'<strong>Progreso en conexión:</strong> D6 mejoró — C1 pasó de mediana {d6_c1_25} a {d6_c1_26}. Rango se redujo de {d6_range_25} a {d6_range_26}. La intervención está cerrando la fractura, pero D6 sigue siendo la díada más amplia del instrumento. <span class="urgency urgency-watch">🟡 D6: {d6_range_26} (era {d6_range_25})</span>'
        ),
        yr(
            f"<strong>Alert your committee:</strong> D6 (Perceived connection) is the most fractured dyad — C1 median {d6_c1_25} (disconnected) while C3 is at {d6_c3_25}. Range: {d6_range_25}. This is the largest internal gap and strongest predictor of emotional disengagement. <span class=&quot;urgency urgency-critical&quot;>🔴 D6: fracture {d6_range_25}</span>",
            f"<strong>Connection progress:</strong> D6 improved — C1 moved from median {d6_c1_25} to {d6_c1_26}. Range reduced from {d6_range_25} to {d6_range_26}. The intervention is closing the fracture, but D6 remains the widest dyad in the instrument. <span class=&quot;urgency urgency-watch&quot;>🟡 D6: {d6_range_26} (was {d6_range_25})</span>"
        )
    )}</p>
    <div class="dyad-grid">
        {dyad_sections}
    </div>

    <!-- Cognitive Edge Synthesis: Dyads -->
    <h3 style="margin:2.5rem 0 1rem;">{bi('Síntesis Cognitiva — Díadas', 'Cognitive Synthesis — Dyads')}</h3>
    <div class="conclusions-grid">
        <div class="conclusion-card">
            <h4>{bi('🔍 Lo Evidente — Fractura operativa, no emocional', '🔍 The Obvious — Operational fracture, not emotional')}</h4>
            <p>{bi(
                yr(
                    f'<strong>Redefina el problema:</strong> D6 (Conexión) tiene un rango de {d6_range_25} entre C1 y C3. D1 confirma: el {d1_fear_pct_25}% de C1 reporta miedo al cambio (D1 > 0.6) vs. solo el {d1_fear_c3_25}% de C3. Una misma transformación genera realidades narrativas opuestas. <strong>MECANISMO:</strong> Establecer "circuitos de escucha" diferenciados en los próximos 45 días. <span class="urgency urgency-critical">🔴 {d1_fear_pct_25}% con miedo en C1</span>',
                    f'<strong>El miedo se redujo.</strong> D6 bajó su rango de {d6_range_25} a {d6_range_26}. El miedo en C1 bajó de {d1_fear_pct_25}% a {d1_fear_pct_26}% (D1 > 0.6). Los circuitos de escucha están funcionando. <strong>SIGUIENTE FASE:</strong> Expandir los circuitos a todas las gerencias con >30% de C1 residual. <span class="urgency urgency-watch">🟡 {d1_fear_pct_26}% con miedo (era {d1_fear_pct_25}%)</span>'
                ),
                yr(
                    f"<strong>Redefine the problem:</strong> D6 (Connection) has a range of {d6_range_25} between C1 and C3. D1 confirms: {d1_fear_pct_25}% of C1 reports fear of change (D1 > 0.6) vs. only {d1_fear_c3_25}% of C3. One transformation generates opposite narrative realities. <strong>MECHANISM:</strong> Establish differentiated listening circuits within 45 days. <span class=&quot;urgency urgency-critical&quot;>🔴 {d1_fear_pct_25}% with fear in C1</span>",
                    f"<strong>Fear has decreased.</strong> D6 range dropped from {d6_range_25} to {d6_range_26}. Fear in C1 dropped from {d1_fear_pct_25}% to {d1_fear_pct_26}% (D1 > 0.6). Listening circuits are working. <strong>NEXT PHASE:</strong> Expand circuits to all management units with >30% residual C1. <span class=&quot;urgency urgency-watch&quot;>🟡 {d1_fear_pct_26}% with fear (was {d1_fear_pct_25}%)</span>"
                )
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #F59E0B;">
            <h4>{bi('💡 Lo Contra-Evidente — La desconexión no es rechazo', '💡 The Counter-Intuitive — Disconnection is not rejection')}</h4>
            <p>{bi(
                yr(
                    f'<strong>No pierda a este grupo — todavía se sienten parte.</strong> D5 (Pertenencia) = {d5_overall_25} mediana global. C1 tiene {d5_c1_25}. La paradoja: C1 se siente <strong>desconectado</strong> (D6={d6_c1_25}) pero <strong>no ha perdido pertenencia</strong> (D5={d5_c1_25}). La desconexión es <em>operativa</em>, no emocional. <strong>MECANISMO:</strong> Crear canales de participación en decisiones de transformación. <span class="urgency urgency-watch">🟡 Ventana temporal limitada</span>',
                    f'<strong>La ventana se aprovechó.</strong> D5 global pasó de {d5_overall_25} a {d5_overall_26}. C1 mejoró de {d5_c1_25} a {d5_c1_26}. La desconexión operativa (D6) también mejoró: C1 bajó de {d6_c1_25} a {d6_c1_26}. La inclusión en decisiones funciona. <strong>SIGUIENTE FASE:</strong> Institucionalizar los canales de participación. <span class="urgency urgency-ontrack">🟢 Ventana aprovechada</span>'
                ),
                yr(
                    f"<strong>Do not lose this group — they still feel part of it.</strong> D5 (Belonging) = {d5_overall_25} overall median. C1 has {d5_c1_25}. The paradox: C1 feels <strong>disconnected</strong> (D6={d6_c1_25}) but <strong>has not lost belonging</strong> (D5={d5_c1_25}). Disconnection is <em>operational</em>, not emotional. <strong>MECHANISM:</strong> Create transformation decision-participation channels. <span class=&quot;urgency urgency-watch&quot;>🟡 Limited time window</span>",
                    f"<strong>The window was seized.</strong> D5 overall moved from {d5_overall_25} to {d5_overall_26}. C1 improved from {d5_c1_25} to {d5_c1_26}. Operational disconnection (D6) also improved: C1 dropped from {d6_c1_25} to {d6_c1_26}. Decision inclusion is working. <strong>NEXT PHASE:</strong> Institutionalize participation channels. <span class=&quot;urgency urgency-ontrack&quot;>🟢 Window seized</span>"
                )
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #EF4444;">
            <h4>{bi('📊 KPI Narrativo — Tablero de control ejecutivo', '📊 Narrative KPI — Executive dashboard')}</h4>
            <p>{bi(
                yr(
                    f'<strong>D6: Conexión percibida</strong> — C1 mediana = {d6_c1_25}. El {d6_disc_c1_25}% de C1 reporta desconexión (D6 > 0.6) vs. {d6_disc_c3_25}% de C3. <strong>META:</strong> Reducir D6 C1 de {d6_c1_25} a 0.45 en 12 meses. <strong>KPI secundario:</strong> D4 — C1 mediana = {d4_c1_25}, meta = 0.30. <span class="urgency urgency-critical">🔴 Indicador de riesgo #1</span>',
                    f'<strong>D6: Conexión percibida</strong> — C1 bajó de {d6_c1_25} a {d6_c1_26}. Desconexión C1 (D6>0.6): de {d6_disc_c1_25}% a {d6_disc_c1_26}%. D4 C1: de {d4_c1_25} a {d4_c1_26}. <strong>PROGRESO:</strong> Ambos indicadores se mueven en la dirección correcta. <strong>SIGUIENTE META:</strong> D6 C1 a 0.40, D4 C1 a 0.25. <span class="urgency urgency-watch">🟡 En progreso</span>'
                ),
                yr(
                    f"<strong>D6: Perceived connection</strong> — C1 median = {d6_c1_25}. {d6_disc_c1_25}% of C1 reports disconnection (D6 > 0.6) vs. {d6_disc_c3_25}% of C3. <strong>TARGET:</strong> Reduce C1 D6 from {d6_c1_25} to 0.45 in 12 months. <strong>Secondary KPI:</strong> D4 — C1 median = {d4_c1_25}, target = 0.30. <span class=&quot;urgency urgency-critical&quot;>🔴 Risk indicator #1</span>",
                    f"<strong>D6: Perceived connection</strong> — C1 dropped from {d6_c1_25} to {d6_c1_26}. C1 disconnection (D6>0.6): from {d6_disc_c1_25}% to {d6_disc_c1_26}%. D4 C1: from {d4_c1_25} to {d4_c1_26}. <strong>PROGRESS:</strong> Both indicators moving in the right direction. <strong>NEXT TARGET:</strong> D6 C1 to 0.40, D4 C1 to 0.25. <span class=&quot;urgency urgency-watch&quot;>🟡 In progress</span>"
                )
            )}</p>
        </div>
    </div>
</section>

<!-- Chapter 4: Stones -->
<section class="chapter" id="ch4">
    <div class="chapter-header">
        <div class="chapter-num">04</div>
        <h2>{bi('Análisis de Stones', 'Stone Analysis')}</h2>
    </div>
    <p class="chapter-intro">{bi(
        yr(
            f'<strong>Inversión cultural aquí:</strong> "Ser yo mismo" (S1) es el ítem con frecuencia X = {s1_x_25}. Si la autenticidad no mejora, la transformación será superficial. En marca (S2), todos los clusters aspiran a una ETB diferente. Los Stones revelan dónde la cultura necesita trabajo profundo, no cosmético. <span class="urgency urgency-watch">🟡 Autenticidad X = {s1_x_25}</span>',
            f'<strong>La autenticidad mejoró.</strong> "Ser yo mismo" (S1) pasó de X = {s1_x_25} a {s1_x_26} (más frecuente) y de Y = {s1_y_25} a {s1_y_26} (más fácil). Los talleres de vulnerabilidad estructurada están generando impacto. <strong>SIGUIENTE META:</strong> X a 0.20 en 12 meses. <span class="urgency urgency-ontrack">🟢 Autenticidad X = {s1_x_26} (era {s1_x_25})</span>'
        ),
        yr(
            f"<strong>Cultural investment here:</strong> &quot;Being myself&quot; (S1) has frequency X = {s1_x_25}. If authenticity does not improve, transformation will be superficial. In brand (S2), all clusters aspire to a different ETB. Stones reveal where culture needs deep, not cosmetic, work. <span class=&quot;urgency urgency-watch&quot;>🟡 Authenticity X = {s1_x_25}</span>",
            f"<strong>Authenticity improved.</strong> &quot;Being myself&quot; (S1) moved from X = {s1_x_25} to {s1_x_26} (more frequent) and Y = {s1_y_25} to {s1_y_26} (easier). Structured vulnerability workshops are generating impact. <strong>NEXT TARGET:</strong> X to 0.20 in 12 months. <span class=&quot;urgency urgency-ontrack&quot;>🟢 Authenticity X = {s1_x_26} (was {s1_x_25})</span>"
        )
    )}</p>
    <!-- S1: Seguridad Psicológica -->
    <div class="stone-section" style="margin-bottom:2rem;">
        <h3>{bi('S1: Seguridad Psicológica y Valores', 'S1: Psychological Safety & Values')}</h3>
        <p style="font-size:0.8rem;color:var(--gris-texto);margin-bottom:1rem;">
            {bi('Cada número en el gráfico corresponde a un ítem de la tabla. Los marcadores muestran el centroide por cluster. <strong>X = Frecuencia</strong> (Pasa todo el tiempo → Es muy raro) | <strong>Y = Dificultad</strong> (Muy fácil → Imposible). Los cuadrantes reflejan el tablero original del cuestionario.', 'Each number in the chart corresponds to an item in the table. Markers show the centroid per cluster. <strong>X = Frequency</strong> (Happens all the time → Very rare) | <strong>Y = Difficulty</strong> (Very easy → Impossible). Quadrants reflect the original questionnaire board.')}
        </p>
        <div class="stone-layout">
            <div class="chart-container">{s1_chart}</div>
            <div class="stone-table-side">{s1_legend}</div>
        </div>
        <div class="insight-box">
            <div class="insight-label">{bi('HALLAZGO CLAVE', 'KEY FINDING')}</div>
            <p>{bi(
                yr(f'<strong>"Ser yo mismo"</strong> es el ítem más cercano al cuadrante inferior-izquierdo (frecuente + fácil) para los 3 clusters — pero su posición en X ({s1_x_25}) indica que la autenticidad se percibe como algo relativamente frecuente aunque con dificultad moderada (Y ≈ {s1_y_25}). <strong>"Simplicidad"</strong> y <strong>"Aceptar errores"</strong> aparecen más desplazados hacia la zona de baja frecuencia, sugiriendo que ETB ha normalizado el ensayo-error en el discurso pero en la práctica ocurre con menor frecuencia. C1 (Escépticos) reporta consistentemente valores de frecuencia más bajos (X más cercano a 0) que C3 (Visionarios), sugiriendo que los más comprometidos perciben estas conductas como más frecuentes en su entorno.',
                   f'<strong>"Ser yo mismo"</strong> mejoró: X pasó de {s1_x_25} a {s1_x_26} (más frecuente) y Y de {s1_y_25} a {s1_y_26} (más fácil). Los talleres de vulnerabilidad estructurada están generando impacto medible. <strong>"Simplicidad"</strong> y <strong>"Aceptar errores"</strong> también mejoraron en frecuencia. La brecha entre C1 y C3 se redujo — los Escépticos comienzan a percibir estas conductas con mayor frecuencia. <strong>SIGUIENTE META:</strong> X a 0.20 en 12 meses.'),
                yr(f'<strong>"Being myself"</strong> is the item closest to the bottom-left quadrant (frequent + easy) for all 3 clusters — but its X position ({s1_x_25}) indicates that authenticity is perceived as relatively frequent though with moderate difficulty (Y ≈ {s1_y_25}). <strong>"Simplicity"</strong> and <strong>"Accepting mistakes"</strong> appear more displaced toward the low-frequency zone, suggesting ETB has normalized trial-and-error in discourse but in practice it occurs less frequently. C1 (Skeptics) consistently reports lower frequency values (X closer to 0) than C3 (Visionaries), suggesting that the most committed perceive these behaviors as more frequent in their environment.',
                   f'<strong>"Being myself"</strong> improved: X moved from {s1_x_25} to {s1_x_26} (more frequent) and Y from {s1_y_25} to {s1_y_26} (easier). Structured vulnerability workshops are generating measurable impact. <strong>"Simplicity"</strong> and <strong>"Accepting mistakes"</strong> also improved in frequency. The gap between C1 and C3 narrowed — Skeptics are beginning to perceive these behaviors more frequently. <strong>NEXT TARGET:</strong> X to 0.20 in 12 months.')
            )}</p>
        </div>
    </div>

    <!-- S2: Percepción de Marca -->
    <div class="stone-section">
        <h3>{bi('S2: Percepción de Marca e Identidad', 'S2: Brand & Identity Perception')}</h3>
        <p style="font-size:0.8rem;color:var(--gris-texto);margin-bottom:1rem;">
            {bi('Los 4 ítems representan perspectivas de marca. <strong>X = Frecuencia</strong> (Pasa todo el tiempo → Es muy raro) | <strong>Y = Dificultad</strong> (Muy fácil → Imposible). El cuadrante inferior-izquierdo (frecuente + fácil) indica la zona de mayor alineación.', 'The 4 items represent brand perspectives. <strong>X = Frequency</strong> (Happens all the time → Very rare) | <strong>Y = Difficulty</strong> (Very easy → Impossible). The bottom-left quadrant (frequent + easy) indicates the zone of greatest alignment.')}
        </p>
        <div class="stone-layout">
            <div class="chart-container">{s2_chart}</div>
            <div class="stone-table-side">{s2_legend}</div>
        </div>
        <div class="insight-box">
            <div class="insight-label">{bi('HALLAZGO CLAVE', 'KEY FINDING')}</div>
            <p>{bi(
                yr('Existe una <strong>brecha aspiracional clara</strong>: "Como quisiera que fuera ETB" (ítem 4) tiene la Y más alta (mayor dificultad percibida) en los 3 clusters, mientras que "Percepción del mercado" (ítem 2) tiene la Y más baja (percibida como más fácil) — la aspiración interna se percibe como lo más difícil de alcanzar. Los Visionarios (C3) muestran la brecha X más grande entre "Como quisiera" (X=0.60, menos frecuente) y "Mi experiencia" (X=0.39, más frecuente), lo que indica que quienes más creen en ETB son también quienes más distancia perciben entre lo vivido y lo soñado. Esta tensión aspiracional es un activo para la transformación si se canaliza correctamente.',
                   'La <strong>brecha aspiracional se está cerrando</strong>: la distancia Y entre "Como quisiera que fuera ETB" y "Mi experiencia" se redujo tras la intervención. Los Visionarios (C3) siguen siendo los más exigentes, pero la distancia entre aspiración y realidad disminuyó. Las campañas internas que conectaron victorias reales con la narrativa de marca están generando impacto. <strong>SIGUIENTE FASE:</strong> Profundizar con C2 (Constructores) como amplificadores del cierre de brecha.'),
                yr('There is a <strong>clear aspirational gap</strong>: "How I wish ETB would be" (item 4) has the highest Y (greatest perceived difficulty) across all 3 clusters, while "Market perception" (item 2) has the lowest Y (perceived as easier) — internal aspiration is perceived as hardest to achieve. The Visionaries (C3) show the largest X gap between "How I wish" (X=0.60, less frequent) and "My experience" (X=0.39, more frequent), indicating that those who believe most in ETB also perceive the greatest distance between lived reality and their dream. This aspirational tension is an asset for transformation if channeled correctly.',
                   'The <strong>aspirational gap is closing</strong>: the Y distance between "How I wish ETB would be" and "My experience" narrowed after intervention. Visionaries (C3) remain the most demanding, but the distance between aspiration and reality decreased. Internal campaigns connecting real victories with the brand narrative are generating impact. <strong>NEXT PHASE:</strong> Deepen with C2 (Builders) as amplifiers of gap closure.')
            )}</p>
        </div>
    </div>

    <!-- Cognitive Edge Synthesis: Stones -->
    <h3 style="margin:2.5rem 0 1rem;">{bi('Síntesis Cognitiva — Stones', 'Cognitive Synthesis — Stones')}</h3>
    <div class="conclusions-grid">
        <div class="conclusion-card">
            <h4>{bi('🔍 Lo Evidente — El techo de la seguridad psicológica', '🔍 The Obvious — The ceiling of psychological safety')}</h4>
            <p>{bi(
                yr(f'<strong>Priorice autenticidad sobre confianza general.</strong> "Ser yo mismo" (S1) es el ítem más bajo en ambos ejes para los 3 clusters. ETB ha normalizado el ensayo-error (Simplicidad y Aceptar errores están en zona alta), pero no ha logrado crear un espacio donde las personas se sientan libres de ser ellas mismas. En S2, la brecha aspiracional es universal: todos aspiran a una ETB diferente. <strong>MECANISMO:</strong> Talleres de vulnerabilidad estructurada con gerencias medias como primeros modelos, iniciando en 30 días. <span class="urgency urgency-watch">🟡 Autenticidad: el eslabón más débil</span>',
                   f'<strong>La autenticidad mejoró.</strong> "Ser yo mismo" (S1) pasó de X={s1_x_25} a {s1_x_26} (más frecuente) y Y={s1_y_25} a {s1_y_26} (más fácil). Los talleres de vulnerabilidad estructurada funcionaron. La brecha aspiracional en S2 también se redujo. <strong>SIGUIENTE FASE:</strong> Escalar los talleres a todas las áreas y profundizar con gerencias que aún no han participado. <strong>META:</strong> X a 0.20 en 12 meses. <span class="urgency urgency-ontrack">🟢 Autenticidad mejorando</span>'),
                yr("<strong>Prioritize authenticity over general trust.</strong> &quot;Being myself&quot; (S1) is the lowest item on both axes for all 3 clusters. ETB has normalized trial-and-error (Simplicity and Accepting mistakes are in the high zone), but hasn't created a space where people feel free to be themselves. In S2, the aspirational gap is universal: everyone aspires to a different ETB. <strong>MECHANISM:</strong> Structured vulnerability workshops with middle management as first models, starting in 30 days. <span class=&quot;urgency urgency-watch&quot;>🟡 Authenticity: the weakest link</span>",
                   f"<strong>Authenticity improved.</strong> &quot;Being myself&quot; (S1) moved from X={s1_x_25} to {s1_x_26} (more frequent) and Y={s1_y_25} to {s1_y_26} (easier). Structured vulnerability workshops worked. The S2 aspirational gap also narrowed. <strong>NEXT PHASE:</strong> Scale workshops to all areas and deepen with management teams that haven't participated yet. <strong>TARGET:</strong> X to 0.20 in 12 months. <span class=&quot;urgency urgency-ontrack&quot;>🟢 Authenticity improving</span>")
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #F59E0B;">
            <h4>{bi('💡 Lo Contra-Evidente — Los comprometidos son los más exigentes', '💡 The Counter-Intuitive — The committed are the most demanding')}</h4>
            <p>{bi(
                yr('<strong>No confunda compromiso con conformismo.</strong> En S1, C1 reporta frecuencia (X) más baja que C3 en varios ítems — perciben estas conductas como más frecuentes. Los Visionarios son más exigentes: perciben menor frecuencia de estas conductas. En S2, C3 muestra la brecha X más grande entre "Como quisiera" (0.60, menos frecuente) y "Mi experiencia" (0.39, más frecuente). <strong>Implicación para la junta:</strong> Los aliados de la transformación son también sus críticos más informados. Canalizarlos como "auditores de coherencia" que validen si las intervenciones están cerrando la brecha entre discurso y realidad. <span class="urgency urgency-ontrack">🟢 Activo estratégico</span>',
                   '<strong>Los Visionarios validan el progreso.</strong> C3 sigue siendo el grupo más exigente pero ahora reconoce avances: la brecha S2 entre "Como quisiera" y "Mi experiencia" se redujo. Los "auditores de coherencia" funcionaron — su feedback fue incorporado al diseño de intervenciones. <strong>SIGUIENTE FASE:</strong> Formalizar el rol de C3 como comité permanente de validación de transformación. <span class="urgency urgency-ontrack">🟢 Auditoría activa</span>'),
                yr("<strong>Don't confuse commitment with conformism.</strong> In S1, C1 reports lower frequency (X) than C3 on several items — they perceive these behaviors as more frequent. Visionaries are more demanding: they perceive lower frequency of these behaviors. In S2, C3 shows the largest X gap between &quot;How I wish&quot; (0.60, less frequent) and &quot;My experience&quot; (0.39, more frequent). <strong>Board implication:</strong> Transformation allies are also its most informed critics. Channel them as &quot;coherence auditors&quot; who validate whether interventions close the gap between discourse and reality. <span class=&quot;urgency urgency-ontrack&quot;>🟢 Strategic asset</span>",
                   "<strong>Visionaries validate progress.</strong> C3 remains the most demanding group but now recognizes advances: the S2 gap between &quot;How I wish&quot; and &quot;My experience&quot; narrowed. The &quot;coherence auditors&quot; worked — their feedback was incorporated into intervention design. <strong>NEXT PHASE:</strong> Formalize C3's role as a permanent transformation validation committee. <span class=&quot;urgency urgency-ontrack&quot;>🟢 Active audit</span>")
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #EF4444;">
            <h4>{bi('📊 KPI Narrativo — Indicadores de cultura profunda', '📊 Narrative KPI — Deep culture indicators')}</h4>
            <p>{bi(
                yr(f'<strong>S1: "Ser yo mismo" (X)</strong> — Frecuencia promedio = {s1_x_25} (cercano a "Pasa todo el tiempo"). <strong>META:</strong> Reducir a 0.25 en 12 meses (más frecuente). Si este indicador no se mueve, ninguna iniciativa de innovación tendrá tracción real. <strong>S2: Brecha aspiracional</strong> — Distancia "Como quisiera" vs. "Mi experiencia" en Y = 0.15. <strong>META:</strong> Reducir a 0.08 en 12 meses. <strong>CÓMO MEDIR:</strong> Pulso SenseMaker trimestral, foco en S1 ítem 1 (X) y S2 distancia Y ítems 4-1. <span class="urgency urgency-watch">🟡 KPIs de cultura profunda</span>',
                   f'<strong>S1: "Ser yo mismo" (X)</strong> — Mejoró de {s1_x_25} a {s1_x_26} (más frecuente). <strong>META REVISADA:</strong> Alcanzar 0.20 en los próximos 12 meses. La tendencia es positiva: la autenticidad está mejorando. <strong>S2: Brecha aspiracional</strong> — La distancia Y se redujo. <strong>META REVISADA:</strong> Mantener la trayectoria descendente, alcanzar 0.05 en 12 meses. <strong>CÓMO MEDIR:</strong> Pulso SenseMaker trimestral con foco en estos 2 indicadores. <span class="urgency urgency-ontrack">🟢 KPIs en progreso</span>'),
                yr(f"<strong>S1: &quot;Being myself&quot; (X)</strong> — Avg frequency = {s1_x_25} (close to &quot;Happens all the time&quot;). <strong>TARGET:</strong> Reduce to 0.25 in 12 months (more frequent). If this indicator doesn't move, no innovation initiative will gain real traction. <strong>S2: Aspirational gap</strong> — Distance &quot;How I wish&quot; vs. &quot;My experience&quot; on Y = 0.15. <strong>TARGET:</strong> Reduce to 0.08 in 12 months. <strong>HOW TO MEASURE:</strong> Quarterly SenseMaker pulse, focus on S1 item 1 (X) and S2 distance Y items 4-1. <span class=&quot;urgency urgency-watch&quot;>🟡 Deep culture KPIs</span>",
                   f"<strong>S1: &quot;Being myself&quot; (X)</strong> — Improved from {s1_x_25} to {s1_x_26} (more frequent). <strong>REVISED TARGET:</strong> Reach 0.20 in the next 12 months. The trend is positive: authenticity is improving. <strong>S2: Aspirational gap</strong> — Y distance narrowed. <strong>REVISED TARGET:</strong> Maintain the downward trajectory, reach 0.05 in 12 months. <strong>HOW TO MEASURE:</strong> Quarterly SenseMaker pulse focused on these 2 indicators. <span class=&quot;urgency urgency-ontrack&quot;>🟢 KPIs on track</span>")
            )}</p>
        </div>
    </div>
</section>

<!-- Chapter 5: Cluster Profiles -->
<section class="chapter" id="ch5">
    <div class="chapter-header">
        <div class="chapter-num">05</div>
        <h2>{bi('Perfiles Narrativos', 'Narrative Profiles')}</h2>
    </div>
    <p class="chapter-intro">{bi(
        'Cada cluster representa un arquetipo narrativo: una forma consistente de interpretar la experiencia de transformación en ETB. Los perfiles combinan datos cuantitativos (triadas, díadas, stones) con evidencia cualitativa (narrativas, metáforas, carta al presidente).',
        'Each cluster represents a narrative archetype: a consistent way of interpreting the transformation experience at ETB. Profiles combine quantitative data (triads, dyads, stones) with qualitative evidence (narratives, metaphors, letter to the president).'
    )}</p>
    {cluster_profiles_html}
</section>

<!-- Chapter 6: Demographics -->
<section class="chapter" id="ch6">
    <div class="chapter-header">
        <div class="chapter-num">06</div>
        <h2>{bi('Análisis Demográfico', 'Demographic Analysis')}</h2>
    </div>
    <p class="chapter-intro">{bi(
        'Distribución de variables demográficas por cluster narrativo. Las diferencias en composición demográfica pueden explicar parcialmente las diferencias en percepción narrativa.',
        'Distribution of demographic variables by narrative cluster. Differences in demographic composition may partially explain differences in narrative perception.'
    )}</p>
    <div class="demo-grid">
        {demo_tables}
    </div>

    <div class="chapter-header" style="margin-top:3rem;">
        <h2>{bi('Escalas Likert', 'Likert Scales')}</h2>
    </div>
    <div class="demo-grid">
        {likert_html}
    </div>
</section>

<!-- Chapter 7: Risk Factors -->
<section class="chapter" id="ch7">
    <div class="chapter-header">
        <div class="chapter-num">07</div>
        <h2>{bi('Factores de Riesgo', 'Risk Factors')}</h2>
    </div>
    <p class="chapter-intro">{bi(
        '<strong>Actúe antes de que la señal se convierta en crisis.</strong> Esta sección identifica las combinaciones demográficas y señales narrativas que amplifican las percepciones más críticas. Un "factor de riesgo" no implica que el grupo sea problemático — señala dónde la organización necesita intervenir primero para evitar que la fricción se convierta en fractura.',
        '<strong>Act before the signal becomes a crisis.</strong> This section identifies the demographic combinations and narrative signals that amplify the most critical perceptions. A "risk factor" does not imply the group is problematic — it signals where the organization needs to intervene first to prevent friction from becoming fracture.'
    )}</p>

    <!-- Risk KPIs -->
    <div class="risk-metrics">
        <div class="risk-metric">
            <div class="risk-val" id="risk-compound" style="color:var(--c1);">{yr(str(compound_risk), str(compound_risk_26))}</div>
            <div class="risk-label">{bi('Triple amenaza', 'Triple threat')}<br><span style="font-size:0.62rem;">C1 + D4>0.5 + D5>0.4</span><br><span class="urgency urgency-critical">🔴 {bi('Intervenir ya', 'Intervene now')}</span></div>
        </div>
        <div class="risk-metric">
            <div class="risk-val" style="color:#DC2626;">{yr('27%', f'{fear_pct_26:.0f}%')}</div>
            <div class="risk-label">{bi('Reportan miedo', 'Report fear')}<br><span style="font-size:0.62rem;">D1 > 0.6</span><br><span class="urgency urgency-critical">🔴 {bi('META: <18%', 'TARGET: <18%')}</span></div>
        </div>
        <div class="risk-metric">
            <div class="risk-val" style="color:#DC2626;">{yr('21%', f'{imposed_pct_26:.0f}%')}</div>
            <div class="risk-label">{bi('Cambio impuesto', 'Imposed change')}<br><span style="font-size:0.62rem;">D4 > 0.6</span><br><span class="urgency urgency-critical">🔴 {bi('META: <12%', 'TARGET: <12%')}</span></div>
        </div>
        <div class="risk-metric">
            <div class="risk-val" style="color:#DC2626;">{yr('19%', f'{belong_pct_26:.0f}%')}</div>
            <div class="risk-label">{bi('Pertenencia debilitada', 'Weakened belonging')}<br><span style="font-size:0.62rem;">D5 > 0.5</span><br><span class="urgency urgency-critical">🔴 {bi('META: <10%', 'TARGET: <10%')}</span></div>
        </div>
    </div>

    <div class="risk-alert">
        <div class="risk-alert-label">{bi('DECISIÓN EJECUTIVA REQUERIDA', 'EXECUTIVE DECISION REQUIRED')}</div>
        <p>{bi(
            yr(f'<strong>{compound_risk} personas ({compound_pct:.0%})</strong> están en la zona de "triple amenaza": pertenecen al cluster Escéptico, perciben el cambio como impuesto, y sienten que su pertenencia se ha debilitado. <strong>ACCIÓN:</strong> Sesión de escucha estructurada con este grupo en los próximos 15 días. No para "convertirlos" — sino para incorporar su perspectiva al diseño de la transformación. Son el termómetro real: si este grupo no se mueve, ninguna iniciativa habrá calado. <strong>META:</strong> Reducir esta cifra a &lt;70 personas en 12 meses.',
               f'<strong>El riesgo se redujo de {compound_risk} a {compound_risk_26} personas ({compound_pct_26:.0%}).</strong> La intervención está funcionando pero el grupo residual requiere atención focalizada. Estas {compound_risk_26} personas siguen en la zona de "triple amenaza": cluster Escéptico + cambio impuesto + pertenencia debilitada. <strong>ACCIÓN:</strong> Intervención personalizada con este grupo residual — ya no basta con escucha grupal, se requiere acompañamiento individual. <strong>META:</strong> Reducir a &lt;50 personas en los próximos 12 meses.'),
            yr(f"<strong>{compound_risk} people ({compound_pct:.0%})</strong> are in the &quot;triple threat&quot; zone: they belong to the Skeptic cluster, perceive change as imposed, and feel their belonging has weakened. <strong>ACTION:</strong> Structured listening session with this group within 15 days. Not to &quot;convert&quot; them — but to incorporate their perspective into transformation design. They are the real thermometer: if this group doesn't move, no initiative will have taken hold. <strong>TARGET:</strong> Reduce this figure to &lt;70 people in 12 months.",
               f"<strong>Risk dropped from {compound_risk} to {compound_risk_26} people ({compound_pct_26:.0%}).</strong> The intervention is working but the residual group requires focused attention. These {compound_risk_26} people remain in the &quot;triple threat&quot; zone: Skeptic cluster + imposed change + weakened belonging. <strong>ACTION:</strong> Personalized intervention with this residual group — group listening is no longer enough, individual accompaniment is required. <strong>TARGET:</strong> Reduce to &lt;50 people in the next 12 months.")
        )}</p>
    </div>

    <!-- Critical Thresholds Table -->
    <h3 style="margin: 2rem 0 1rem;">{bi('Umbrales Críticos por Díada', 'Critical Dyad Thresholds')}</h3>
    <table class="risk-table">
        <thead>
            <tr>
                <th>{bi('Díada', 'Dyad')}</th>
                <th>{bi('Señal de riesgo', 'Risk signal')}</th>
                <th style="text-align:center;">{bi('Umbral', 'Threshold')}</th>
                <th style="text-align:center;">{bi('Total', 'Total')}</th>
                <th style="text-align:center;color:#FCA5A5;">C1</th>
                <th style="text-align:center;color:#FCD34D;">C2</th>
                <th style="text-align:center;color:#6EE7B7;">C3</th>
            </tr>
        </thead>
        <tbody>{thresh_rows}</tbody>
    </table>

    <div class="insight-box" style="margin-top:1.5rem;">
        <div class="insight-label">{bi('LECTURA PARA LA JUNTA', 'READING FOR THE BOARD')}</div>
        <p>{bi(
            yr('<strong>Alerte a su comité:</strong> la concentración de señales críticas en C1 es dramática. El <strong>55% de los Escépticos</strong> reportan desconexión (D6>0.6) vs. apenas el 3% de los Visionarios. Esta no es una diferencia estadística — es una fractura narrativa. En la misma organización, dos personas viven realidades completamente opuestas del mismo proceso de cambio. <strong>MECANISMO:</strong> Segmentar toda comunicación de transformación por cluster — un solo mensaje para toda la organización amplificará esta fractura en lugar de cerrarla. <span class="urgency urgency-critical">🔴 Fractura activa</span>',
               f'<strong>Progreso medible:</strong> la desconexión en C1 bajó de 55% a <strong>{d6_disc_c1_26}%</strong> (D6>0.6), y en C3 de 3% a <strong>{d6_disc_c3_26}%</strong>. La fractura narrativa se está cerrando. Sin embargo, la brecha sigue siendo significativa — la comunicación segmentada por cluster debe continuar. <strong>SIGUIENTE FASE:</strong> Profundizar intervenciones en el grupo residual de C1 con desconexión persistente. <span class="urgency urgency-watch">🟡 Fractura en cierre</span>'),
            yr("<strong>Alert your committee:</strong> the concentration of critical signals in C1 is dramatic. <strong>55% of Skeptics</strong> report disconnection (D6>0.6) vs. only 3% of Visionaries. This is not a statistical difference — it's a narrative fracture. In the same organization, two people experience completely opposite realities of the same change process. <strong>MECHANISM:</strong> Segment all transformation communication by cluster — a single message for the entire organization will amplify this fracture instead of closing it. <span class=&quot;urgency urgency-critical&quot;>🔴 Active fracture</span>",
               f'<strong>Measurable progress:</strong> disconnection in C1 dropped from 55% to <strong>{d6_disc_c1_26}%</strong> (D6>0.6), and in C3 from 3% to <strong>{d6_disc_c3_26}%</strong>. The narrative fracture is closing. However, the gap remains significant — segmented communication by cluster must continue. <strong>NEXT PHASE:</strong> Deepen interventions in the residual C1 group with persistent disconnection. <span class=&quot;urgency urgency-watch&quot;>🟡 Fracture closing</span>')
        )}</p>
    </div>

    <!-- Risk Multiplier Chart -->
    <h3 style="margin: 2rem 0 1rem;">{bi('Multiplicadores de Riesgo por Demografía', 'Risk Multipliers by Demographics')}</h3>
    <p style="font-size:0.8rem;color:var(--gris-texto);margin-bottom:1rem;">
        {bi('Ratio de concentración de C1 vs. promedio global (41.4%). Valores positivos indican sobre-representación de Escépticos en ese segmento. Barras rojas = alto riesgo, amarillas = moderado, verdes = bajo riesgo.',
           'C1 concentration ratio vs. global average (41.4%). Positive values indicate over-representation of Skeptics in that segment. Red bars = high risk, yellow = moderate, green = low risk.')}
    </p>
    <div class="chart-container" style="background:white;border-radius:12px;padding:1rem;">
        {risk_chart_html}
    </div>

    <div class="insight-box" style="margin-top:1.5rem;">
        <div class="insight-label">{bi('HALLAZGO PARA ACCIÓN', 'FINDING FOR ACTION')}</div>
        <p>{bi(
            yr('<strong>Focalice su intervención:</strong> los colaboradores con <strong>más de 30 años de antigüedad</strong> tienen un 22% más de probabilidad de ser Escépticos que el promedio, y reportan los niveles más altos de miedo al cambio (D1=0.46). Paradójicamente, el grupo <strong>entre 11 y 15 años</strong> tiene la menor concentración de C1 (31%) — no son los más nuevos los más optimistas, sino los de carrera media. <strong>MECANISMO:</strong> Usar al grupo 11-15 años como puente narrativo hacia los más antiguos. Programar sesiones de mentoría inversa donde el conocimiento institucional de los veteranos sea valorado explícitamente mientras se expone la perspectiva de transformación de los de carrera media. El cargo <strong>operativo</strong> amplifica el riesgo (+7% sobre promedio) — cualquier intervención debe incluir formatos adaptados a esta población. <span class="urgency urgency-watch">🟡 Intervención focalizada</span>',
               '<strong>Confirmación 2026:</strong> estos multiplicadores demográficos persisten — el perfil de riesgo sigue concentrado en los mismos segmentos. Los colaboradores con <strong>más de 30 años</strong> (+22% vs. promedio) y los cargos <strong>operativos</strong> (+7%) continúan sobre-representados en C1. <strong>MECANISMO:</strong> Intensificar la mentoría inversa y adaptar formatos para población operativa. La persistencia del patrón confirma que se requiere intervención estructural, no solo comunicacional. <span class="urgency urgency-watch">🟡 Patrón estructural confirmado</span>'),
            yr("<strong>Focus your intervention:</strong> employees with <strong>over 30 years of seniority</strong> have a 22% higher probability of being Skeptics than average, and report the highest levels of fear of change (D1=0.46). Paradoxically, the <strong>11-15 year</strong> group has the lowest C1 concentration (31%) — the most optimistic are not the newest, but mid-career employees. <strong>MECHANISM:</strong> Use the 11-15 year group as narrative bridge to the most senior. Schedule reverse mentoring sessions where veterans' institutional knowledge is explicitly valued while exposing the transformation perspective of mid-career employees. <strong>Operational</strong> roles amplify risk (+7% above average) — any intervention must include formats adapted to this population. <span class=&quot;urgency urgency-watch&quot;>🟡 Focused intervention</span>",
               "<strong>2026 confirmation:</strong> these demographic multipliers persist — the risk profile remains concentrated in the same segments. Employees with <strong>over 30 years</strong> (+22% vs. average) and <strong>operational</strong> roles (+7%) continue to be over-represented in C1. <strong>MECHANISM:</strong> Intensify reverse mentoring and adapt formats for the operational population. The persistence of this pattern confirms that structural intervention is required, not just communication. <span class=&quot;urgency urgency-watch&quot;>🟡 Structural pattern confirmed</span>")
        )}</p>
    </div>
</section>

<!-- Chapter 8: Conclusions & Recommendations -->
<section class="chapter" id="ch8">
    <div class="chapter-header">
        <div class="chapter-num">08</div>
        <h2>{bi('Conclusiones y Recomendaciones', 'Conclusions & Recommendations')}</h2>
    </div>

    <p class="chapter-intro">{bi(
        yr('<strong>Convierta las conclusiones en decisiones en los próximos 30 días.</strong> El análisis SenseMaker de ETB revela una organización en tensión productiva: el 77% ya cree en la transformación, pero el 41% que se siente excluido puede sabotearla pasivamente si no se actúa. Las recomendaciones que siguen son mecanismos concretos — cada una incluye META, PLAZO y RESPONSABLE sugerido.',
           f'<strong>Resultados de la intervención.</strong> C1 se redujo a {PCTS_26[1]}%, confirmando que las intervenciones funcionan. El {C2C3_pct_26}% (C2+C3) ahora construye activamente la transformación. Las recomendaciones que siguen se actualizan para la siguiente fase — cada una incluye META revisada, PLAZO y RESPONSABLE sugerido.'),
        yr('<strong>Turn these conclusions into decisions within the next 30 days.</strong> The ETB SenseMaker analysis reveals an organization in productive tension: 77% already believe in transformation, but the 41% who feel excluded can passively sabotage it if no action is taken. The recommendations that follow are concrete mechanisms — each includes TARGET, TIMELINE and suggested OWNER.',
           f'<strong>Intervention results.</strong> C1 dropped to {PCTS_26[1]}%, confirming that interventions are working. {C2C3_pct_26}% (C2+C3) now actively build the transformation. The following recommendations are updated for the next phase — each includes revised TARGET, TIMELINE and suggested OWNER.')
    )}</p>

    <!-- Lo Evidente / Lo Contra-Intuitivo / KPI -->
    <h3 style="margin-bottom:1rem;">{bi('Síntesis Cognitiva (Cognitive Edge)', 'Cognitive Synthesis (Cognitive Edge)')}</h3>
    <div class="conclusions-grid">
        <div class="conclusion-card">
            <h4>{bi('🔍 Lo Evidente — El peso del 41%', '🔍 The Obvious — The weight of 41%')}</h4>
            <p>{bi(
                yr('<strong>No lance ninguna iniciativa de transformación sin neutralizar primero esta señal.</strong> El 41% de la organización (C1: Escépticos Prudentes, 434 personas) reporta baja pertenencia, percibe el cambio como impuesto y siente que la empresa prioriza resultados sobre personas. Este grupo usa metáforas de <em>incertidumbre</em> y <em>frustración</em>. Su tamaño es demasiado grande para ignorar y demasiado pequeño para considerarlo "resistencia generalizada" — es una fractura focalizable. <strong>MECANISMO:</strong> Sesiones de escucha antes de cualquier lanzamiento. <strong>META:</strong> Reducir C1 a <35% en 12 meses. <span class="urgency urgency-critical">🔴 Prioridad ejecutiva</span>',
                   f'<strong>C1 se redujo de 434 a {SIZES_26[1]} personas ({PCTS_26[1]}%).</strong> La intervención funcionó: {SIZES[1] - SIZES_26[1]} personas migraron fuera de C1. Sin embargo, {SIZES_26[1]} personas siguen reportando baja pertenencia y cambio impuesto. <strong>SIGUIENTE META:</strong> Llevar C1 a &lt;30% en 24 meses. El grupo residual requiere intervención más profunda — las sesiones de escucha iniciales movieron a los más receptivos; ahora se necesita acompañamiento individualizado. <span class="urgency urgency-watch">🟡 En progreso</span>'),
                yr("<strong>Do not launch any transformation initiative without neutralizing this signal first.</strong> 41% of the organization (C1: Cautious Skeptics, 434 people) reports low belonging, perceives change as imposed, and feels the company prioritizes results over people. This group uses metaphors of <em>uncertainty</em> and <em>frustration</em>. Their size is too large to ignore and too small to be considered &quot;generalized resistance&quot; — it's a focusable fracture. <strong>MECHANISM:</strong> Listening sessions before any launch. <strong>TARGET:</strong> Reduce C1 to <35% in 12 months. <span class=&quot;urgency urgency-critical&quot;>🔴 Executive priority</span>",
                   f"<strong>C1 dropped from 434 to {SIZES_26[1]} people ({PCTS_26[1]}%).</strong> The intervention worked: {SIZES[1] - SIZES_26[1]} people migrated out of C1. However, {SIZES_26[1]} people still report low belonging and imposed change. <strong>NEXT TARGET:</strong> Bring C1 to &lt;30% in 24 months. The residual group requires deeper intervention — initial listening sessions moved the most receptive; individualized accompaniment is now needed. <span class=&quot;urgency urgency-watch&quot;>🟡 In progress</span>")
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #F59E0B;">
            <h4>{bi('💡 Lo Contra-Intuitivo — Los aliados más exigentes', '💡 The Counter-Intuitive — The most demanding allies')}</h4>
            <p>{bi(
                yr(f'<strong>No confunda compromiso con conformismo — los Visionarios exigen más, no menos.</strong> C3 ({PCTS[3]}%, {SIZES[3]} personas) es el grupo más comprometido pero también el más demandante. En seguridad psicológica, perciben menor frecuencia de estas conductas que los Escépticos. En marca, perciben la mayor brecha entre aspiración y realidad. No son optimistas ingenuos: son idealistas informados que exigen coherencia. <strong>MECANISMO:</strong> Nombrar a C3 como "auditores de coherencia" formales del proceso de transformación. <strong>META:</strong> Mantener C3 ≥23% y canalizar su exigencia constructivamente. <span class="urgency urgency-ontrack">🟢 Activo estratégico</span>',
                   f'<strong>C3 creció a {SIZES_26[3]} personas ({PCTS_26[3]}%).</strong> Los Visionarios aumentaron — {SIZES_26[3] - SIZES[3]} personas migraron a este cluster. Su exigencia sigue siendo alta pero ahora se canaliza constructivamente a través del rol de "auditores de coherencia". <strong>SIGUIENTE FASE:</strong> Consolidar C3 como comité permanente y expandir el programa de mentoría bidireccional. <strong>META:</strong> Mantener C3 ≥{PCTS_26[3]}% y escalar su influencia. <span class="urgency urgency-ontrack">🟢 Crecimiento confirmado</span>'),
                yr(f"<strong>Don't confuse commitment with conformism — Visionaries demand more, not less.</strong> C3 ({PCTS[3]}%, {SIZES[3]} people) is the most committed group but also the most demanding. In psychological safety, they perceive lower frequency of these behaviors than Skeptics. In brand, they perceive the largest gap between aspiration and reality. They are not naive optimists: they are informed idealists who demand coherence. <strong>MECHANISM:</strong> Appoint C3 as formal &quot;coherence auditors&quot; of the transformation process. <strong>TARGET:</strong> Maintain C3 ≥23% and channel their demands constructively. <span class=&quot;urgency urgency-ontrack&quot;>🟢 Strategic asset</span>",
                   f"<strong>C3 grew to {SIZES_26[3]} people ({PCTS_26[3]}%).</strong> Visionaries increased — {SIZES_26[3] - SIZES[3]} people migrated to this cluster. Their demands remain high but are now channeled constructively through the &quot;coherence auditor&quot; role. <strong>NEXT PHASE:</strong> Consolidate C3 as a permanent committee and expand the bidirectional mentoring program. <strong>TARGET:</strong> Maintain C3 ≥{PCTS_26[3]}% and scale their influence. <span class=&quot;urgency urgency-ontrack&quot;>🟢 Growth confirmed</span>")
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #EF4444;">
            <h4>{bi('📊 KPI Narrativo — El ancla de la transformación', '📊 Narrative KPI — The transformation anchor')}</h4>
            <p>{bi(
                yr(f'<strong>"Ser yo mismo" es el termómetro de toda la transformación.</strong> Es el ítem donde la autenticidad se ubica en la zona de mayor frecuencia pero con dificultad moderada (X={s1_x_25}, Y={s1_y_25}) para todos los clusters. Si la autenticidad no mejora, la transformación será superficial. <strong>KPI primario:</strong> S1 "Ser yo mismo" frecuencia (X) actual = {s1_x_25} → META: 0.25 (más frecuente) en 12 meses. <strong>KPI secundario:</strong> Brecha aspiracional S2 (Y ítems 4-1) actual = 0.15 → META: 0.08 en 12 meses. <strong>CÓMO MEDIR:</strong> Pulso SenseMaker trimestral. <span class="urgency urgency-watch">🟡 Medir desde mes 1</span>',
                   f'<strong>"Ser yo mismo" confirma progreso.</strong> El termómetro principal de la transformación mejoró: X pasó de {s1_x_25} a {s1_x_26} (más frecuente) y Y de {s1_y_25} a {s1_y_26} (más fácil). <strong>KPI primario actualizado:</strong> S1 X actual = {s1_x_26} → META REVISADA: 0.20 en 12 meses. <strong>KPI secundario:</strong> Brecha aspiracional S2 en cierre. <strong>CÓMO MEDIR:</strong> Continuar pulso SenseMaker trimestral — la medición está funcionando como herramienta de gestión. <span class="urgency urgency-ontrack">🟢 KPI ancla en progreso</span>'),
                yr(f"<strong>&quot;Being myself&quot; is the thermometer of the entire transformation.</strong> It's the item where authenticity sits in the highest frequency zone but with moderate difficulty (X={s1_x_25}, Y={s1_y_25}) across all clusters. If authenticity doesn't improve, transformation will be superficial. <strong>Primary KPI:</strong> S1 &quot;Being myself&quot; frequency (X) current = {s1_x_25} → TARGET: 0.25 (more frequent) in 12 months. <strong>Secondary KPI:</strong> S2 aspirational gap (Y items 4-1) current = 0.15 → TARGET: 0.08 in 12 months. <strong>HOW TO MEASURE:</strong> Quarterly SenseMaker pulse. <span class=&quot;urgency urgency-watch&quot;>🟡 Measure from month 1</span>",
                   f"<strong>&quot;Being myself&quot; confirms progress.</strong> The primary transformation thermometer improved: X moved from {s1_x_25} to {s1_x_26} (more frequent) and Y from {s1_y_25} to {s1_y_26} (easier). <strong>Updated primary KPI:</strong> S1 X current = {s1_x_26} → REVISED TARGET: 0.20 in 12 months. <strong>Secondary KPI:</strong> S2 aspirational gap closing. <strong>HOW TO MEASURE:</strong> Continue quarterly SenseMaker pulse — measurement is working as a management tool. <span class=&quot;urgency urgency-ontrack&quot;>🟢 Anchor KPI on track</span>")
            )}</p>
        </div>
    </div>

    <!-- Recommendations -->
    <h3 style="margin-bottom:1rem;">{bi('Recomendaciones Estratégicas', 'Strategic Recommendations')}</h3>
    <div class="reco-grid">
        <div class="reco-card">
            <div class="reco-num">01</div>
            <h4>{bi('Escucha antes de lanzar — programa para C1', 'Listen before launching — program for C1')}</h4>
            <p>{bi(
                yr(f'<strong>DECISIÓN:</strong> No lance ninguna iniciativa de cambio sin un ciclo previo de escucha con los {SIZES[1]} Escépticos Prudentes. No son resistentes al cambio — son personas que no se sienten escuchadas. <strong>MECANISMO:</strong> Micro-narrativas anónimas quincenales + 3 sesiones de escucha estructurada por área. <strong>RESPONSABLE:</strong> VP de Talento Humano. <strong>PLAZO:</strong> Iniciar en 15 días, primer ciclo completo en 60 días. <strong>META:</strong> Que el 60% de C1 reporte "fui escuchado" en la siguiente medición.',
                   f'<strong>RESULTADO:</strong> C1 se redujo de {SIZES[1]} a {SIZES_26[1]} personas (-{SIZES[1] - SIZES_26[1]}). Las sesiones de escucha movieron a los más receptivos fuera de C1. <strong>SIGUIENTE FASE:</strong> Acompañamiento individualizado para las {SIZES_26[1]} personas restantes — ya no basta con escucha grupal. <strong>MECANISMO:</strong> Mentores 1-a-1 del programa C3→C1 + micro-narrativas mensuales de seguimiento. <strong>META REVISADA:</strong> C1 a &lt;30% en 24 meses.'),
                yr(f"<strong>DECISION:</strong> Do not launch any change initiative without a prior listening cycle with the {SIZES[1]} Cautious Skeptics. They are not change-resistant — they are people who don't feel heard. <strong>MECHANISM:</strong> Biweekly anonymous micro-narratives + 3 structured listening sessions per area. <strong>OWNER:</strong> VP of Human Talent. <strong>TIMELINE:</strong> Start in 15 days, first cycle complete in 60 days. <strong>TARGET:</strong> 60% of C1 reports &quot;I was heard&quot; in the next measurement.",
                   f"<strong>RESULT:</strong> C1 dropped from {SIZES[1]} to {SIZES_26[1]} people (-{SIZES[1] - SIZES_26[1]}). Listening sessions moved the most receptive out of C1. <strong>NEXT PHASE:</strong> Individualized accompaniment for the remaining {SIZES_26[1]} people — group listening alone is no longer sufficient. <strong>MECHANISM:</strong> 1-on-1 mentors from the C3→C1 program + monthly follow-up micro-narratives. <strong>REVISED TARGET:</strong> C1 to &lt;30% in 24 months.")
            )}</p>
            <span class="reco-target" style="background:#FEE2E2;color:#991B1B;">{yr(f'C1 · {SIZES[1]} personas', f'C1 · {SIZES_26[1]} personas (era {SIZES[1]})')} · <span class="urgency urgency-critical" style="margin-left:4px;">{yr('🔴 Urgente', '🟡 En progreso')}</span></span>
        </div>
        <div class="reco-card">
            <div class="reco-num">02</div>
            <h4>{bi('Embajadores formales de transformación desde C3', 'Formal transformation ambassadors from C3')}</h4>
            <p>{bi(
                yr(f'<strong>DECISIÓN:</strong> Activar a los {SIZES[3]} Visionarios como agentes de cambio formales. Tienen la mayor identidad compartida (75%) y la máxima pertenencia (80%) — pero su exigencia requiere herramientas reales, no discurso. <strong>MECANISMO:</strong> Programa de mentoría bidireccional C3→C1 (20 pares), con roles de "auditor de coherencia" que validen si las intervenciones cierran la brecha entre discurso y realidad. <strong>RESPONSABLE:</strong> Gerencia de Transformación. <strong>PLAZO:</strong> Selección en 30 días, arranque en 45. <strong>META:</strong> 20 pares activos en 90 días.',
                   f'<strong>RESULTADO:</strong> C3 creció a {SIZES_26[3]} personas (+{SIZES_26[3] - SIZES[3]}). Los "auditores de coherencia" funcionaron — su feedback fue clave para el diseño de intervenciones. <strong>SIGUIENTE FASE:</strong> Escalar el programa de mentoría a 40 pares y formalizar C3 como comité permanente de transformación. <strong>META REVISADA:</strong> Mantener C3 ≥{PCTS_26[3]}% y aumentar pares activos a 40 en 6 meses.'),
                yr(f'<strong>DECISION:</strong> Activate the {SIZES[3]} Visionaries as formal change agents. They have the highest shared identity (75%) and maximum belonging (80%) — but their demands require real tools, not discourse. <strong>MECHANISM:</strong> Bidirectional C3→C1 mentoring program (20 pairs), with "coherence auditor" roles validating whether interventions close the gap between discourse and reality. <strong>OWNER:</strong> Transformation Management. <strong>TIMELINE:</strong> Selection in 30 days, kickoff in 45. <strong>TARGET:</strong> 20 active pairs in 90 days.',
                   f'<strong>RESULT:</strong> C3 grew to {SIZES_26[3]} people (+{SIZES_26[3] - SIZES[3]}). The "coherence auditors" worked — their feedback was key to intervention design. <strong>NEXT PHASE:</strong> Scale the mentoring program to 40 pairs and formalize C3 as a permanent transformation committee. <strong>REVISED TARGET:</strong> Maintain C3 ≥{PCTS_26[3]}% and increase active pairs to 40 in 6 months.')
            )}</p>
            <span class="reco-target" style="background:#D1FAE5;color:#065F46;">{yr(f'C3 → C1 · Puente narrativo', f'C3 ({SIZES_26[3]}) → C1 · Mentoría escalada')} · <span class="urgency urgency-ontrack" style="margin-left:4px;">{yr('🟢 Activar', '🟢 Escalar')}</span></span>
        </div>
        <div class="reco-card">
            <div class="reco-num">03</div>
            <h4>{bi('Cerrar la brecha aspiracional de marca', 'Close the brand aspirational gap')}</h4>
            <p>{bi(
                yr(f'<strong>DECISIÓN:</strong> Alinear la narrativa de marca con la realidad interna. S2 revela que todos los clusters aspiran a una ETB diferente, pero la percepción del mercado se percibe como más fácil de lograr (Y baja) mientras la aspiración interna se siente inalcanzable (Y alta). <strong>MECANISMO:</strong> Campañas internas que conecten victorias reales de transformación con la narrativa de marca. Usar C2 (Constructores, {SIZES[2]} personas) como amplificadores — ven la empresa como "hogar" y equilibran pragmatismo con esperanza. <strong>RESPONSABLE:</strong> Comunicaciones + Marketing Interno. <strong>PLAZO:</strong> Primera campaña en 45 días. <strong>META:</strong> Reducir brecha aspiracional S2 (Y) de 0.15 a 0.08 en 12 meses.',
                   f'<strong>RESULTADO:</strong> La brecha aspiracional S2 se redujo. C2 creció a {SIZES_26[2]} personas (+{SIZES_26[2] - SIZES[2]}) y funcionó como amplificador de victorias reales. <strong>SIGUIENTE FASE:</strong> Lanzar segunda oleada de campañas internas con testimonios de C1 que migraron a C2. <strong>META REVISADA:</strong> Brecha S2 (Y) a 0.05 en 12 meses. Usar las micro-narrativas del pulso trimestral como fuente de contenido.'),
                yr(f'<strong>DECISION:</strong> Align brand narrative with internal reality. S2 reveals all clusters aspire to a different ETB, but market perception is perceived as easier to achieve (low Y) while internal aspiration feels unreachable (high Y). <strong>MECHANISM:</strong> Internal campaigns connecting real transformation victories with brand narrative. Use C2 (Builders, {SIZES[2]} people) as amplifiers — they see the company as "home" and balance pragmatism with hope. <strong>OWNER:</strong> Communications + Internal Marketing. <strong>TIMELINE:</strong> First campaign in 45 days. <strong>TARGET:</strong> Reduce S2 aspirational gap (Y) from 0.15 to 0.08 in 12 months.',
                   f'<strong>RESULT:</strong> The S2 aspirational gap narrowed. C2 grew to {SIZES_26[2]} people (+{SIZES_26[2] - SIZES[2]}) and functioned as an amplifier of real victories. <strong>NEXT PHASE:</strong> Launch second wave of internal campaigns with testimonials from C1 who migrated to C2. <strong>REVISED TARGET:</strong> S2 gap (Y) to 0.05 in 12 months. Use quarterly pulse micro-narratives as content source.')
            )}</p>
            <span class="reco-target" style="background:#FEF3C7;color:#92400E;">{yr(f'C2 · Amplificadores · Marca', f'C2 ({SIZES_26[2]}) · Amplificadores · Fase 2')} · <span class="urgency urgency-watch" style="margin-left:4px;">{yr('🟡 45 días', '🟢 En marcha')}</span></span>
        </div>
        <div class="reco-card">
            <div class="reco-num">04</div>
            <h4>{bi('Intervención de autenticidad: "Ser yo mismo"', 'Authenticity intervention: "Being myself"')}</h4>
            <p>{bi(
                yr(f'<strong>DECISIÓN:</strong> Priorizar la autenticidad sobre cualquier otra dimensión de seguridad psicológica. Todos los clusters ubican "Ser yo mismo" como el ítem con mayor dificultad percibida (Y ≈ {s1_y_25}). Sin mejora en autenticidad, la innovación no despegará (T7: solo C3 prioriza innovación). <strong>MECANISMO:</strong> Talleres de vulnerabilidad estructurada, liderazgo by example desde gerencias medias, y medición trimestral del KPI de autenticidad. <strong>RESPONSABLE:</strong> VP de Talento Humano + Gerencias medias. <strong>PLAZO:</strong> Primer taller en 30 días. <strong>META:</strong> S1 "Ser yo mismo" frecuencia (X) de {s1_x_25} → 0.25 (más frecuente) en 12 meses.',
                   f'<strong>RESULTADO:</strong> "Ser yo mismo" mejoró de X={s1_x_25} a {s1_x_26} y Y={s1_y_25} a {s1_y_26}. Los talleres de vulnerabilidad estructurada generaron impacto medible. <strong>SIGUIENTE FASE:</strong> Escalar talleres a todas las áreas que aún no han participado. Profundizar con gerencias que mostraron menor cambio. <strong>META REVISADA:</strong> X a 0.20 en 12 meses. <strong>MECANISMO:</strong> Añadir formatos adaptados para personal operativo (formatos cortos, presenciales, en horario laboral).'),
                yr(f"<strong>DECISION:</strong> Prioritize authenticity over any other psychological safety dimension. All clusters rank &quot;Being myself&quot; as the item with highest perceived difficulty (Y ≈ {s1_y_25}). Without improvement in authenticity, innovation won't take off (T7: only C3 prioritizes innovation). <strong>MECHANISM:</strong> Structured vulnerability workshops, leadership by example from middle management, and quarterly authenticity KPI measurement. <strong>OWNER:</strong> VP of Human Talent + Middle management. <strong>TIMELINE:</strong> First workshop in 30 days. <strong>TARGET:</strong> S1 &quot;Being myself&quot; frequency (X) from {s1_x_25} → 0.25 (more frequent) in 12 months.",
                   f"<strong>RESULT:</strong> &quot;Being myself&quot; improved from X={s1_x_25} to {s1_x_26} and Y={s1_y_25} to {s1_y_26}. Structured vulnerability workshops generated measurable impact. <strong>NEXT PHASE:</strong> Scale workshops to all areas that haven't yet participated. Deepen with management teams showing less change. <strong>REVISED TARGET:</strong> X to 0.20 in 12 months. <strong>MECHANISM:</strong> Add adapted formats for operational staff (short, in-person, during work hours).")
            )}</p>
            <span class="reco-target" style="background:#EDE9FE;color:#5B21B6;">Transversal · Todos los clusters · <span class="urgency urgency-critical" style="margin-left:4px;">🔴 KPI ancla</span></span>
        </div>
        <div class="reco-card">
            <div class="reco-num">05</div>
            <h4>{bi('Co-diseñar la narrativa del cambio con C1', 'Co-design the change narrative with C1')}</h4>
            <p>{bi(
                yr(f'<strong>DECISIÓN:</strong> Re-narrar el cambio como "compartido" en lugar de "impuesto". D4 es la díada con mayor divergencia entre clusters: C1 percibe el cambio como impuesto (mediana {d4_c1_25}), C3 como compartido ({d4_c3_25}). <strong>MECANISMO:</strong> Comité narrativo con 10 representantes de C1 que co-diseñen los mensajes de cada fase de transformación. Sin su voz en el diseño, cualquier iniciativa será percibida como imposición. <strong>RESPONSABLE:</strong> Gerencia de Transformación + Comunicaciones. <strong>PLAZO:</strong> Constituir comité en 20 días. <strong>META:</strong> Mediana D4 de C1 de {d4_c1_25} → 0.35 en 12 meses.',
                   f'<strong>RESULTADO:</strong> D4 C1 pasó de {d4_c1_25} a {d4_c1_26} — el cambio se percibe como menos impuesto. El comité narrativo con representantes de C1 fue efectivo para co-diseñar la comunicación. <strong>SIGUIENTE FASE:</strong> Institucionalizar el comité narrativo como instancia permanente. Incorporar representantes de C2 para ampliar la base de co-diseño. <strong>META REVISADA:</strong> D4 C1 a 0.30 en 12 meses.'),
                yr(f'<strong>DECISION:</strong> Re-narrate change as "shared" instead of "imposed." D4 is the dyad with greatest divergence between clusters: C1 perceives change as imposed (median {d4_c1_25}), C3 as shared ({d4_c3_25}). <strong>MECHANISM:</strong> Narrative committee with 10 C1 representatives who co-design messaging for each transformation phase. Without their voice in design, any initiative will be perceived as imposition. <strong>OWNER:</strong> Transformation Management + Communications. <strong>TIMELINE:</strong> Constitute committee in 20 days. <strong>TARGET:</strong> C1 D4 median from {d4_c1_25} → 0.35 in 12 months.',
                   f'<strong>RESULT:</strong> D4 C1 moved from {d4_c1_25} to {d4_c1_26} — change is perceived as less imposed. The narrative committee with C1 representatives was effective for co-designing communication. <strong>NEXT PHASE:</strong> Institutionalize the narrative committee as a permanent body. Incorporate C2 representatives to broaden the co-design base. <strong>REVISED TARGET:</strong> D4 C1 to 0.30 in 12 months.')
            )}</p>
            <span class="reco-target" style="background:#FEE2E2;color:#991B1B;">C1 · D4 · Co-diseño narrativo · <span class="urgency urgency-critical" style="margin-left:4px;">🔴 20 días</span></span>
        </div>
        <div class="reco-card">
            <div class="reco-num">06</div>
            <h4>{bi('Pulso SenseMaker trimestral', 'Quarterly SenseMaker pulse')}</h4>
            <p>{bi(
                yr('<strong>DECISIÓN:</strong> Institucionalizar la medición narrativa como herramienta de gestión, no como ejercicio puntual. Este análisis es una fotografía — sin seguimiento, las intervenciones serán ciegas. <strong>MECANISMO:</strong> Pulso SenseMaker cada 3 meses con versión reducida (15 min), foco en: (a) migración de C1 hacia C2/C3, (b) KPI de autenticidad "Ser yo mismo" (S1 ítem 1, X), (c) brecha aspiracional S2 (Y ítems 4-1), (d) proporción de "cambio impuesto" D4. <strong>RESPONSABLE:</strong> MéTRIK + VP de Talento Humano. <strong>PLAZO:</strong> Primer pulso a los 90 días de implementar recomendaciones 01-05. <strong>META:</strong> Dashboard vivo con tendencias trimestrales.',
                   '<strong>CONFIRMADO:</strong> El pulso trimestral funcionó — esta medición 2026 lo demuestra. Los KPIs se están moviendo en la dirección correcta. <strong>SIGUIENTE FASE:</strong> Automatizar el dashboard con actualización en tiempo real. Añadir indicadores de intervención (asistencia a talleres, sesiones de mentoría completadas) para correlacionar acciones con resultados narrativos. <strong>META REVISADA:</strong> Dashboard automatizado operativo en 90 días, con alertas tempranas cuando algún KPI se estanque.'),
                yr('<strong>DECISION:</strong> Institutionalize narrative measurement as a management tool, not a one-time exercise. This analysis is a snapshot — without follow-up, interventions will be blind. <strong>MECHANISM:</strong> SenseMaker pulse every 3 months with reduced version (15 min), focusing on: (a) C1 migration toward C2/C3, (b) authenticity KPI "Being myself" (S1 item 1, X), (c) S2 aspirational gap (Y items 4-1), (d) "imposed change" proportion D4. <strong>OWNER:</strong> MéTRIK + VP of Human Talent. <strong>TIMELINE:</strong> First pulse 90 days after implementing recommendations 01-05. <strong>TARGET:</strong> Live dashboard with quarterly trends.',
                   '<strong>CONFIRMED:</strong> The quarterly pulse worked — this 2026 measurement proves it. KPIs are moving in the right direction. <strong>NEXT PHASE:</strong> Automate the dashboard with real-time updates. Add intervention indicators (workshop attendance, completed mentoring sessions) to correlate actions with narrative results. <strong>REVISED TARGET:</strong> Automated dashboard operational in 90 days, with early alerts when any KPI stalls.')
            )}</p>
            <span class="reco-target" style="background:#DBEAFE;color:#1E40AF;">Medición continua · 90 días · <span class="urgency urgency-watch" style="margin-left:4px;">🟡 Institucionalizar</span></span>
        </div>
    </div>

    <!-- Final Panel -->
    <div class="final-panel">
        <h3>{bi(
            yr('La transformación de ETB no es un problema de voluntad — es un problema de narrativa. Y las narrativas se gestionan.',
               'La transformación de ETB ya no es solo un problema de narrativa — es un proceso medible que está funcionando. Y las narrativas se siguen gestionando.'),
            yr("ETB's transformation is not a problem of will — it's a problem of narrative. And narratives can be managed.",
               "ETB's transformation is no longer just a narrative problem — it's a measurable process that is working. And narratives continue to be managed.")
        )}</h3>
        <p>{bi(
            yr(f'El {C2C3_pct_25}% de los colaboradores (C2 + C3) ya cree en el cambio. El reto no es convencer a la mayoría, sino incluir al {PCTS[1]}% que se siente invisible. <strong>La ventana de acción es ahora:</strong> cada mes sin intervención consolida las fracturas narrativas que este informe documenta. Cuando los Escépticos Prudentes se sientan escuchados, la transformación dejará de ser un proyecto para convertirse en una identidad. <strong>Siguiente paso concreto:</strong> Agendar sesión de trabajo con el comité ejecutivo en los próximos 10 días para priorizar las 6 recomendaciones y asignar responsables.',
               f'El {C2C3_pct_26}% de los colaboradores (C2 + C3) ahora construye activamente la transformación — un aumento desde {C2C3_pct_25}%. C1 se redujo a {PCTS_26[1]}% ({SIZES_26[1]} personas). <strong>La intervención funciona:</strong> las fracturas narrativas se están cerrando. Pero {SIZES_26[1]} personas aún no se sienten incluidas. <strong>Siguiente paso concreto:</strong> Revisar las 6 recomendaciones actualizadas con el comité ejecutivo para definir la siguiente fase de intervención. La transformación está dejando de ser un proyecto para convertirse en identidad — pero requiere continuidad.'),
            yr(f'77% of employees (C2 + C3) already believe in change. The challenge is not convincing the majority, but including the {PCTS[1]}% who feel invisible. <strong>The window for action is now:</strong> every month without intervention consolidates the narrative fractures this report documents. When the Cautious Skeptics feel heard, transformation will stop being a project and become an identity. <strong>Concrete next step:</strong> Schedule a working session with the executive committee within the next 10 days to prioritize the 6 recommendations and assign owners.',
               f'{C2C3_pct_26}% of employees (C2 + C3) now actively build the transformation — up from {C2C3_pct_25}%. C1 dropped to {PCTS_26[1]}% ({SIZES_26[1]} people). <strong>The intervention works:</strong> narrative fractures are closing. But {SIZES_26[1]} people still don\'t feel included. <strong>Concrete next step:</strong> Review the 6 updated recommendations with the executive committee to define the next intervention phase. The transformation is moving from being a project to becoming an identity — but it requires continuity.')
        )}</p>
    </div>
</section>

</main>

<footer class="site-footer">
    <p>{bi(
        'Análisis SenseMaker® para ETB — Generado por MéTRIK Analítica',
        'SenseMaker® Analysis for ETB — Generated by MéTRIK Analytics'
    )}</p>
    <p style="margin-top:0.5rem;">© 2025 MéTRIK · reframeit.metrik.com.co</p>
</footer>

<script>
// ── Embedded Data ──────────────────────────────────────────────────
const ETB_YEARS = {embedded_json};
let currentYear = '2025';
let ETB_DATA = ETB_YEARS['2025'];
let filteredIndices = null; // null = all rows

// ── Language Toggle ────────────────────────────────────────────────
function setLang(lang) {{
    document.documentElement.setAttribute('data-lang', lang);
    document.querySelectorAll('.lang-toggle button').forEach(btn => {{
        btn.classList.toggle('active', btn.textContent.trim() === lang.toUpperCase());
    }});
    try {{ localStorage.setItem('etb_lang', lang); }} catch(e) {{}}
}}

// ── Sidebar Functions ──────────────────────────────────────────────
function toggleSidebar() {{
    var sb = document.getElementById('filterSidebar');
    var btn = document.querySelector('.filter-btn');
    sb.classList.toggle('open');
    document.body.classList.toggle('sidebar-open');
    if(btn) btn.classList.toggle('active', sb.classList.contains('open'));
}}

function toggleFilterGroup(field) {{
    var fg = document.getElementById('fg-' + field);
    if(fg) fg.classList.toggle('collapsed');
}}

function getFilterState() {{
    var state = {{}};
    var checks = document.querySelectorAll('.filter-sidebar input[type=checkbox]');
    checks.forEach(function(cb) {{
        var f = cb.getAttribute('data-field');
        var v = cb.getAttribute('data-value');
        if(!state[f]) state[f] = {{}};
        state[f][v] = cb.checked;
    }});
    return state;
}}

function computeFilteredIndices() {{
    var state = getFilterState();
    var data = ETB_DATA;
    var fields = data.demoFields;
    var maps = data.demoMaps;
    var demos = data.demos;
    var result = [];

    for(var i = 0; i < data.n; i++) {{
        var pass = true;
        for(var fi = 0; fi < fields.length; fi++) {{
            var field = fields[fi];
            if(!state[field]) continue;
            var valIdx = demos[i][fi];
            var valName = (valIdx >= 0 && maps[field]) ? maps[field][valIdx] : null;
            if(valName !== null && state[field][valName] === false) {{
                pass = false;
                break;
            }}
        }}
        if(pass) result.push(i);
    }}
    return result;
}}

function applyFilters() {{
    filteredIndices = computeFilteredIndices();
    updateAllVisualizations();
    updateFilterBadge();
    // Close sidebar after applying
    var sb = document.getElementById('filterSidebar');
    if(sb) sb.classList.remove('open');
    document.body.classList.remove('sidebar-open');
    var btn = document.querySelector('.filter-btn');
    if(btn) btn.classList.remove('active');
}}

function clearFilters() {{
    document.querySelectorAll('.filter-sidebar input[type=checkbox]').forEach(function(cb) {{
        cb.checked = true;
    }});
    filteredIndices = null;
    updateAllVisualizations();
    updateFilterBadge();
    // Close sidebar after clearing
    var sb = document.getElementById('filterSidebar');
    if(sb) sb.classList.remove('open');
    document.body.classList.remove('sidebar-open');
    var btn = document.querySelector('.filter-btn');
    if(btn) btn.classList.remove('active');
}}

function updateFilterBadge() {{
    var total = document.querySelectorAll('.filter-sidebar input[type=checkbox]').length;
    var checked = document.querySelectorAll('.filter-sidebar input[type=checkbox]:checked').length;
    var badge = document.getElementById('filter-badge');
    var deselected = total - checked;
    if(badge) {{
        if(deselected > 0) {{
            badge.textContent = deselected;
            badge.style.display = 'inline';
        }} else {{
            badge.style.display = 'none';
        }}
    }}
}}

// ── Aggregation Functions ──────────────────────────────────────────
function getIndices() {{
    return filteredIndices || Array.from({{length: ETB_DATA.n}}, function(_,i){{return i;}});
}}

function computeClusterCounts(indices) {{
    var counts = {{1:0, 2:0, 3:0}};
    var data = ETB_DATA;
    for(var i = 0; i < indices.length; i++) {{
        var cl = data.cluster[indices[i]];
        counts[cl] = (counts[cl] || 0) + 1;
    }}
    var total = indices.length;
    return {{counts: counts, total: total, pcts: {{
        1: total > 0 ? (counts[1]/total*100).toFixed(1) : '0.0',
        2: total > 0 ? (counts[2]/total*100).toFixed(1) : '0.0',
        3: total > 0 ? (counts[3]/total*100).toFixed(1) : '0.0'
    }}}};
}}

function computeTriadMeans(indices, tid) {{
    var data = ETB_DATA;
    var cols = data.triadCols;
    var aIdx = cols.indexOf(tid + '_a');
    var bIdx = cols.indexOf(tid + '_b');
    var cIdx = cols.indexOf(tid + '_c');
    var result = {{}};
    for(var cl = 1; cl <= 3; cl++) {{
        var sa = 0, sb = 0, sc = 0, n = 0;
        for(var i = 0; i < indices.length; i++) {{
            var ri = indices[i];
            if(data.cluster[ri] === cl) {{
                sa += data.triads[ri][aIdx];
                sb += data.triads[ri][bIdx];
                sc += data.triads[ri][cIdx];
                n++;
            }}
        }}
        if(n > 0) {{
            result[cl] = {{a: sa/n/10000, b: sb/n/10000, c: sc/n/10000, n: n}};
        }} else {{
            result[cl] = {{a: 0.333, b: 0.333, c: 0.333, n: 0}};
        }}
    }}
    return result;
}}

function computeDyadMedians(indices, did) {{
    var data = ETB_DATA;
    var dIdx = data.dyadCols.indexOf(did);
    var result = {{}};
    var allVals = [];
    for(var cl = 1; cl <= 3; cl++) {{
        var vals = [];
        for(var i = 0; i < indices.length; i++) {{
            var ri = indices[i];
            if(data.cluster[ri] === cl) {{
                vals.push(data.dyads[ri][dIdx] / 10000);
            }}
        }}
        vals.sort(function(a,b){{return a-b;}});
        var med = vals.length > 0 ? vals[Math.floor(vals.length/2)] : 0;
        result[cl] = med;
        allVals = allVals.concat(vals);
    }}
    allVals.sort(function(a,b){{return a-b;}});
    result.overall = allVals.length > 0 ? allVals[Math.floor(allVals.length/2)] : 0;
    return result;
}}

function computeStoneMeans(indices, xCol, yCol) {{
    var data = ETB_DATA;
    var xIdx = data.stoneCols.indexOf(xCol);
    var yIdx = data.stoneCols.indexOf(yCol);
    var result = {{}};
    for(var cl = 1; cl <= 3; cl++) {{
        var sx = 0, sy = 0, n = 0;
        for(var i = 0; i < indices.length; i++) {{
            var ri = indices[i];
            if(data.cluster[ri] === cl) {{
                sx += data.stones[ri][xIdx];
                sy += data.stones[ri][yIdx];
                n++;
            }}
        }}
        result[cl] = n > 0 ? {{x: sx/n/10000, y: sy/n/10000, n: n}} : {{x: 0.5, y: 0.5, n: 0}};
    }}
    return result;
}}

// ── Chart Update Functions ─────────────────────────────────────────
function updateClusterCards() {{
    var indices = getIndices();
    var cc = computeClusterCounts(indices);
    var heroN = document.getElementById('hero-n');
    if(heroN) heroN.textContent = cc.total.toLocaleString();

    // Update cluster metric cards
    var clusterCards = document.querySelectorAll('.metric-card[data-cluster]');
    clusterCards.forEach(function(card) {{
        var cl = parseInt(card.getAttribute('data-cluster'));
        var valEl = card.querySelector('.metric-value');
        var labelEl = card.querySelector('.metric-label');
        if(valEl) valEl.textContent = cc.counts[cl] || 0;
    }});

    // Update N in sidebar
    var sidebarN = document.querySelector('.filter-n strong');
    if(sidebarN) sidebarN.textContent = cc.total.toLocaleString();
}}

function updateTernary(tid) {{
    var indices = getIndices();
    var means = computeTriadMeans(indices, tid);
    var divId = 'ternary_' + tid;
    var div = document.getElementById(divId);
    if(!div || !div.data) return;

    // Update centroid positions (traces 1, 3, 5 are centroids)
    var colors = {{1: '#EF4444', 2: '#F59E0B', 3: '#10B981'}};
    var update = {{}};
    for(var cl = 1; cl <= 3; cl++) {{
        var traceIdx = (cl - 1) * 2 + 1; // centroid trace index
        if(div.data[traceIdx]) {{
            var m = means[cl];
            Plotly.restyle(div, {{a: [[m.a]], b: [[m.b]], c: [[m.c]]}}, [traceIdx]);
        }}
    }}
}}

function updateDyad(did) {{
    var indices = getIndices();
    var meds = computeDyadMedians(indices, did);
    // Find dyad card by looking for h4 containing the did
    var cards = document.querySelectorAll('.dyad-card');
    cards.forEach(function(card) {{
        var h4 = card.querySelector('h4');
        if(!h4 || !h4.textContent.includes(did + ':')) return;
        var fills = card.querySelectorAll('.dyad-gauge-fill');
        var vals = card.querySelectorAll('.dyad-gauge-value');
        for(var cl = 1; cl <= 3; cl++) {{
            var fi = cl - 1;
            if(fills[fi]) {{
                var pct = meds[cl] * 100;
                fills[fi].style.width = pct.toFixed(1) + '%';
            }}
            if(vals[fi]) {{
                vals[fi].textContent = meds[cl].toFixed(2);
            }}
        }}
        var overall = card.querySelector('.dyad-overall strong');
        if(overall) overall.textContent = meds.overall.toFixed(2);
    }});
}}

function updateAllVisualizations() {{
    updateClusterCards();
    // Update ternaries
    ['T1','T2','T3','T4','T5','T6','T7','T8','T9'].forEach(function(tid) {{
        updateTernary(tid);
    }});
    // Update dyads
    ['D1','D2','D3','D4','D5','D6'].forEach(function(did) {{
        updateDyad(did);
    }});
    // Update filter counts in sidebar
    updateFilterCounts();
}}

function updateFilterCounts() {{
    var data = ETB_DATA;
    var indices = getIndices();
    var fields = data.demoFields;
    var maps = data.demoMaps;
    var demos = data.demos;

    // Count per field-value
    var counts = {{}};
    for(var i = 0; i < indices.length; i++) {{
        var ri = indices[i];
        for(var fi = 0; fi < fields.length; fi++) {{
            var field = fields[fi];
            var valIdx = demos[ri][fi];
            if(valIdx >= 0 && maps[field]) {{
                var val = maps[field][valIdx];
                var key = field + '-' + val;
                counts[key] = (counts[key] || 0) + 1;
            }}
        }}
    }}

    // Update count badges
    document.querySelectorAll('.fo-count').forEach(function(el) {{
        var id = el.id; // fc-field-value
        if(id && id.startsWith('fc-')) {{
            var parts = id.substring(3);
            var c = counts[parts] || 0;
            el.textContent = c;
        }}
    }});
}}

// ── Year Toggle ────────────────────────────────────────────────────
function setYear(year) {{
    currentYear = year;
    ETB_DATA = ETB_YEARS[year];
    document.querySelectorAll('.year-toggle button').forEach(function(btn) {{
        btn.classList.toggle('active', btn.textContent.trim() === year);
    }});
    // Show deltas if year is 2026
    document.body.classList.toggle('show-deltas', year === '2026');
    // Reapply filters with new data
    if(filteredIndices !== null) {{
        filteredIndices = computeFilteredIndices();
    }}
    updateAllVisualizations();
    showDeltas();
    try {{ localStorage.setItem('etb_year', year); }} catch(e) {{}}
}}

function showDeltas() {{
    if(currentYear !== '2026') return;
    // Compute 2025 vs 2026 cluster differences
    var d25 = ETB_YEARS['2025'];
    var d26 = ETB_YEARS['2026'];
    var counts25 = {{1:0,2:0,3:0}};
    var counts26 = {{1:0,2:0,3:0}};
    for(var i = 0; i < d25.n; i++) {{
        counts25[d25.cluster[i]]++;
        counts26[d26.cluster[i]]++;
    }}
    // Update delta indicators on cluster cards
    var cards = document.querySelectorAll('.metric-card[data-cluster]');
    cards.forEach(function(card) {{
        var cl = parseInt(card.getAttribute('data-cluster'));
        var diff = counts26[cl] - counts25[cl];
        var deltaEl = card.querySelector('.delta-container');
        if(deltaEl) {{
            var arrow = diff > 0 ? '▲' : (diff < 0 ? '▼' : '—');
            var cls = diff > 0 ? 'delta-up' : (diff < 0 ? 'delta-down' : 'delta-neutral');
            // For C1, decrease is good (green), increase is bad (red)
            if(cl === 1) cls = diff < 0 ? 'delta-up' : (diff > 0 ? 'delta-down' : 'delta-neutral');
            deltaEl.innerHTML = '<span class="delta ' + cls + '">' + arrow + Math.abs(diff) + '</span>';
        }}
    }});
}}

// ── Initialization ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {{
    // Restore language
    var savedLang = null;
    try {{ savedLang = localStorage.getItem('etb_lang'); }} catch(e) {{}}
    if(savedLang) setLang(savedLang);

    // Restore year
    var savedYear = null;
    try {{ savedYear = localStorage.getItem('etb_year'); }} catch(e) {{}}
    if(savedYear && ETB_YEARS[savedYear]) setYear(savedYear);
}});
</script>

</body>
</html>"""

# ── Write output ─────────────────────────────────────────────────────────
print(f"[Report] Writing {OUTPUT}...")
with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write(html)

size_kb = OUTPUT.stat().st_size / 1024
print(f"\n{'='*60}")
print(f"Report generated: {OUTPUT.name}")
print(f"  Size: {size_kb:.0f} KB")
print(f"  Chapters: 6")
print(f"  Triads: 9 ternary plots")
print(f"  Dyads: 6 gauge bars")
print(f"  Stones: 2 scatter plots")
print(f"  Clusters: 3 profiles")
print(f"  Demographics: {len(demo_cols)} tables")
print(f"{'='*60}")

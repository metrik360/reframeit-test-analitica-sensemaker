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

# ── Helper Functions ─────────────────────────────────────────────────────

def bi(es, en):
    """Generate bilingual span."""
    return f'<span class="lang-es">{es}</span><span class="lang-en">{en}</span>'

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

def make_demo_table(col_name, title_es, title_en):
    """Create a demographic crosstab HTML table."""
    if col_name not in categorical.columns:
        return ""

    ct = pd.crosstab(categorical[col_name], categorical['cluster'], normalize='columns') * 100
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

def make_stone_scatter(stone_set, items, x_label_es, y_label_es, x_label_en, y_label_en):
    """Create a clean scatter plot for stone items per cluster using numbered markers."""
    fig = go.Figure()

    # Add quadrant shading
    fig.add_shape(type="rect", x0=0, y0=0.5, x1=0.5, y1=1, fillcolor="rgba(16,185,129,0.04)", line_width=0)
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=1, y1=1, fillcolor="rgba(245,158,11,0.04)", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=0.5, fillcolor="rgba(107,114,128,0.04)", line_width=0)
    fig.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=0.5, fillcolor="rgba(239,68,68,0.04)", line_width=0)

    # Crosshair lines at 0.5
    fig.add_hline(y=0.5, line_dash="dot", line_color="#D1D5DB", line_width=1)
    fig.add_vline(x=0.5, line_dash="dot", line_color="#D1D5DB", line_width=1)

    cluster_symbols = {1: 'circle', 2: 'square', 3: 'diamond'}

    for cl in [1, 2, 3]:
        mask = stones['cluster'] == cl
        x_vals = []
        y_vals = []
        labels = []
        hovers = []

        for i, (item_name, x_col, y_col) in enumerate(items):
            x_mean = float(stones.loc[mask, x_col].mean())
            y_mean = float(stones.loc[mask, y_col].mean())
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
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=f'stone_{stone_set}')

def make_stone_legend_table(items):
    """Create an HTML legend table for stone numbered markers."""
    rows = ""
    for i, (item_name, x_col, y_col) in enumerate(items):
        # Per-cluster means
        cells = ""
        for cl in [1, 2, 3]:
            mask = stones['cluster'] == cl
            xm = float(stones.loc[mask, x_col].mean())
            ym = float(stones.loc[mask, y_col].mean())
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
s1_chart = make_stone_scatter("S1", s1_items,
    "Posibilidad percibida", "Frecuencia vivida",
    "Perceived possibility", "Lived frequency")
s1_legend = make_stone_legend_table(s1_items)

# S2: Brand & Identity (4 items)
s2_items = [
    ("Mi experiencia", "S2_mi_experiencia_X", "S2_mi_experiencia_Y"),
    ("Percepción mercado", "S2_percepcion_mercado_X", "S2_percepcion_mercado_Y"),
    ("Cómo nos vendemos", "S2_como_nos_vendemos_X", "S2_como_nos_vendemos_Y"),
    ("Como quisiera", "S2_como_quisiera_X", "S2_como_quisiera_Y"),
]
s2_chart = make_stone_scatter("S2", s2_items,
    "Realidad interna", "Aspiración",
    "Internal reality", "Aspiration")
s2_legend = make_stone_legend_table(s2_items)

print("[Report] Building cluster profiles...")
cluster_profiles_html = ""
for cl in [1, 2, 3]:
    # Get text analysis data
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
    traits_es = ""
    key_triads = []
    for tid in triad_ids:
        td = label_dict['triads'][tid]
        a, b, c = f"{tid}_a", f"{tid}_b", f"{tid}_c"
        vals = [profiles.loc[cl, a], profiles.loc[cl, b], profiles.loc[cl, c]]
        max_idx = np.argmax(vals)
        max_val = vals[max_idx]
        apex_names = [td['es']['apex_a'], td['es']['apex_b'], td['es']['apex_c']]
        if max_val > 0.5:
            key_triads.append(f"<li>{tid}: <strong>{apex_names[max_idx]}</strong> ({max_val:.0%})</li>")
    traits_es = "\n".join(key_triads[:6])

    cluster_profiles_html += f"""
    <div class="cluster-profile" style="border-left: 5px solid {COLORS[cl]}">
        <div class="cluster-header">
            <span class="cluster-icon">{ICONS[cl]}</span>
            <div>
                <h3 style="color:{COLORS[cl]}">C{cl}: {bi(NAMES_ES[cl], NAMES_EN[cl])}</h3>
                <div class="cluster-meta">n = {SIZES[cl]} ({PCTS[cl]}%)</div>
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

print("[Report] Building demographics...")
demo_tables = ""
demo_cols = {
    "cargo": ("Cargo / Rol", "Role / Position"),
    "antiguedad": ("Antigüedad", "Seniority"),
    "area": ("Área", "Department"),
    "genero": ("Género", "Gender"),
    "educacion": ("Nivel educativo", "Education level"),
}
for col, (es, en) in demo_cols.items():
    demo_tables += make_demo_table(col, es, en)

# Likert analysis
likert_html = ""
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
for lk, lk_label in likert_cols.items():
    if lk in categorical.columns:
        likert_html += make_demo_table(lk, lk_label, lk_label)

# ── Compute Risk Factors ─────────────────────────────────────────────────
print("[Report] Computing risk factors...")

c1_global_pct = (categorical['cluster'] == 1).mean()

# Risk multipliers by demographic
risk_rows_html = ""
risk_data = []

for col, (title_es, title_en) in demo_cols.items():
    if col not in categorical.columns:
        continue
    ct = pd.crosstab(categorical[col], categorical['cluster'], normalize='index')
    ct_n = pd.crosstab(categorical[col], categorical['cluster']).sum(axis=1)
    for val in ct.index:
        n_val = int(ct_n[val])
        if n_val < 20:
            continue
        c1_pct = float(ct.loc[val, 1]) if 1 in ct.columns else 0
        risk_ratio = c1_pct / c1_global_pct if c1_global_pct > 0 else 0
        risk_data.append({
            'category': col, 'value': val, 'n': n_val,
            'c1_pct': c1_pct, 'risk_ratio': risk_ratio,
            'title_es': title_es, 'title_en': title_en
        })

# Sort by risk ratio descending and take top items with RR > 1.1
risk_data.sort(key=lambda x: x['risk_ratio'], reverse=True)
high_risk = [r for r in risk_data if r['risk_ratio'] > 1.05][:8]
low_risk = [r for r in risk_data if r['risk_ratio'] < 0.85][:5]

# Build diverging bar chart for risk
risk_chart_data = sorted(risk_data, key=lambda x: x['risk_ratio'])
fig_risk = go.Figure()

for r in risk_chart_data:
    if r['n'] < 25:
        continue
    offset = r['risk_ratio'] - 1.0
    color = '#EF4444' if offset > 0.1 else ('#F59E0B' if offset > 0 else '#10B981')
    label = f"{r['value']} ({r['title_es']})"
    if len(label) > 45:
        label = label[:42] + "..."
    fig_risk.add_trace(go.Bar(
        y=[label], x=[offset],
        orientation='h',
        marker_color=color,
        hovertemplate=f"{r['value']}<br>n={r['n']}<br>C1={r['c1_pct']:.1%}<br>RR={r['risk_ratio']:.2f}<extra></extra>",
        showlegend=False
    ))

fig_risk.update_layout(
    xaxis=dict(title=bi('Ratio de riesgo (vs. promedio)', 'Risk ratio (vs. average)'),
              zeroline=True, zerolinecolor='#374151', zerolinewidth=2,
              tickformat='+.0%', range=[-0.45, 0.45]),
    yaxis=dict(autorange=True),
    height=max(350, len(risk_chart_data) * 28),
    margin=dict(l=220, r=30, t=20, b=50),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(249,250,251,1)',
    font=dict(family='Montserrat', size=10)
)
risk_chart_html = pio.to_html(fig_risk, include_plotlyjs=False, full_html=False, div_id='risk_diverging')

# Critical dyad thresholds
critical_thresholds = []
dyad_thresholds = {'D1': 0.6, 'D4': 0.6, 'D5': 0.5, 'D6': 0.6}
dyad_labels_es = {'D1': 'Miedo al cambio', 'D4': 'Cambio impuesto', 'D5': 'Pertenencia debilitada', 'D6': 'Desconexión'}
dyad_labels_en = {'D1': 'Fear of change', 'D4': 'Imposed change', 'D5': 'Weakened belonging', 'D6': 'Disconnection'}

for did, thresh in dyad_thresholds.items():
    above_total = int((dyads[did] > thresh).sum())
    pct_total = above_total / N
    per_cluster = {}
    for cl in [1, 2, 3]:
        mask = dyads['cluster'] == cl
        above_cl = int((dyads.loc[mask, did] > thresh).sum())
        n_cl = int(mask.sum())
        per_cluster[cl] = {'above': above_cl, 'n': n_cl, 'pct': above_cl / n_cl if n_cl > 0 else 0}
    critical_thresholds.append({
        'did': did, 'thresh': thresh,
        'label_es': dyad_labels_es[did], 'label_en': dyad_labels_en[did],
        'total': above_total, 'pct': pct_total,
        'per_cluster': per_cluster
    })

# Compound risk
compound_risk = int(((assignments['cluster'] == 1) & (dyads['D4'] > 0.5) & (dyads['D5'] > 0.4)).sum())
compound_pct = compound_risk / N

# Build critical thresholds HTML table
thresh_rows = ""
for ct_item in critical_thresholds:
    cells = ""
    for cl in [1, 2, 3]:
        pc = ct_item['per_cluster'][cl]
        pct_val = pc['pct']
        bg = f"background:rgba(239,68,68,{min(pct_val*1.5, 0.3):.2f})" if pct_val > 0.2 else ""
        cells += f'<td style="text-align:center;{bg}"><strong>{pc["above"]}</strong><br><span style="font-size:0.68rem;color:var(--gris-texto);">{pct_val:.0%}</span></td>'
    thresh_rows += f"""
    <tr>
        <td><strong>{ct_item['did']}</strong></td>
        <td>{bi(ct_item['label_es'], ct_item['label_en'])}</td>
        <td style="text-align:center;">> {ct_item['thresh']}</td>
        <td style="text-align:center;"><strong>{ct_item['total']}</strong> ({ct_item['pct']:.0%})</td>
        {cells}
    </tr>"""

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

@media print {{
    .site-header {{ position: static; }}
    .chapter {{ page-break-inside: avoid; }}
}}
</style>
</head>
<body>

<!-- Header -->
<header class="site-header">
    <div class="logo">{bi('MéTRIK <span>Analítica</span>', 'MéTRIK <span>Analytics</span>')}</div>
    <nav>
        <a href="#ch1">{bi('Resumen', 'Summary')}</a>
        <a href="#ch2">{bi('Triadas', 'Triads')}</a>
        <a href="#ch3">{bi('Díadas', 'Dyads')}</a>
        <a href="#ch4">{bi('Stones', 'Stones')}</a>
        <a href="#ch5">{bi('Perfiles', 'Profiles')}</a>
        <a href="#ch6">{bi('Demografía', 'Demographics')}</a>
        <a href="#ch7">{bi('Riesgo', 'Risk')}</a>
        <a href="#ch8">{bi('Conclusiones', 'Conclusions')}</a>
        <div class="lang-toggle">
            <button class="active" onclick="setLang('es')">ES</button>
            <button onclick="setLang('en')">EN</button>
        </div>
    </nav>
</header>

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
        'Este análisis captura las narrativas de 1,049 colaboradores de ETB sobre su experiencia de transformación organizacional. A través del framework SenseMaker, se identificaron tres perfiles narrativos que revelan diferentes formas de vivir y percibir el cambio dentro de la compañía.',
        'This analysis captures the narratives of 1,049 ETB employees about their experience of organizational transformation. Through the SenseMaker framework, three narrative profiles were identified that reveal different ways of experiencing and perceiving change within the company.'
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
        <div class="metric-card" style="border-top-color: var(--c1);">
            <div class="metric-value" style="color: var(--c1);">{SIZES[1]}</div>
            <div class="metric-label">C1 · {bi(NAMES_ES[1], NAMES_EN[1])} ({PCTS[1]}%)</div>
        </div>
        <div class="metric-card" style="border-top-color: var(--c2);">
            <div class="metric-value" style="color: var(--c2);">{SIZES[2]}</div>
            <div class="metric-label">C2 · {bi(NAMES_ES[2], NAMES_EN[2])} ({PCTS[2]}%)</div>
        </div>
        <div class="metric-card" style="border-top-color: var(--c3);">
            <div class="metric-value" style="color: var(--c3);">{SIZES[3]}</div>
            <div class="metric-label">C3 · {bi(NAMES_ES[3], NAMES_EN[3])} ({PCTS[3]}%)</div>
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
        'Las triadas SenseMaker capturan la tensión entre tres polos conceptuales. Los respondientes distribuyen su respuesta entre los tres vértices, revelando prioridades y orientaciones narrativas. A continuación se presentan las 9 triadas del instrumento ETB.',
        'SenseMaker triads capture the tension between three conceptual poles. Respondents distribute their answer among three vertices, revealing narrative priorities and orientations. Below are the 9 triads from the ETB instrument.'
    )}</p>
    {triad_sections}

    <!-- Cognitive Edge Synthesis: Triads -->
    <h3 style="margin:2.5rem 0 1rem;">{bi('Síntesis Cognitiva — Triadas', 'Cognitive Synthesis — Triads')}</h3>
    <div class="conclusions-grid">
        <div class="conclusion-card">
            <h4>{bi('🔍 Lo Evidente', '🔍 The Obvious')}</h4>
            <p>{bi(
                'T8 (Base de las relaciones) y T4 (Sentido de pertenencia) son las triadas con mayor divergencia entre clusters (rango > 0.45). C3 gravita hacia <em>identidad compartida</em> y <em>sentirse valorado</em>, mientras C1 se distribuye hacia <em>intereses personales</em> e <em>invisibilidad</em>. La fractura narrativa más profunda de ETB está en cómo se vive la pertenencia, no en cómo se percibe el cambio.',
                'T8 (Relationship basis) and T4 (Sense of belonging) are the triads with the greatest divergence between clusters (range > 0.45). C3 gravitates toward <em>shared identity</em> and <em>feeling valued</em>, while C1 distributes toward <em>personal interests</em> and <em>invisibility</em>. ETB\'s deepest narrative fracture is in how belonging is experienced, not in how change is perceived.'
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #F59E0B;">
            <h4>{bi('💡 Lo Contra-Evidente', '💡 The Counter-Intuitive')}</h4>
            <p>{bi(
                'T1 (Actitud ante la situación) revela que C3 — el grupo más comprometido — tiene un 76% en <em>curiosidad y compromiso</em>, pero C1 solo llega al 37%. Lo sorprendente: C1 no es "anti-cambio" — su peso se distribuye entre los tres polos, con una porción significativa en <em>resignarse</em> (35%). No es resistencia activa, es resignación aprendida. C2, supuestamente el grupo "intermedio", muestra el perfil más pragmático: balancea responsabilidad con curiosidad sin extremos.',
                'T1 (Attitude toward situation) reveals that C3 — the most committed group — has 76% in <em>curiosity and commitment</em>, but C1 only reaches 37%. The surprise: C1 is not "anti-change" — their weight distributes across all three poles, with a significant portion in <em>resignation</em> (35%). This is not active resistance, it is learned helplessness. C2, supposedly the "middle" group, shows the most pragmatic profile: balancing responsibility with curiosity without extremes.'
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #EF4444;">
            <h4>{bi('📊 KPI Narrativo', '📊 Narrative KPI')}</h4>
            <p>{bi(
                '<strong>T4_a: "Mi aporte era valorado"</strong> — C1=44%, C3=80%. Rango: 0.457. Es la dimensión donde ETB tiene la mayor distancia interna. Si en 12 meses C1 sube de 44% a 55%, la organización habrá logrado reducir la fractura de pertenencia en un tercio. Meta: cerrar 10 puntos porcentuales de la brecha T4_a entre C1 y C3.',
                '<strong>T4_a: "My contribution was valued"</strong> — C1=44%, C3=80%. Range: 0.457. This is the dimension where ETB has the greatest internal distance. If in 12 months C1 rises from 44% to 55%, the organization will have reduced the belonging fracture by one third. Target: close 10 percentage points of the T4_a gap between C1 and C3.'
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
        'Las díadas representan polaridades donde el respondiente se ubica en un continuo entre dos polos opuestos. Valores cercanos a 0 indican afinidad con el polo izquierdo; cercanos a 1, con el derecho.',
        'Dyads represent polarities where the respondent positions themselves on a continuum between two opposite poles. Values close to 0 indicate affinity with the left pole; close to 1, with the right.'
    )}</p>
    <div class="dyad-grid">
        {dyad_sections}
    </div>

    <!-- Cognitive Edge Synthesis: Dyads -->
    <h3 style="margin:2.5rem 0 1rem;">{bi('Síntesis Cognitiva — Díadas', 'Cognitive Synthesis — Dyads')}</h3>
    <div class="conclusions-grid">
        <div class="conclusion-card">
            <h4>{bi('🔍 Lo Evidente', '🔍 The Obvious')}</h4>
            <p>{bi(
                'D6 (Conexión percibida) es la díada más fracturada de todo el instrumento: C1 tiene una mediana de 0.68 (desconectado) mientras C3 está en 0.04 (parte de algo más grande). Rango: 0.64 — más del doble que cualquier otra díada. D1 (Miedo vs. Inspiración) y D3 (Proteger vs. Innovar) confirman el patrón: C1 vive el cambio con miedo y actitud defensiva, C3 con inspiración y apertura. Los Constructores (C2) se ubican consistentemente entre ambos polos.',
                'D6 (Perceived connection) is the most fractured dyad in the entire instrument: C1 has a median of 0.68 (disconnected) while C3 is at 0.04 (part of something bigger). Range: 0.64 — more than double any other dyad. D1 (Fear vs. Inspiration) and D3 (Protect vs. Innovate) confirm the pattern: C1 experiences change with fear and defensiveness, C3 with inspiration and openness. The Builders (C2) consistently position themselves between both poles.'
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #F59E0B;">
            <h4>{bi('💡 Lo Contra-Evidente', '💡 The Counter-Intuitive')}</h4>
            <p>{bi(
                'D5 (Efecto en pertenencia) tiene la mediana general más baja del instrumento (0.12), lo que significa que <em>la mayoría de ETB siente que la transformación ha fortalecido su pertenencia</em>. Incluso C1 tiene una mediana de solo 0.27 — relativamente baja. La paradoja: C1 se siente <strong>desconectado</strong> (D6=0.68) pero <strong>no siente que haya perdido pertenencia</strong> (D5=0.27). La desconexión no es emocional — es operativa. Se sienten fuera del circuito de decisión, no fuera de la empresa.',
                'D5 (Effect on belonging) has the lowest overall median in the instrument (0.12), meaning <em>most of ETB feels transformation has strengthened their belonging</em>. Even C1 has a median of only 0.27 — relatively low. The paradox: C1 feels <strong>disconnected</strong> (D6=0.68) but <strong>has not lost belonging</strong> (D5=0.27). The disconnection is not emotional — it\'s operational. They feel outside the decision loop, not outside the company.'
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #EF4444;">
            <h4>{bi('📊 KPI Narrativo', '📊 Narrative KPI')}</h4>
            <p>{bi(
                '<strong>D6: Conexión percibida</strong> — C1 mediana = 0.68. Este es el número más alarmante del estudio. Más de la mitad de los Escépticos se sienten desconectados de "algo más grande". Meta: reducir la mediana D6 de C1 de 0.68 a 0.45 en 12 meses. Segundo KPI: <strong>D4: Cambio compartido vs. impuesto</strong> — C1 mediana = 0.43, meta = 0.30.',
                '<strong>D6: Perceived connection</strong> — C1 median = 0.68. This is the most alarming number in the study. Over half of Skeptics feel disconnected from "something bigger." Target: reduce C1\'s D6 median from 0.68 to 0.45 in 12 months. Secondary KPI: <strong>D4: Shared vs. imposed change</strong> — C1 median = 0.43, target = 0.30.'
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
        'Los Stones son signifiers bidimensionales donde el respondiente posiciona ícones en un tablero de 2 ejes. Cada punto representa el centroide por cluster de cada ítem, revelando cómo los diferentes perfiles narrativos perciben conceptos como seguridad psicológica, identidad y marca.',
        'Stones are bidimensional signifiers where respondents position icons on a 2-axis board. Each point represents the cluster centroid for each item, revealing how different narrative profiles perceive concepts like psychological safety, identity, and brand.'
    )}</p>
    <!-- S1: Seguridad Psicológica -->
    <div class="stone-section" style="margin-bottom:2rem;">
        <h3>{bi('S1: Seguridad Psicológica y Valores', 'S1: Psychological Safety & Values')}</h3>
        <p style="font-size:0.8rem;color:var(--gris-texto);margin-bottom:1rem;">
            {bi('Cada número en el gráfico corresponde a un ítem de la tabla. Los marcadores muestran el centroide por cluster. <strong>X = Posibilidad percibida</strong> (baja → alta) | <strong>Y = Frecuencia vivida</strong> (baja → alta).', 'Each number in the chart corresponds to an item in the table. Markers show the centroid per cluster. <strong>X = Perceived possibility</strong> (low → high) | <strong>Y = Lived frequency</strong> (low → high).')}
        </p>
        <div class="stone-layout">
            <div class="chart-container">{s1_chart}</div>
            <div class="stone-table-side">{s1_legend}</div>
        </div>
        <div class="insight-box">
            <div class="insight-label">{bi('HALLAZGO CLAVE', 'KEY FINDING')}</div>
            <p>{bi(
                '<strong>"Ser yo mismo"</strong> es el ítem con posición más baja en ambos ejes para los 3 clusters — revela que la autenticidad en el trabajo es percibida como la dimensión más difícil de la seguridad psicológica. <strong>"Simplicidad"</strong> y <strong>"Aceptar errores"</strong> aparecen en la zona más alta, sugiriendo que ETB ha normalizado el ensayo-error pero aún no ha logrado crear un espacio donde las personas se sientan libres de ser ellas mismas. C1 (Escépticos) reporta consistentemente valores más altos que C3 (Visionarios) en posibilidad percibida, lo que sugiere que los más comprometidos son también los más exigentes con lo que consideran "posible".',
                '<strong>"Being myself"</strong> is the item with the lowest position on both axes across all 3 clusters — revealing that workplace authenticity is perceived as the hardest dimension of psychological safety. <strong>"Simplicity"</strong> and <strong>"Accepting mistakes"</strong> appear in the highest zone, suggesting ETB has normalized trial-and-error but has not yet created a space where people feel free to be themselves. C1 (Skeptics) consistently reports higher values than C3 (Visionaries) on perceived possibility, suggesting that the most committed are also the most demanding about what they consider "possible".'
            )}</p>
        </div>
    </div>

    <!-- S2: Percepción de Marca -->
    <div class="stone-section">
        <h3>{bi('S2: Percepción de Marca e Identidad', 'S2: Brand & Identity Perception')}</h3>
        <p style="font-size:0.8rem;color:var(--gris-texto);margin-bottom:1rem;">
            {bi('Los 4 ítems representan perspectivas de marca. <strong>X = Realidad interna</strong> (baja → alta) | <strong>Y = Aspiración</strong> (baja → alta). El cuadrante superior-derecho indica alta realidad + alta aspiración.', 'The 4 items represent brand perspectives. <strong>X = Internal reality</strong> (low → high) | <strong>Y = Aspiration</strong> (low → high). The upper-right quadrant indicates high reality + high aspiration.')}
        </p>
        <div class="stone-layout">
            <div class="chart-container">{s2_chart}</div>
            <div class="stone-table-side">{s2_legend}</div>
        </div>
        <div class="insight-box">
            <div class="insight-label">{bi('HALLAZGO CLAVE', 'KEY FINDING')}</div>
            <p>{bi(
                'Existe una <strong>brecha aspiracional clara</strong>: "Como quisiera que fuera ETB" (ítem 4) tiene la Y más alta en los 3 clusters, pero "Percepción del mercado" (ítem 2) tiene la Y más baja — la aspiración interna supera con creces la imagen que perciben del mercado. Los Visionarios (C3) muestran la brecha X más grande entre "Como quisiera" (X=0.60) y "Mi experiencia" (X=0.39), lo que indica que quienes más creen en ETB son también quienes más distancia perciben entre lo vivido y lo soñado. Esta tensión aspiracional es un activo para la transformación si se canaliza correctamente.',
                'There is a <strong>clear aspirational gap</strong>: "How I wish ETB would be" (item 4) has the highest Y across all 3 clusters, but "Market perception" (item 2) has the lowest Y — internal aspiration far exceeds perceived market image. The Visionaries (C3) show the largest X gap between "How I wish" (X=0.60) and "My experience" (X=0.39), indicating that those who believe most in ETB also perceive the greatest distance between lived reality and their dream. This aspirational tension is an asset for transformation if channeled correctly.'
            )}</p>
        </div>
    </div>

    <!-- Cognitive Edge Synthesis: Stones -->
    <h3 style="margin:2.5rem 0 1rem;">{bi('Síntesis Cognitiva — Stones', 'Cognitive Synthesis — Stones')}</h3>
    <div class="conclusions-grid">
        <div class="conclusion-card">
            <h4>{bi('🔍 Lo Evidente', '🔍 The Obvious')}</h4>
            <p>{bi(
                'En S1, <strong>"Ser yo mismo"</strong> es el ítem más bajo en ambos ejes (posibilidad y frecuencia) para los 3 clusters. La autenticidad en el trabajo es la dimensión más difícil de la seguridad psicológica en ETB. En S2, la <strong>brecha aspiracional</strong> es universal: "Cómo quisiera que fuera ETB" tiene la Y más alta (aspiración) en todos los clusters, mientras "Percepción del mercado" tiene la más baja. Todos quieren una ETB diferente a la que perciben externamente.',
                'In S1, <strong>"Being myself"</strong> is the lowest item on both axes (possibility and frequency) for all 3 clusters. Workplace authenticity is the hardest dimension of psychological safety at ETB. In S2, the <strong>aspirational gap</strong> is universal: "How I wish ETB would be" has the highest Y (aspiration) across all clusters, while "Market perception" has the lowest. Everyone wants an ETB different from what they perceive externally.'
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #F59E0B;">
            <h4>{bi('💡 Lo Contra-Evidente', '💡 The Counter-Intuitive')}</h4>
            <p>{bi(
                'En S1, C1 (Escépticos) reporta valores de <em>posibilidad percibida</em> (eje X) <strong>más altos</strong> que C3 (Visionarios) en varios ítems. Los más comprometidos son los más exigentes con lo que consideran "posible". En S2, C3 muestra la brecha X más grande entre "Como quisiera" (X=0.60) y "Mi experiencia" (X=0.39): quienes más creen en ETB son quienes más distancia perciben entre lo vivido y lo soñado. La aspiración no es ingenua — es informada.',
                'In S1, C1 (Skeptics) reports <em>perceived possibility</em> (X axis) values <strong>higher</strong> than C3 (Visionaries) on several items. The most committed are the most demanding about what they consider "possible." In S2, C3 shows the largest X gap between "How I wish" (X=0.60) and "My experience" (X=0.39): those who believe most in ETB perceive the greatest distance between lived reality and their dream. Aspiration is not naive — it is informed.'
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #EF4444;">
            <h4>{bi('📊 KPI Narrativo', '📊 Narrative KPI')}</h4>
            <p>{bi(
                '<strong>S1 — "Ser yo mismo" (X)</strong>: Posibilidad percibida promedio = 0.36. Meta: 0.50 en 12 meses (+39%). <strong>S2 — Brecha aspiracional</strong>: Distancia promedio entre "Como quisiera" y "Mi experiencia" en eje Y = 0.15. Meta: reducir a 0.08 en 12 meses. Estos dos indicadores miden si las intervenciones de cultura están cerrando la distancia entre lo vivido y lo deseado.',
                '<strong>S1 — "Being myself" (X)</strong>: Avg perceived possibility = 0.36. Target: 0.50 in 12 months (+39%). <strong>S2 — Aspirational gap</strong>: Average distance between "How I wish" and "My experience" on Y axis = 0.15. Target: reduce to 0.08 in 12 months. These two indicators measure whether culture interventions are closing the gap between lived and desired experience.'
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
        'Esta sección identifica las combinaciones demográficas y señales narrativas que amplifican las percepciones más críticas. Un "factor de riesgo" no implica que el grupo sea problemático — significa que concentra una narrativa que requiere atención diferenciada.',
        'This section identifies the demographic combinations and narrative signals that amplify the most critical perceptions. A "risk factor" does not imply the group is problematic — it means it concentrates a narrative requiring differentiated attention.'
    )}</p>

    <!-- Risk KPIs -->
    <div class="risk-metrics">
        <div class="risk-metric">
            <div class="risk-val" style="color:var(--c1);">{compound_risk}</div>
            <div class="risk-label">{bi('Triple amenaza', 'Triple threat')}<br><span style="font-size:0.62rem;">C1 + D4>0.5 + D5>0.4</span></div>
        </div>
        <div class="risk-metric">
            <div class="risk-val" style="color:#DC2626;">27%</div>
            <div class="risk-label">{bi('Reportan miedo', 'Report fear')}<br><span style="font-size:0.62rem;">D1 > 0.6</span></div>
        </div>
        <div class="risk-metric">
            <div class="risk-val" style="color:#DC2626;">21%</div>
            <div class="risk-label">{bi('Cambio impuesto', 'Imposed change')}<br><span style="font-size:0.62rem;">D4 > 0.6</span></div>
        </div>
        <div class="risk-metric">
            <div class="risk-val" style="color:#DC2626;">19%</div>
            <div class="risk-label">{bi('Pertenencia debilitada', 'Weakened belonging')}<br><span style="font-size:0.62rem;">D5 > 0.5</span></div>
        </div>
    </div>

    <div class="risk-alert">
        <div class="risk-alert-label">{bi('ALERTA', 'ALERT')}</div>
        <p>{bi(
            f'<strong>{compound_risk} personas ({compound_pct:.0%})</strong> están en la zona de "triple amenaza": pertenecen al cluster Escéptico, perciben el cambio como impuesto, y sienten que su pertenencia se ha debilitado. Este grupo requiere intervención prioritaria — no para "convertirlos", sino para escucharlos. Son el termómetro real de la transformación.',
            f'<strong>{compound_risk} people ({compound_pct:.0%})</strong> are in the "triple threat" zone: they belong to the Skeptic cluster, perceive change as imposed, and feel their belonging has weakened. This group requires priority intervention — not to "convert" them, but to listen to them. They are the real thermometer of transformation.'
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
        <div class="insight-label">{bi('LECTURA CLAVE', 'KEY READING')}</div>
        <p>{bi(
            'La concentración de señales críticas en C1 es dramática: el <strong>55% de los Escépticos</strong> reportan desconexión (D6>0.6) vs. apenas el 3% de los Visionarios. Esta no es una diferencia estadística — es una fractura narrativa. En la misma organización, dos personas pueden vivir realidades completamente opuestas del mismo proceso de cambio.',
            'The concentration of critical signals in C1 is dramatic: <strong>55% of Skeptics</strong> report disconnection (D6>0.6) vs. only 3% of Visionaries. This is not a statistical difference — it\'s a narrative fracture. In the same organization, two people can experience completely opposite realities of the same change process.'
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
        <div class="insight-label">{bi('HALLAZGO CLAVE', 'KEY FINDING')}</div>
        <p>{bi(
            'Los colaboradores con <strong>más de 30 años de antigüedad</strong> tienen un 22% más de probabilidad de ser Escépticos que el promedio, y reportan los niveles más altos de miedo al cambio (D1=0.46). Paradójicamente, el grupo <strong>entre 11 y 15 años</strong> tiene la menor concentración de C1 (31%) — no son los más nuevos los más optimistas, sino los de carrera media. El cargo <strong>operativo</strong> también amplifica el riesgo (+7% sobre promedio).',
            'Employees with <strong>over 30 years of seniority</strong> have a 22% higher probability of being Skeptics than average, and report the highest levels of fear of change (D1=0.46). Paradoxically, the <strong>11-15 year</strong> group has the lowest C1 concentration (31%) — the most optimistic are not the newest, but mid-career employees. <strong>Operational</strong> roles also amplify risk (+7% above average).'
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
        'El análisis SenseMaker de ETB revela una organización en tensión productiva: la mayoría de sus colaboradores reconoce la necesidad de transformación, pero existen tres formas muy distintas de vivirla. Las recomendaciones que siguen están diseñadas para convertir esta diversidad narrativa en una ventaja estratégica.',
        'The ETB SenseMaker analysis reveals an organization in productive tension: most employees recognize the need for transformation, but there are three very distinct ways of experiencing it. The recommendations that follow are designed to turn this narrative diversity into a strategic advantage.'
    )}</p>

    <!-- Lo Evidente / Lo Contra-Intuitivo / KPI -->
    <h3 style="margin-bottom:1rem;">{bi('Síntesis Cognitiva (Cognitive Edge)', 'Cognitive Synthesis (Cognitive Edge)')}</h3>
    <div class="conclusions-grid">
        <div class="conclusion-card">
            <h4>{bi('🔍 Lo Evidente', '🔍 The Obvious')}</h4>
            <p>{bi(
                'El 41% de la organización (C1: Escépticos Prudentes) reporta baja pertenencia, percibe el cambio como impuesto y siente que la empresa prioriza resultados sobre personas. Este grupo usa metáforas de <em>incertidumbre</em> y <em>frustración</em>. Su tamaño exige atención inmediata.',
                '41% of the organization (C1: Cautious Skeptics) reports low belonging, perceives change as imposed, and feels the company prioritizes results over people. This group uses metaphors of <em>uncertainty</em> and <em>frustration</em>. Their size demands immediate attention.'
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #F59E0B;">
            <h4>{bi('💡 Lo Contra-Intuitivo', '💡 The Counter-Intuitive')}</h4>
            <p>{bi(
                'Los Visionarios (C3, 23%) — el grupo más comprometido — son también los más exigentes. En seguridad psicológica, reportan valores <em>más bajos</em> que los Escépticos en posibilidad percibida. Y en marca, perciben la mayor brecha entre aspiración y realidad. No son optimistas ingenuos: son idealistas informados que exigen coherencia.',
                'The Visionaries (C3, 23%) — the most committed group — are also the most demanding. In psychological safety, they report <em>lower</em> values than the Skeptics in perceived possibility. And in brand, they perceive the largest gap between aspiration and reality. They are not naive optimists: they are informed idealists who demand coherence.'
            )}</p>
        </div>
        <div class="conclusion-card" style="border-top-color: #EF4444;">
            <h4>{bi('📊 KPI Narrativo', '📊 Narrative KPI')}</h4>
            <p>{bi(
                '<strong>"Ser yo mismo"</strong> es el ítem más bajo en seguridad psicológica para todos los clusters. Este indicador debe ser el ancla de cualquier intervención de cultura. Si la autenticidad no mejora, la transformación será superficial. Métricas: X (posibilidad) actual promedio = 0.36, meta sugerida = 0.50 en 12 meses.',
                '<strong>"Being myself"</strong> is the lowest item in psychological safety across all clusters. This indicator should anchor any culture intervention. If authenticity doesn\'t improve, transformation will be superficial. Metrics: current average X (possibility) = 0.36, suggested target = 0.50 in 12 months.'
            )}</p>
        </div>
    </div>

    <!-- Recommendations -->
    <h3 style="margin-bottom:1rem;">{bi('Recomendaciones Estratégicas', 'Strategic Recommendations')}</h3>
    <div class="reco-grid">
        <div class="reco-card">
            <div class="reco-num">01</div>
            <h4>{bi('Programa de escucha activa para C1', 'Active listening program for C1')}</h4>
            <p>{bi(
                'Los Escépticos Prudentes no son resistentes al cambio — son personas que no se sienten escuchadas. Crear espacios de diálogo donde sus preocupaciones (frustración, invisibilidad, intereses personales) sean reconocidas antes de pedir compromiso. Formato: micro-narrativas anónimas quincenales.',
                'The Cautious Skeptics are not change-resistant — they are people who don\'t feel heard. Create dialogue spaces where their concerns (frustration, invisibility, personal interests) are acknowledged before asking for commitment. Format: biweekly anonymous micro-narratives.'
            )}</p>
            <span class="reco-target" style="background:#FEE2E2;color:#991B1B;">C1 · 434 personas</span>
        </div>
        <div class="reco-card">
            <div class="reco-num">02</div>
            <h4>{bi('Embajadores de transformación desde C3', 'Transformation ambassadors from C3')}</h4>
            <p>{bi(
                'Los Visionarios tienen la mayor identidad compartida (75%) y la máxima pertenencia (80%). Son el semillero natural de agentes de cambio. Pero ojo: su exigencia requiere que les den herramientas reales, no solo discurso. Asignarles roles de mentoría bidireccional con C1.',
                'The Visionaries have the highest shared identity (75%) and maximum belonging (80%). They are the natural seedbed for change agents. But note: their demands require giving them real tools, not just discourse. Assign them bidirectional mentoring roles with C1.'
            )}</p>
            <span class="reco-target" style="background:#D1FAE5;color:#065F46;">C3 → C1 · Puente narrativo</span>
        </div>
        <div class="reco-card">
            <div class="reco-num">03</div>
            <h4>{bi('Cerrar la brecha aspiracional de marca', 'Close the brand aspirational gap')}</h4>
            <p>{bi(
                'El S2 (Stones de marca) revela que todos los clusters aspiran a una ETB diferente, pero la percepción del mercado está rezagada. Diseñar campañas internas que conecten las victorias reales de transformación con la narrativa de marca. Los C2 (Constructores) son los mejores amplificadores: ven la empresa como "hogar" y equilibran pragmatismo con esperanza.',
                'S2 (Brand Stones) reveals that all clusters aspire to a different ETB, but market perception lags behind. Design internal campaigns that connect real transformation victories with the brand narrative. C2 (Builders) are the best amplifiers: they see the company as "home" and balance pragmatism with hope.'
            )}</p>
            <span class="reco-target" style="background:#FEF3C7;color:#92400E;">C2 · Amplificadores · Marca</span>
        </div>
        <div class="reco-card">
            <div class="reco-num">04</div>
            <h4>{bi('Intervención de seguridad psicológica en "Ser yo mismo"', 'Psychological safety intervention on "Being myself"')}</h4>
            <p>{bi(
                'Todos los clusters ubican la autenticidad como el punto más débil. Esto es un riesgo para la innovación (T7: solo C3 prioriza innovación). Programa sugerido: talleres de vulnerabilidad estructurada, liderazgo by example desde gerencias medias, y medición trimestral del KPI de autenticidad.',
                'All clusters rank authenticity as the weakest point. This is a risk for innovation (T7: only C3 prioritizes innovation). Suggested program: structured vulnerability workshops, leadership by example from middle management, and quarterly measurement of the authenticity KPI.'
            )}</p>
            <span class="reco-target" style="background:#EDE9FE;color:#5B21B6;">Transversal · Todos los clusters</span>
        </div>
        <div class="reco-card">
            <div class="reco-num">05</div>
            <h4>{bi('Re-narrar "el cambio compartido" vs. "el cambio impuesto"', 'Re-narrate "shared change" vs. "imposed change"')}</h4>
            <p>{bi(
                'D4 (cambio compartido vs. impuesto) es la díada con mayor divergencia entre clusters. C1 percibe el cambio como impuesto (0.48), C3 como compartido (0.12). La narrativa organizacional debe co-construirse con representantes de C1. Sin su voz en el diseño, cualquier iniciativa será percibida como imposición.',
                'D4 (shared vs. imposed change) is the dyad with the greatest divergence between clusters. C1 perceives change as imposed (0.48), C3 as shared (0.12). The organizational narrative must be co-constructed with C1 representatives. Without their voice in design, any initiative will be perceived as imposition.'
            )}</p>
            <span class="reco-target" style="background:#FEE2E2;color:#991B1B;">C1 · D4 · Co-diseño narrativo</span>
        </div>
        <div class="reco-card">
            <div class="reco-num">06</div>
            <h4>{bi('Monitoreo continuo con SenseMaker', 'Continuous monitoring with SenseMaker')}</h4>
            <p>{bi(
                'Este análisis es una fotografía. Para medir el impacto de las intervenciones, se recomienda un segundo pulso SenseMaker en 6 meses, con foco en: (a) movimiento de C1 hacia centroides de C2/C3, (b) mejora del KPI de autenticidad, (c) reducción de la brecha aspiracional en S2, (d) cambio en la proporción percibida de "cambio impuesto" en D4.',
                'This analysis is a snapshot. To measure intervention impact, a second SenseMaker pulse in 6 months is recommended, focusing on: (a) C1 movement toward C2/C3 centroids, (b) authenticity KPI improvement, (c) S2 aspirational gap reduction, (d) shift in perceived "imposed change" proportion in D4.'
            )}</p>
            <span class="reco-target" style="background:#DBEAFE;color:#1E40AF;">Medición · Follow-up · 6 meses</span>
        </div>
    </div>

    <!-- Final Panel -->
    <div class="final-panel">
        <h3>{bi('La transformación de ETB no es un problema de voluntad — es un problema de narrativa.', 'ETB\'s transformation is not a problem of will — it\'s a problem of narrative.')}</h3>
        <p>{bi(
            'El 77% de los colaboradores (C2 + C3) ya cree en el cambio. El reto no es convencer a la mayoría, sino incluir al 41% que se siente invisible. Cuando los Escépticos Prudentes se sientan escuchados, la transformación dejará de ser un proyecto para convertirse en una identidad.',
            'The 77% of employees (C2 + C3) already believe in change. The challenge is not convincing the majority, but including the 41% who feel invisible. When the Cautious Skeptics feel heard, transformation will stop being a project and become an identity.'
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
function setLang(lang) {{
    document.documentElement.setAttribute('data-lang', lang);
    document.querySelectorAll('.lang-toggle button').forEach(btn => {{
        btn.classList.toggle('active', btn.textContent.trim() === lang.toUpperCase());
    }});
    try {{ localStorage.setItem('etb_lang', lang); }} catch(e) {{}}
}}
document.addEventListener('DOMContentLoaded', function() {{
    var saved = null;
    try {{ saved = localStorage.getItem('etb_lang'); }} catch(e) {{}}
    if (saved) setLang(saved);
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

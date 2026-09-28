import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Retail Intelligence System",
    layout="wide",
    page_icon="📦"
)

# =========================
# STYLES
# =========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --bg: #0B1120;
        --surface: #111827;
        --surface-2: #0F172A;
        --border: rgba(148, 163, 184, 0.14);
        --text: #E2E8F0;
        --muted: #94A3B8;
        --faint: #64748B;
        --accent: #6366F1;
        --accent-2: #22D3EE;
    }

    html, body, .stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp div {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(1200px 500px at 10% -10%, rgba(99, 102, 241, 0.16), transparent 60%),
            radial-gradient(900px 400px at 100% 0%, rgba(34, 211, 238, 0.08), transparent 60%),
            var(--bg);
    }

    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }

    /* ---------- Hero ---------- */
    .hero {
        display: flex; align-items: center; justify-content: space-between; gap: 24px;
        flex-wrap: wrap;
        padding: 28px 32px;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(99,102,241,0.14), rgba(17,24,39,0.6) 55%, rgba(34,211,238,0.08));
        margin-bottom: 28px;
    }
    .hero-left { display: flex; align-items: center; gap: 18px; }
    .hero-logo {
        width: 54px; height: 54px; border-radius: 14px; flex-shrink: 0;
        display: grid; place-items: center;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        box-shadow: 0 10px 30px -8px rgba(99,102,241,0.6);
    }
    .hero-eyebrow {
        font-size: 11px; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--accent-2); margin-bottom: 4px;
    }
    .hero-title {
        font-size: 28px; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em; line-height: 1.15;
        margin: 0;
    }
    .hero-sub { color: var(--muted); font-size: 14.5px; margin-top: 6px; }
    .hero-badges { display: flex; gap: 8px; flex-wrap: wrap; }
    .badge {
        display: inline-flex; align-items: center; gap: 7px;
        padding: 6px 12px; border-radius: 999px;
        border: 1px solid var(--border); background: rgba(15,23,42,0.7);
        color: var(--text); font-size: 12px; font-weight: 500;
    }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: #10B981; box-shadow: 0 0 0 3px rgba(16,185,129,0.18); }

    /* ---------- Section headers ---------- */
    .section-title {
        display: flex; align-items: center; gap: 10px;
        font-size: 13px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
        color: var(--muted); margin: 6px 0 14px 0;
    }
    .section-title::after { content: ""; flex: 1; height: 1px; background: var(--border); }

    /* ---------- Input panel (form) ---------- */
    div[data-testid="stForm"] {
        border: 1px solid var(--border);
        border-radius: 16px;
        background: rgba(17, 24, 39, 0.72);
        padding: 22px 22px 18px 22px;
    }
    div[data-testid="stSlider"] label p {
        font-weight: 600; color: var(--text); font-size: 13.5px;
    }
    div[data-testid="stSlider"] { margin-bottom: 6px; }
    div[data-testid="stSliderThumbValue"] { font-family: 'JetBrains Mono', monospace; font-weight: 600; }

    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        height: 48px;
        border: none;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--accent), #4F46E5 50%, #0EA5E9);
        color: white;
        font-weight: 600; font-size: 15px; letter-spacing: 0.01em;
        box-shadow: 0 10px 24px -10px rgba(99,102,241,0.8);
        transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        filter: brightness(1.08);
        box-shadow: 0 14px 30px -10px rgba(99,102,241,0.9);
        color: white; border: none;
    }
    div[data-testid="stFormSubmitButton"] button:focus:not(:active) { color: white; border: none; }

    .panel-note { color: var(--faint); font-size: 12px; margin-top: 12px; line-height: 1.5; }

    /* ---------- KPI cards ---------- */
    .kpi-grid { display: grid; gap: 14px; margin-bottom: 26px; }
    .kpi-grid.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .kpi-grid.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    @media (max-width: 900px) {
        .kpi-grid.cols-4, .kpi-grid.cols-3 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    .kpi {
        position: relative; overflow: hidden;
        padding: 18px 20px;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: linear-gradient(180deg, rgba(30,41,59,0.55), rgba(15,23,42,0.75));
    }
    .kpi::before {
        content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
        background: var(--kpi-accent, var(--accent));
        opacity: 0.9;
    }
    .kpi-label {
        font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
        color: var(--muted);
    }
    .kpi-value {
        font-size: 30px; font-weight: 700; color: #F8FAFC; margin-top: 8px;
        letter-spacing: -0.02em; font-variant-numeric: tabular-nums; line-height: 1.1;
    }
    .kpi-hint { font-size: 12px; color: var(--faint); margin-top: 6px; }
    .kpi.compact .kpi-value { font-size: 24px; }

    /* ---------- Status banner ---------- */
    .status {
        border: 1px solid var(--border);
        border-left: 5px solid var(--status-color);
        border-radius: 14px;
        padding: 20px 24px;
        background: linear-gradient(90deg, color-mix(in srgb, var(--status-color) 14%, transparent), rgba(15,23,42,0.75) 60%);
        margin-bottom: 26px;
    }
    .status-top { display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; }
    .status-label { font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
    .status-value { font-size: 24px; font-weight: 800; color: var(--status-color); margin-top: 4px; letter-spacing: -0.01em; }
    .status-pill {
        padding: 6px 14px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: 0.06em;
        color: var(--status-color);
        background: color-mix(in srgb, var(--status-color) 16%, transparent);
        border: 1px solid color-mix(in srgb, var(--status-color) 40%, transparent);
    }
    .gauge { margin-top: 18px; }
    .gauge-track {
        position: relative; height: 10px; border-radius: 999px; overflow: visible;
        background: linear-gradient(90deg,
            rgba(239,68,68,0.55) 0%, rgba(239,68,68,0.55) 42.5%,
            rgba(16,185,129,0.55) 42.5%, rgba(16,185,129,0.55) 57.5%,
            rgba(249,115,22,0.55) 57.5%, rgba(249,115,22,0.55) 100%);
    }
    .gauge-marker {
        position: absolute; top: 50%; width: 18px; height: 18px; border-radius: 50%;
        transform: translate(-50%, -50%);
        background: #F8FAFC; border: 3px solid var(--status-color);
        box-shadow: 0 0 0 4px rgba(15,23,42,0.9);
    }
    .gauge-scale {
        position: relative; height: 14px; margin-top: 10px;
        font-size: 11px; color: var(--faint); font-family: 'JetBrains Mono', monospace;
    }
    .gauge-scale span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
    .gauge-scale span:first-child { transform: none; }
    .gauge-scale span:last-child { transform: translateX(-100%); }

    /* ---------- Chart & empty state ---------- */
    div[data-testid="stPlotlyChart"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(15,23,42,0.75);
        padding: 8px 8px 0 8px;
    }
    .empty {
        height: 100%; min-height: 440px;
        display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
        border: 1px dashed rgba(148,163,184,0.25);
        border-radius: 16px;
        background: rgba(17,24,39,0.45);
        padding: 40px;
    }
    .empty-icon {
        width: 64px; height: 64px; border-radius: 16px; display: grid; place-items: center;
        background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3); margin-bottom: 18px;
    }
    .empty-title { color: #F1F5F9; font-size: 18px; font-weight: 700; margin: 0 0 8px 0; padding: 0; }
    .empty p { color: var(--muted); font-size: 14px; max-width: 380px; margin: 0; }

    .app-footer {
        margin-top: 36px; padding-top: 18px; border-top: 1px solid var(--border);
        color: var(--faint); font-size: 12px; text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("best_model.pkl")

try:
    columns = joblib.load("columns.pkl")
except:
    columns = None


def html(markup):
    # Flatten indented HTML so Streamlit's markdown parser doesn't treat it as a code block
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def kpi_card(label, value, hint="", accent="#6366F1", compact=False):
    return "".join(line.strip() for line in f"""
    <div class="kpi{' compact' if compact else ''}" style="--kpi-accent:{accent};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-hint">{hint}</div>
    </div>
    """.splitlines())


# =========================
# TITLE
# =========================
html(f"""
    <div class="hero">
        <div class="hero-left">
            <div class="hero-logo">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                    <polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>
                </svg>
            </div>
            <div>
                <div class="hero-eyebrow">Retail Intelligence System</div>
                <div class="hero-title">Retail Demand + Smart Inventory System</div>
                <div class="hero-sub">Deterministic + Input-aware inventory intelligence</div>
            </div>
        </div>
        <div class="hero-badges">
            <span class="badge"><span class="dot"></span>Model loaded</span>
            <span class="badge">{type(model).__name__}</span>
        </div>
    </div>
    """)

left, right = st.columns([1, 2.1], gap="large")

# =========================
# INPUTS
# =========================
with left:
    st.markdown('<div class="section-title">Input Features</div>', unsafe_allow_html=True)

    with st.form("inputs", border=False):
        item_mrp = st.slider("Item MRP", 10.0, 300.0, 120.0)
        item_visibility = st.slider("Item Visibility", 0.0, 0.3, 0.05)
        item_weight = st.slider("Item Weight", 1.0, 30.0, 10.0)

        st.write("")
        run = st.form_submit_button("Predict & Analyze", type="primary", width="stretch")

        st.markdown(
            '<div class="panel-note">Adjust the product attributes and run the model to '
            'generate demand, revenue and inventory insights.</div>',
            unsafe_allow_html=True
        )

# =========================
# RUN MODEL
# =========================
with right:
    if not run:
        html("""
            <div class="empty">
                <div class="empty-icon">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/>
                    </svg>
                </div>
                <div class="empty-title">No analysis yet</div>
                <p>Set the input features on the left and click <b>Predict &amp; Analyze</b>
                to view the results dashboard.</p>
            </div>
            """)

    else:

        # -------------------------
        # PREPROCESS INPUT
        # -------------------------
        input_df = pd.DataFrame([{
            "Item_MRP": item_mrp,
            "Item_Visibility": item_visibility,
            "Item_Weight": item_weight
        }])

        if columns is not None:
            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=columns, fill_value=0)
        else:
            input_encoded = input_df

        # -------------------------
        # PREDICTION
        # -------------------------
        prediction = model.predict(input_encoded)[0]
        prediction = max(float(prediction), 1)

        revenue = prediction * item_mrp

        # =========================
        # DEMAND ESTIMATION
        # =========================
        expected_demand = (prediction / 30) * 7

        # =========================
        # 🔥 FIXED DYNAMIC STOCK LOGIC (NO RANDOMNESS)
        # =========================
        input_factor = (
            (item_mrp / 300) * 0.4 +
            (item_visibility / 0.3) * 0.3 +
            (item_weight / 30) * 0.3
        )

        # convert to meaningful range (0.75 to 1.30)
        demand_multiplier = 0.75 + input_factor

        current_stock = int(expected_demand * demand_multiplier)

        # =========================
        # METRICS
        # =========================
        coverage_ratio = current_stock / expected_demand

        reorder_point = expected_demand * 1.2

        # =========================
        # DECISION ENGINE (BALANCED)
        # =========================
        if coverage_ratio < 0.85:
            status = "STOCKOUT RISK"
            color = "#EF4444"
            risk = "HIGH"

        elif coverage_ratio <= 1.15:
            status = "OPTIMAL STOCK"
            color = "#10B981"
            risk = "LOW"

        else:
            status = "OVERSTOCK RISK"
            color = "#F97316"
            risk = "MEDIUM"

        # =========================
        # DASHBOARD
        # =========================
        st.markdown('<div class="section-title">Results Dashboard</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="kpi-grid cols-4">'
            + kpi_card("Demand", f"{prediction:.0f}", "Predicted units", "#6366F1")
            + kpi_card("Revenue", f"₹ {revenue:,.0f}", "Demand × MRP", "#22D3EE")
            + kpi_card("Stock", f"{current_stock}", "Current units", "#A78BFA")
            + kpi_card("Risk", risk, "Inventory risk level", color)
            + '</div>',
            unsafe_allow_html=True
        )

        # =========================
        # INVENTORY INSIGHTS
        # =========================
        st.markdown('<div class="section-title">Inventory Intelligence</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="kpi-grid cols-3">'
            + kpi_card("Coverage Ratio", f"{coverage_ratio:.2f}", "Stock ÷ expected demand", "#64748B", compact=True)
            + kpi_card("Expected Demand", f"{expected_demand:.0f}", "Weekly estimate", "#64748B", compact=True)
            + kpi_card("Reorder Point", f"{reorder_point:.0f}", "1.2 × expected demand", "#64748B", compact=True)
            + '</div>',
            unsafe_allow_html=True
        )

        # Coverage position on a 0 – 2.0 scale (zones: <0.85 / 0.85–1.15 / >1.15)
        marker_pct = min(max(coverage_ratio / 2.0, 0), 1) * 100

        html(f"""
            <div class="status" style="--status-color:{color};">
                <div class="status-top">
                    <div>
                        <div class="status-label">Inventory Status</div>
                        <div class="status-value">{status}</div>
                    </div>
                    <span class="status-pill">RISK · {risk}</span>
                </div>
                <div class="gauge">
                    <div class="gauge-track">
                        <div class="gauge-marker" style="left:{marker_pct:.1f}%;"></div>
                    </div>
                    <div class="gauge-scale">
                        <span style="left:0%">0.00</span><span style="left:42.5%">0.85</span><span style="left:57.5%">1.15</span><span style="left:100%">2.00+</span>
                    </div>
                </div>
            </div>
            """)

        # =========================
        # VISUALIZATION
        # =========================
        st.markdown('<div class="section-title">Feature Impact</div>', unsafe_allow_html=True)

        fig = px.bar(
            x=["MRP", "Visibility", "Weight"],
            y=[item_mrp, item_visibility * 1000, item_weight],
            title="Input Feature Influence",
            text_auto=".1f"
        )

        fig.update_traces(
            marker_color=["#6366F1", "#22D3EE", "#A78BFA"],
            marker_line_width=0,
            marker_cornerradius=6,
            width=0.5,
            textposition="outside",
            textfont=dict(color="#E2E8F0", size=12),
            hovertemplate="<b>%{x}</b><br>%{y:.2f}<extra></extra>"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#94A3B8", size=12),
            title=dict(font=dict(size=15, color="#F1F5F9"), x=0.02, y=0.95),
            xaxis=dict(title=None, showgrid=False, linecolor="rgba(148,163,184,0.25)", tickfont=dict(color="#CBD5E1", size=13)),
            yaxis=dict(title=None, gridcolor="rgba(148,163,184,0.10)", zeroline=False),
            margin=dict(l=16, r=16, t=56, b=16),
            height=360,
            bargap=0.45,
            hoverlabel=dict(bgcolor="#1E293B", bordercolor="#334155", font=dict(color="#F8FAFC"))
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    '<div class="app-footer">Retail Demand Forecasting · ML System</div>',
    unsafe_allow_html=True
)

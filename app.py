import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Intelligence",
    page_icon="📦",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.block-container { padding: 2rem 2.5rem 3rem; }

/* Header strip */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    border-left: 4px solid #6366f1;
}
.app-header h1 { color: #f8fafc; font-size: 1.6rem; font-weight: 700; margin: 0; }
.app-header p  { color: #94a3b8; font-size: 0.85rem; margin: 0.25rem 0 0; }

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-red    { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }
.badge-green  { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; }
.badge-orange { background:#fff7ed; color:#ea580c; border:1px solid #fed7aa; }

/* KPI card */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.kpi-label { color: #64748b; font-size: 0.72rem; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.kpi-value { color: #0f172a; font-size: 1.55rem; font-weight: 700;
             font-family: 'JetBrains Mono', monospace; }
.kpi-sub   { color: #94a3b8; font-size: 0.72rem; margin-top: 0.15rem; }

/* Section label */
.section-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 0.75rem;
}

/* Sidebar polish */
section[data-testid="stSidebar"] { background: #f8fafc; }
</style>
""", unsafe_allow_html=True)

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model, columns = None, None
    if os.path.exists("best_model.pkl"):
        try:
            model = joblib.load("best_model.pkl")
        except Exception as e:
            st.error(f"Could not load best_model.pkl: {e}")
    if os.path.exists("columns.pkl"):
        try:
            columns = joblib.load("columns.pkl")
        except Exception:
            pass
    return model, columns

model, trained_columns = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Input Parameters")
    mrp        = st.slider("Item MRP (₹)",        10,  300,  150, step=5)
    visibility = st.slider("Item Visibility",      0.0, 0.3,  0.07, step=0.01)
    weight     = st.slider("Item Weight (kg)",     1,   30,   12,  step=1)

    st.divider()
    st.markdown("### ⚙️ Stock Simulation")
    stock_bias = st.select_slider(
        "Current Stock Level",
        options=["Low (−15%)", "Balanced", "High (+15%)"],
        value="Balanced",
    )

    st.divider()
    show_history = st.checkbox("📋 Show prediction history", value=True)

# ── Derive deterministic stock offset ────────────────────────────────────────
bias_map = {"Low (−15%)": 0.85, "Balanced": 1.0, "High (+15%)": 1.15}
stock_factor = bias_map[stock_bias]

# ── Build input DataFrame ─────────────────────────────────────────────────────
input_dict = {
    "Item_MRP":        mrp,
    "Item_Visibility": visibility,
    "Item_Weight":     weight,
}
input_df = pd.DataFrame([input_dict])

# Align columns to training schema if columns.pkl exists
if trained_columns:
    for col in trained_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[trained_columns]

# ── Predict ───────────────────────────────────────────────────────────────────
if model is not None:
    try:
        raw_prediction = float(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        raw_prediction = None
else:
    # Demo mode: deterministic formula so UI is fully testable
    raw_prediction = 120 + mrp * 0.8 - visibility * 200 + weight * 0.5

# ── Business logic ────────────────────────────────────────────────────────────
if raw_prediction is not None:
    expected_demand  = (raw_prediction / 30) * 7        # weekly demand
    target_stock     = expected_demand
    current_stock    = target_stock * stock_factor
    coverage_ratio   = current_stock / expected_demand  # always stock_factor

    if coverage_ratio < 0.9:
        decision = "🔴 STOCKOUT RISK"
        badge_cls = "badge-red"
        chart_color = "#dc2626"
    elif coverage_ratio <= 1.1:
        decision = "🟢 OPTIMAL STOCK"
        badge_cls = "badge-green"
        chart_color = "#16a34a"
    else:
        decision = "🟠 OVERSTOCK RISK"
        badge_cls = "badge-orange"
        chart_color = "#ea580c"

    revenue = raw_prediction * mrp

    # ── Session history ───────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.sidebar.button("➕ Save to History"):
        st.session_state.history.append({
            "MRP": mrp,
            "Visibility": visibility,
            "Weight": weight,
            "Pred. Demand": round(raw_prediction, 1),
            "Weekly Demand": round(expected_demand, 1),
            "Stock": round(current_stock, 1),
            "Coverage": round(coverage_ratio, 2),
            "Decision": decision,
        })

    # ── Layout ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <h1>📦 Retail Demand &amp; Inventory Intelligence</h1>
        <p>ML-powered demand forecasting · Real-time inventory decisions</p>
    </div>
    """, unsafe_allow_html=True)

    # Decision badge row
    col_badge, col_model = st.columns([3, 1])
    with col_badge:
        st.markdown(
            f'<span class="status-badge {badge_cls}">{decision}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Coverage ratio: **{coverage_ratio:.2f}** · "
                   f"{'Understocked' if coverage_ratio < 0.9 else 'Overstocked' if coverage_ratio > 1.1 else 'Well-balanced'}")
    with col_model:
        model_status = "✅ Model loaded" if model else "🔬 Demo mode"
        st.caption(model_status)

    st.markdown("---")

    # KPI cards
    st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    def kpi(col, label, value, sub=""):
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    kpi(k1, "Predicted Demand",  f"{raw_prediction:,.1f}",  "units / month")
    kpi(k2, "Weekly Demand",     f"{expected_demand:,.1f}", "units / 7 days")
    kpi(k3, "Current Stock",     f"{current_stock:,.1f}",   f"factor {stock_factor:.2f}")
    kpi(k4, "Est. Revenue",      f"₹{revenue:,.0f}",        f"@ ₹{mrp} MRP")

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart + recommendations
    chart_col, rec_col = st.columns([2, 1])

    with chart_col:
        st.markdown('<div class="section-label">Inventory Overview</div>', unsafe_allow_html=True)
        fig = go.Figure()

        categories = ["Monthly Demand", "Weekly Demand", "Current Stock", "Target Stock"]
        values     = [raw_prediction, expected_demand, current_stock, target_stock]
        colors     = ["#6366f1", "#8b5cf6", chart_color, "#94a3b8"]

        fig.add_trace(go.Bar(
            x=categories, y=values,
            marker_color=colors,
            text=[f"{v:,.1f}" for v in values],
            textposition="outside",
            textfont=dict(size=11, family="JetBrains Mono"),
        ))

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(
                title="Units",
                gridcolor="#f1f5f9",
                gridwidth=1,
            ),
            xaxis=dict(tickfont=dict(size=11)),
            height=320,
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with rec_col:
        st.markdown('<div class="section-label">Recommendations</div>', unsafe_allow_html=True)

        if coverage_ratio < 0.9:
            reorder_qty = target_stock - current_stock
            st.error(f"**Reorder immediately**\n\nShortfall: **{reorder_qty:,.1f} units**")
            st.markdown("- Trigger emergency restock\n- Notify procurement team\n- Check supplier lead time")
        elif coverage_ratio <= 1.1:
            st.success("**Stock levels are healthy**\n\nNo action required.")
            st.markdown("- Continue current replenishment\n- Monitor weekly demand trend\n- Review in 7 days")
        else:
            excess = current_stock - target_stock
            st.warning(f"**Reduce incoming orders**\n\nExcess: **{excess:,.1f} units**")
            st.markdown("- Pause next restock cycle\n- Consider promotional markdown\n- Review min/max thresholds")

        st.markdown("---")
        st.metric("Coverage Ratio", f"{coverage_ratio:.2f}",
                  delta=f"{(coverage_ratio - 1) * 100:+.1f}% vs ideal")

    # History table
    if show_history and st.session_state.history:
        st.markdown("---")
        st.markdown('<div class="section-label">Prediction History</div>', unsafe_allow_html=True)
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

else:
    st.error("Prediction unavailable. Check model file or slider inputs.")

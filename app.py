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
# LOAD MODEL
# =========================
model = joblib.load("best_model.pkl")
columns = joblib.load("columns.pkl")

# =========================
# TITLE
# =========================
st.title("📦 Retail Demand + Inventory Intelligence System")
st.markdown("AI-powered demand forecasting with real inventory decision engine")

st.divider()

# =========================
# INPUTS
# =========================
st.subheader("🧾 Product Inputs")

item_mrp = st.slider("Item MRP", 10.0, 300.0, 150.0)
item_visibility = st.slider("Item Visibility", 0.0, 0.3, 0.05)
item_weight = st.slider("Item Weight", 1.0, 30.0, 10.0)

st.divider()

# =========================
# PREDICTION ENGINE
# =========================
if st.button("🚀 Predict & Analyze Inventory"):

    # -------------------------
    # INPUT PREP
    # -------------------------
    input_df = pd.DataFrame([{
        "Item_MRP": item_mrp,
        "Item_Visibility": item_visibility,
        "Item_Weight": item_weight
    }])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # -------------------------
    # MODEL PREDICTION
    # -------------------------
    prediction = model.predict(input_encoded)[0]

    revenue = prediction * item_mrp

    # uncertainty band
    demand_std = prediction * 0.20
    lower = prediction - demand_std
    upper = prediction + demand_std

    # =========================
    # INVENTORY SIMULATION
    # =========================

    lead_time_days = 7

    # realistic stock simulation
    current_stock = np.random.randint(
        max(50, int(prediction * 0.3)),
        int(prediction * 1.8)
    )

    # demand during lead time
    expected_demand_lt = (prediction / 30) * lead_time_days

    # safety buffer
    safety_stock = 1.65 * demand_std

    # coverage ratio (CORE METRIC)
    coverage_ratio = current_stock / expected_demand_lt

    # reorder point
    reorder_point = expected_demand_lt + safety_stock

    # =========================
    # FINAL DECISION ENGINE (FIXED)
    # =========================

    if coverage_ratio < 1:
        status = "🔴 STOCKOUT RISK (NOT OPTIMAL)"
        risk = "HIGH"
        color = "#EF4444"

    elif coverage_ratio > 2.5:
        status = "🟠 OVERSTOCK RISK (NOT OPTIMAL)"
        risk = "MEDIUM"
        color = "#F59E0B"

    else:
        status = "🟢 OPTIMAL STOCK LEVEL"
        risk = "LOW"
        color = "#10B981"

    # =========================
    # KPI DASHBOARD
    # =========================
    st.subheader("📊 Forecast Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📦 Predicted Demand", f"{prediction:.0f}")
    c2.metric("💰 Revenue", f"₹ {revenue:,.0f}")
    c3.metric("📦 Current Stock", f"{current_stock}")
    c4.metric("⚠️ Risk Level", risk)

    st.divider()

    # =========================
    # INVENTORY METRICS
    # =========================
    st.subheader("📦 Inventory Intelligence Engine")

    i1, i2, i3 = st.columns(3)

    i1.metric("📍 Coverage Ratio", f"{coverage_ratio:.2f}")
    i2.metric("📉 Expected Demand (7D)", f"{expected_demand_lt:.0f}")
    i3.metric("📦 Reorder Point", f"{reorder_point:.0f}")

    st.markdown(
        f"""
        <div style="
            padding:20px;
            border-radius:12px;
            background-color:#111827;
            border-left:6px solid {color};
        ">
            <h3 style="color:white;">Inventory Decision</h3>
            <h2 style="color:{color};">{status}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # =========================
    # VISUALIZATION
    # =========================
    st.subheader("📈 Feature Impact View")

    fig = px.bar(
        x=["MRP", "Visibility", "Weight"],
        y=[item_mrp, item_visibility * 1000, item_weight],
        title="Input Feature Contribution"
    )

    st.plotly_chart(fig, use_container_width=True)

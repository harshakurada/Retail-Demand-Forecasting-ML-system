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
# UI
# =========================
st.title("📦 Retail Demand + Inventory Intelligence System")

st.divider()

st.subheader("🧾 Input Features")

item_mrp = st.slider("Item MRP", 10.0, 300.0, 150.0)
item_visibility = st.slider("Item Visibility", 0.0, 0.3, 0.05)
item_weight = st.slider("Item Weight", 1.0, 30.0, 10.0)

st.divider()

# =========================
# PREDICTION BUTTON
# =========================
if st.button("🚀 Run Forecast & Inventory Check"):

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

    # =========================
    # BUSINESS METRICS
    # =========================
    revenue = prediction * item_mrp

    # uncertainty simulation
    demand_std = prediction * 0.25

    lower = prediction - demand_std
    upper = prediction + demand_std

    # =========================
    # REALISTIC INVENTORY ENGINE
    # =========================

    lead_time_days = 7

    # simulate current stock (real-world constraint)
    current_stock = np.random.randint(
        int(prediction * 0.5),
        int(prediction * 1.8)
    )

    # expected demand during lead time
    daily_demand = prediction / 30
    expected_demand_lt = daily_demand * lead_time_days

    # safety stock (uncertainty buffer)
    safety_stock = 1.65 * np.std([lower, prediction, upper])

    # reorder point
    reorder_point = expected_demand_lt + safety_stock

    # inventory gap
    stock_gap = current_stock - expected_demand_lt

    # coverage ratio
    coverage_ratio = current_stock / expected_demand_lt

    # =========================
    # DECISION ENGINE
    # =========================

    if stock_gap < 0:
        status = "🔴 CRITICAL STOCKOUT RISK"
        risk = "HIGH"
        color = "#EF4444"

    elif coverage_ratio < 1.2:
        status = "🟡 LOW STOCK - REPLENISH SOON"
        risk = "MEDIUM"
        color = "#F59E0B"

    else:
        status = "🟢 STOCK LEVEL OPTIMAL"
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
    c4.metric("⚠️ Risk", risk)

    st.divider()

    # =========================
    # INVENTORY INTELLIGENCE
    # =========================
    st.subheader("📦 Inventory Intelligence Engine")

    i1, i2, i3 = st.columns(3)

    i1.metric("📍 Reorder Point", f"{reorder_point:.0f}")
    i2.metric("📉 Expected Demand (7D)", f"{expected_demand_lt:.0f}")
    i3.metric("📊 Coverage Ratio", f"{coverage_ratio:.2f}")

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

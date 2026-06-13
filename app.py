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

try:
    columns = joblib.load("columns.pkl")
except:
    columns = None

# =========================
# TITLE
# =========================
st.title("📦 Multi-Product Retail Intelligence System")
st.markdown("Production-grade deterministic inventory + demand system")

st.divider()

# =========================
# PRODUCT SELECTION (MULTI PRODUCT)
# =========================
product = st.selectbox(
    "📦 Select Product",
    ["Product A", "Product B", "Product C"]
)

# preset realistic variations (simulating catalog)
product_profiles = {
    "Product A": {"mrp": 120, "vis": 0.10, "wt": 12},
    "Product B": {"mrp": 250, "vis": 0.05, "wt": 20},
    "Product C": {"mrp": 60,  "vis": 0.20, "wt": 8}
}

profile = product_profiles[product]

# =========================
# INPUTS (PRE-FILLED BUT EDITABLE)
# =========================
st.subheader("🧾 Input Features")

item_mrp = st.slider("Item MRP", 10.0, 300.0, float(profile["mrp"]))
item_visibility = st.slider("Item Visibility", 0.0, 0.3, float(profile["vis"]))
item_weight = st.slider("Item Weight", 1.0, 30.0, float(profile["wt"]))

# =========================
# RUN MODEL
# =========================
if st.button("🚀 Predict & Analyze"):

    # -------------------------
    # INPUT PREPARATION
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
    # PREDICTION (DETERMINISTIC)
    # -------------------------
    prediction = model.predict(input_encoded)[0]
    prediction = max(float(prediction), 1)

    revenue = prediction * item_mrp

    # =========================
    # DEMAND ENGINE
    # =========================
    lead_time_days = 7

    expected_demand = (prediction / 30) * lead_time_days

    # deterministic stock (NO RANDOMNESS)
    current_stock = int(expected_demand * 1.05)

    coverage_ratio = current_stock / expected_demand

    # =========================
    # SMOOTH RISK ENGINE (NO HARD THRESHOLDS)
    # =========================
    risk_score = abs(1 - coverage_ratio)

    if risk_score < 0.1:
        status = "🟢 OPTIMAL STOCK"
        color = "#10B981"
        risk = "LOW"

    elif risk_score < 0.25:
        status = "🟡 NEAR OPTIMAL STOCK"
        color = "#FBBF24"
        risk = "MEDIUM"

    else:
        status = "🟠 IMBALANCED STOCK"
        color = "#F97316"
        risk = "HIGH"

    reorder_point = expected_demand * 1.2

    # =========================
    # KPI DASHBOARD
    # =========================
    st.subheader("📊 Business Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📦 Demand", f"{prediction:.0f}")
    c2.metric("💰 Revenue", f"₹ {revenue:,.0f}")
    c3.metric("📦 Stock", f"{current_stock}")
    c4.metric("⚠️ Risk", risk)

    st.divider()

    # =========================
    # INVENTORY METRICS
    # =========================
    st.subheader("📦 Inventory Intelligence")

    i1, i2, i3 = st.columns(3)

    i1.metric("Coverage Ratio", f"{coverage_ratio:.2f}")
    i2.metric("Expected Demand", f"{expected_demand:.0f}")
    i3.metric("Reorder Point", f"{reorder_point:.0f}")

    st.markdown(
        f"""
        <div style="
            padding:20px;
            border-radius:12px;
            background-color:#111827;
            border-left:6px solid {color};
        ">
            <h3 style="color:white;">Inventory Status</h3>
            <h2 style="color:{color};">{status}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # =========================
    # VISUALIZATION
    # =========================
    st.subheader("📈 Feature Impact")

    fig = px.bar(
        x=["MRP", "Visibility", "Weight"],
        y=[item_mrp, item_visibility * 1000, item_weight],
        title="Feature Influence"
    )

    st.plotly_chart(fig, use_container_width=True)

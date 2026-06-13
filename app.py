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
st.title("📦 Retail Demand + Smart Inventory System")
st.markdown("Soft-threshold AI inventory decision engine")

st.divider()

# =========================
# INPUTS
# =========================
st.subheader("🧾 Input Features")

item_mrp = st.slider("Item MRP", 10.0, 300.0, 120.0)
item_visibility = st.slider("Item Visibility", 0.0, 0.3, 0.05)
item_weight = st.slider("Item Weight", 1.0, 30.0, 10.0)

st.divider()

# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict"):

    # -------------------------
    # INPUT PREP
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
    # INVENTORY LOGIC (SMOOTH)
    # =========================

    # demand window
    expected_demand = (prediction / 30) * 7

    # soft stock variation (not strict)
    noise = np.random.uniform(0.85, 1.15)
    current_stock = int(expected_demand * noise)

    # ratio
    coverage_ratio = current_stock / expected_demand

    # =========================
    # SOFT RISK SCORE (KEY FIX)
    # =========================
    # Instead of hard thresholds → smooth scoring
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
        status = "🟠 IMBALANCED STOCK (CHECK)"
        color = "#F97316"
        risk = "HIGH"

    # =========================
    # KPI DASHBOARD
    # =========================
    st.subheader("📊 Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📦 Demand", f"{prediction:.0f}")
    c2.metric("💰 Revenue", f"₹ {revenue:,.0f}")
    c3.metric("📦 Stock", f"{current_stock}")
    c4.metric("⚠️ Risk", risk)

    st.divider()

    # =========================
    # INVENTORY INSIGHTS
    # =========================
    st.subheader("📦 Inventory Insights")

    i1, i2, i3 = st.columns(3)

    i1.metric("Coverage Ratio", f"{coverage_ratio:.2f}")
    i2.metric("Expected Demand", f"{expected_demand:.0f}")
    i3.metric("Risk Score", f"{risk_score:.2f}")

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
        title="Input Feature Influence"
    )

    st.plotly_chart(fig, use_container_width=True)

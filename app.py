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
# SESSION STATE (NO DB NEEDED)
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# TITLE
# =========================
st.title("📦 Retail Demand + Inventory Intelligence System")
st.markdown("Stable ML + Adaptive Inventory Decision Engine (Cloud Safe)")

st.divider()

# =========================
# INPUTS
# =========================
st.subheader("🧾 Input Features")

item_mrp = st.slider("Item MRP", 10.0, 300.0, 150.0)
item_visibility = st.slider("Item Visibility", 0.0, 0.3, 0.05)
item_weight = st.slider("Item Weight", 1.0, 30.0, 10.0)

st.divider()

# =========================
# PREDICTION
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
    # MODEL PREDICTION
    # -------------------------
    prediction = model.predict(input_encoded)[0]
    prediction = max(float(prediction), 1)

    revenue = prediction * item_mrp

    # =========================
    # INVENTORY ENGINE
    # =========================
    lead_time_days = 7

    expected_demand = (prediction / 30) * lead_time_days

    safety_stock = prediction * 0.10

    reorder_point = expected_demand + safety_stock

    current_stock = int(prediction * 1.2)

    coverage_ratio = current_stock / expected_demand

    # =========================
    # ADAPTIVE RISK ENGINE
    # =========================
    risk_score = (1 / coverage_ratio) * 100

    if risk_score > 120:
        status = "🔴 STOCKOUT RISK"
        color = "#EF4444"
        risk = "HIGH"

    elif 80 <= risk_score <= 120:
        status = "🟢 OPTIMAL STOCK LEVEL"
        color = "#10B981"
        risk = "LOW"

    else:
        status = "🟠 OVERSTOCK RISK"
        color = "#F97316"
        risk = "MEDIUM"

    # =========================
    # SAVE HISTORY (SESSION STATE)
    # =========================
    st.session_state.history.append({
        "Item MRP": item_mrp,
        "Prediction": float(prediction),
        "Stock": current_stock,
        "Coverage Ratio": round(coverage_ratio, 2),
        "Risk Score": round(risk_score, 2),
        "Status": status
    })

    # =========================
    # KPI DASHBOARD
    # =========================
    st.subheader("📊 Forecast Results")

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
    i2.metric("Risk Score", f"{risk_score:.1f}")
    i3.metric("Reorder Point", f"{reorder_point:.0f}")

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
    st.subheader("📈 Feature Impact")

    fig = px.bar(
        x=["MRP", "Visibility", "Weight"],
        y=[item_mrp, item_visibility * 1000, item_weight],
        title="Input Feature Contribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# HISTORY TABLE
# =========================
st.divider()
st.subheader("📊 Prediction History")

if len(st.session_state.history) > 0:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("No predictions yet.")

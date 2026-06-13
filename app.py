import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import sqlite3
from datetime import datetime

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
# DB (OPTIONAL HISTORY)
# =========================
conn = sqlite3.connect("inventory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_mrp REAL,
    prediction REAL,
    stock INTEGER,
    risk_score REAL,
    status TEXT,
    timestamp TEXT
)
""")
conn.commit()

# =========================
# TITLE
# =========================
st.title("📦 Retail Demand + Adaptive Inventory Intelligence System")
st.markdown("Next-gen ML + Dynamic Risk Scoring Engine")

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
# RUN MODEL
# =========================
if st.button("🚀 Predict & Analyze"):

    # -------------------------
    # PREPROCESS
    # -------------------------
    input_df = pd.DataFrame([{
        "Item_MRP": item_mrp,
        "Item_Visibility": item_visibility,
        "Item_Weight": item_weight
    }])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # -------------------------
    # PREDICTION
    # -------------------------
    prediction = model.predict(input_encoded)[0]
    prediction = max(prediction, 1)

    revenue = prediction * item_mrp

    # =========================
    # INVENTORY SIMULATION
    # =========================
    lead_time_days = 7

    expected_demand_lt = (prediction / 30) * lead_time_days

    demand_std = prediction * 0.20
    safety_stock = 1.65 * demand_std

    reorder_point = expected_demand_lt + safety_stock

    # deterministic stock (stable system)
    current_stock = int(prediction * 1.2)

    coverage_ratio = current_stock / expected_demand_lt

    # =========================
    # 🔥 ADAPTIVE RISK ENGINE (FIX)
    # =========================
    risk_score = (1 / coverage_ratio) * 100

    if risk_score > 120:
        status = "🔴 CRITICAL STOCKOUT RISK"
        color = "#EF4444"
        risk = "HIGH"

    elif 90 <= risk_score <= 120:
        status = "🟡 LOW STOCK WARNING"
        color = "#F59E0B"
        risk = "MEDIUM"

    elif 60 <= risk_score < 90:
        status = "🟢 OPTIMAL STOCK LEVEL"
        color = "#10B981"
        risk = "LOW"

    else:
        status = "🟠 OVERSTOCK RISK"
        color = "#F97316"
        risk = "MEDIUM"

    # =========================
    # SAVE HISTORY
    # =========================
    cursor.execute("""
        INSERT INTO inventory (item_mrp, prediction, stock, risk_score, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        float(item_mrp),
        float(prediction),
        int(current_stock),
        float(risk_score),
        status,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()

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
    st.subheader("📦 Adaptive Inventory Intelligence")

    i1, i2, i3 = st.columns(3)

    i1.metric("📊 Coverage Ratio", f"{coverage_ratio:.2f}")
    i2.metric("⚠️ Risk Score", f"{risk_score:.1f}")
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
    st.subheader("📈 Feature Impact")

    fig = px.bar(
        x=["MRP", "Visibility", "Weight"],
        y=[item_mrp, item_visibility * 1000, item_weight],
        title="Input Feature Contribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# HISTORY
# =========================
st.divider()
st.subheader("📊 Prediction History")

df = pd.read_sql("SELECT * FROM inventory ORDER BY id DESC", conn)
st.dataframe(df, use_container_width=True)

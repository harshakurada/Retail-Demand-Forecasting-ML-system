import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Retail Demand Forecasting AI System",
    layout="wide",
    page_icon="📦"
)

# -------------------------
# LOAD MODEL + COLUMNS
# -------------------------
model = joblib.load("best_model.pkl")
columns = joblib.load("columns.pkl")

# -------------------------
# KPI CARD FUNCTION (POWERBI STYLE)
# -------------------------
def kpi_card(title, value, subtitle="", color="#4CAF50"):
    st.markdown(f"""
    <div style="
        background-color: #111827;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid {color};
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        margin-bottom: 10px;
    ">
        <h4 style="color: #9CA3AF; margin:0;">{title}</h4>
        <h2 style="color: white; margin:5px 0;">{value}</h2>
        <p style="color: #6B7280; margin:0;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.title("📦 Retail Demand Forecasting AI System")
st.markdown("### Industry-Level ML Dashboard for Sales Prediction")

st.divider()

# -------------------------
# INPUTS
# -------------------------
st.sidebar.header("🧾 Input Features")

item_mrp = st.sidebar.slider("Item MRP", 10.0, 300.0, 150.0)
item_visibility = st.sidebar.slider("Item Visibility", 0.0, 0.3, 0.05)
item_weight = st.sidebar.slider("Item Weight", 1.0, 30.0, 10.0)

# -------------------------
# INPUT DATA
# -------------------------
input_dict = {
    "Item_MRP": item_mrp,
    "Item_Visibility": item_visibility,
    "Item_Weight": item_weight
}

input_df = pd.DataFrame([input_dict])

input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

# -------------------------
# PREDICTION BLOCK
# -------------------------
if st.button("🚀 Predict Sales"):

    prediction = model.predict(input_encoded)[0]

    # -------------------------
    # BUSINESS METRICS
    # -------------------------
    avg_price = item_mrp
    revenue = prediction * avg_price

    lower_bound = prediction * 0.85
    upper_bound = prediction * 1.15

    if prediction < 1000:
        stock = "LOW STOCK ALERT 📉"
    elif prediction < 2500:
        stock = "MODERATE STOCK 📦"
    else:
        stock = "HIGH STOCK REQUIRED 📈"

    # -------------------------
    # KPI DASHBOARD (POWERBI STYLE)
    # -------------------------
    st.subheader("📊 Business Performance Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        kpi_card(
            "📦 Predicted Sales",
            f"{prediction:.2f}",
            "AI Forecast Output",
            "#3B82F6"
        )

    with col2:
        kpi_card(
            "💰 Revenue Impact",
            f"₹ {revenue:,.0f}",
            "Estimated Business Value",
            "#10B981"
        )

    col3, col4 = st.columns(2)

    with col3:
        kpi_card(
            "📉 Min Demand",
            f"{lower_bound:.0f}",
            "Conservative Estimate",
            "#F59E0B"
        )

    with col4:
        kpi_card(
            "📈 Max Demand",
            f"{upper_bound:.0f}",
            "Optimistic Estimate",
            "#EF4444"
        )

    st.divider()

    # -------------------------
    # STOCK RECOMMENDATION
    # -------------------------
    st.subheader("📦 Inventory Recommendation")
    st.success(stock)

    st.divider()

    # -------------------------
    # FEATURE VISUALIZATION
    # -------------------------
    st.subheader("📈 Input Feature Overview")

    fig = px.bar(
        x=["MRP", "Visibility", "Weight"],
        y=[item_mrp, item_visibility * 1000, item_weight],
        title="Feature Impact View (Scaled)"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# BUSINESS INSIGHTS
# -------------------------
st.divider()

st.subheader("📌 Business Insights")

st.markdown("""
- Higher MRP affects revenue significantly  
- Visibility impacts customer purchase behavior  
- Ensemble models perform best for retail forecasting  
- Inventory planning depends on demand prediction  
""")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("🚀 Built with Streamlit | End-to-End ML System | Retail Demand Forecasting")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Retail Demand Forecasting",
    layout="wide",
    page_icon="📦"
)

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("best_model.pkl")

# -------------------------
# HEADER
# -------------------------
st.title("📦 Retail Demand Forecasting Dashboard")
st.markdown("### AI-powered Sales Prediction System for BigMart")

st.divider()

# -------------------------
# SIDEBAR INPUTS
# -------------------------
st.sidebar.header("🧾 Input Features")

item_mrp = st.sidebar.slider("Item MRP", 10.0, 300.0, 150.0)
item_visibility = st.sidebar.slider("Item Visibility", 0.0, 0.3, 0.05)
item_weight = st.sidebar.slider("Item Weight", 1.0, 30.0, 10.0)

outlet_type = st.sidebar.selectbox(
    "Outlet Type",
    ["Supermarket Type1", "Supermarket Type2", "Grocery Store"]
)

item_fat = st.sidebar.selectbox(
    "Item Fat Content",
    ["Low Fat", "Regular"]
)

# -------------------------
# PREDICTION INPUT DF
# -------------------------
input_data = pd.DataFrame([{
    "Item_MRP": item_mrp,
    "Item_Visibility": item_visibility,
    "Item_Weight": item_weight
}])

# -------------------------
# PREDICTION
# -------------------------
prediction = model.predict(input_data)[0]

# -------------------------
# KPI SECTION
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("📊 Predicted Sales", f"{prediction:.2f}")
col2.metric("🏆 Model", "Gradient Boosting")
col3.metric("📉 RMSE", "1035")

st.divider()

# -------------------------
# BUTTON ACTION
# -------------------------
if st.button("🚀 Predict Demand"):

    st.success(f"Predicted Sales: {prediction:.2f} units")

    # -------------------------
    # SIMULATED INSIGHT CHART
    # -------------------------
    st.subheader("📈 Sales Impact Analysis")

    features = ["MRP", "Visibility", "Weight"]
    values = [item_mrp, item_visibility * 1000, item_weight]

    fig = px.bar(
        x=features,
        y=values,
        title="Feature Contribution View (Scaled)"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# BUSINESS INSIGHTS SECTION
# -------------------------
st.divider()

st.subheader("📌 Business Insights")

st.markdown("""
- Higher **Item MRP** generally increases revenue contribution  
- **Visibility** plays a moderate role in demand  
- Outlet type strongly influences sales performance  
- Ensemble models perform best for retail forecasting  
""")

# -------------------------
# SIMULATED SALES TREND
# -------------------------
st.subheader("📊 Sales Trend Simulation")

days = np.arange(1, 31)
sales = np.random.randint(1500, 3000, 30)

trend_df = pd.DataFrame({
    "Day": days,
    "Sales": sales
})

fig2 = px.line(trend_df, x="Day", y="Sales", markers=True)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("🚀 Built with Streamlit | ML Project - Retail Demand Forecasting")

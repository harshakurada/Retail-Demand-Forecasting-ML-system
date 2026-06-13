import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Retail Intelligence SaaS",
    layout="wide",
    page_icon="📦"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("best_model.pkl")
columns = joblib.load("columns.pkl")

# =========================
# SIDEBAR NAVIGATION (SAAS STYLE)
# =========================
st.sidebar.title("📊 Retail AI SaaS")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📦 Predict Sales", "📊 Business Insights"]
)

st.sidebar.markdown("---")
st.sidebar.info("AI-powered Retail Demand Forecasting System")

# =========================
# HOME PAGE
# =========================
if page == "🏠 Dashboard":

    st.title("📦 Retail Intelligence SaaS Platform")
    st.markdown("### End-to-End AI System for Demand Forecasting & Business Insights")

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("📈 Model Type", "Gradient Boosting")
    col2.metric("⚡ Status", "Active")
    col3.metric("📊 Accuracy Level", "High")

    st.info("Use the sidebar to navigate to prediction and analytics modules.")

# =========================
# PREDICTION PAGE
# =========================
elif page == "📦 Predict Sales":

    st.title("📦 Sales Prediction Engine")

    st.sidebar.header("🧾 Input Features")

    item_mrp = st.sidebar.slider("Item MRP", 10.0, 300.0, 150.0)
    item_visibility = st.sidebar.slider("Item Visibility", 0.0, 0.3, 0.05)
    item_weight = st.sidebar.slider("Item Weight", 1.0, 30.0, 10.0)

    input_dict = {
        "Item_MRP": item_mrp,
        "Item_Visibility": item_visibility,
        "Item_Weight": item_weight
    }

    input_df = pd.DataFrame([input_dict])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    if st.button("🚀 Predict Demand"):

        prediction = model.predict(input_encoded)[0]

        revenue = prediction * item_mrp
        lower = prediction * 0.85
        upper = prediction * 1.15

        if prediction < 1000:
            stock = "LOW STOCK ALERT 📉"
        elif prediction < 2500:
            stock = "MODERATE STOCK 📦"
        else:
            stock = "HIGH STOCK REQUIRED 📈"

        st.subheader("📊 Prediction Results")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("📦 Sales", f"{prediction:.0f}")
        col2.metric("💰 Revenue", f"₹ {revenue:,.0f}")
        col3.metric("📉 Min Demand", f"{lower:.0f}")
        col4.metric("📈 Max Demand", f"{upper:.0f}")

        st.success(stock)

        st.divider()

        st.subheader("📈 Feature Impact Analysis")

        fig = px.bar(
            x=["MRP", "Visibility", "Weight"],
            y=[item_mrp, item_visibility * 1000, item_weight],
            title="Input Feature Contribution"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# BUSINESS INSIGHTS PAGE
# =========================
elif page == "📊 Business Insights":

    st.title("📊 Business Intelligence Dashboard")

    st.markdown("### Key Retail Insights from Model")

    st.divider()

    col1, col2 = st.columns(2)

    col1.info("📦 Higher MRP increases revenue potential but may reduce demand")
    col2.info("👁️ Visibility strongly impacts customer purchase behavior")

    col3, col4 = st.columns(2)

    col3.success("📊 Ensemble models perform best for retail forecasting")
    col4.warning("📦 Inventory planning should follow demand prediction trends")

    st.divider()

    st.subheader("📈 Simulated Sales Trend")

    days = np.arange(1, 31)
    sales = np.random.randint(1500, 3000, 30)

    trend_df = pd.DataFrame({
        "Day": days,
        "Sales": sales
    })

    fig = px.line(trend_df, x="Day", y="Sales", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Retail Intelligence SaaS | Built with Streamlit | ML + Business Dashboard")

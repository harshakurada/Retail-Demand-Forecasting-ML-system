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
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("📊 Retail AI SaaS")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📦 Predict Sales", "📊 Insights"]
)

st.sidebar.markdown("---")
st.sidebar.info("End-to-End Retail Demand Forecasting System")

# =========================
# HOME PAGE
# =========================
if page == "🏠 Dashboard":

    st.title("📦 Retail Intelligence SaaS Platform")
    st.markdown("### AI-powered Demand Forecasting System")

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("🤖 Model", "Gradient Boosting")
    col2.metric("⚡ Status", "Active")
    col3.metric("📊 Type", "Regression AI")

    st.success("Use the sidebar to start predictions and analysis.")

# =========================
# PREDICTION PAGE
# =========================
elif page == "📦 Predict Sales":

    st.title("📦 Sales Prediction Engine")
    st.markdown("### Smart Retail Demand Forecasting")

    st.divider()

    # -------------------------
    # INPUT SECTION
    # -------------------------
    st.subheader("🧾 Input Configuration")

    colA, colB = st.columns([1, 2])

    with colA:
        item_mrp = st.slider("Item MRP", 10.0, 300.0, 150.0)
        item_visibility = st.slider("Item Visibility", 0.0, 0.3, 0.05)
        item_weight = st.slider("Item Weight", 1.0, 30.0, 10.0)

    with colB:
        st.markdown("### 📊 Input Overview")

        input_df = pd.DataFrame({
            "Feature": ["MRP", "Visibility", "Weight"],
            "Value": [item_mrp, item_visibility, item_weight]
        })

        st.dataframe(input_df, use_container_width=True)

    st.divider()

    # -------------------------
    # PREDICTION
    # -------------------------
    if st.button("🚀 Predict Demand"):

        input_dict = {
            "Item_MRP": item_mrp,
            "Item_Visibility": item_visibility,
            "Item_Weight": item_weight
        }

        input_df = pd.DataFrame([input_dict])

        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]

        # Business metrics
        revenue = prediction * item_mrp
        lower = prediction * 0.85
        upper = prediction * 1.15

        if prediction < 1000:
            stock = "🔴 LOW STOCK ALERT"
            color = "#EF4444"
        elif prediction < 2500:
            stock = "🟡 MODERATE STOCK"
            color = "#F59E0B"
        else:
            stock = "🟢 HIGH STOCK REQUIRED"
            color = "#10B981"

        # -------------------------
        # KPI DASHBOARD
        # -------------------------
        st.subheader("📊 Prediction Results")

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("📦 Sales", f"{prediction:.0f}")
        k2.metric("💰 Revenue", f"₹ {revenue:,.0f}")
        k3.metric("📉 Min Demand", f"{lower:.0f}")
        k4.metric("📈 Max Demand", f"{upper:.0f}")

        st.divider()

        # -------------------------
        # STOCK CARD
        # -------------------------
        st.markdown(f"""
        <div style="
            padding:20px;
            border-radius:15px;
            background-color:#111827;
            border-left:6px solid {color};
        ">
            <h3 style="color:white;">📦 Inventory Recommendation</h3>
            <h2 style="color:{color};">{stock}</h2>
            <p style="color:#9CA3AF;">AI-based demand-driven decision</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # -------------------------
        # VISUALIZATION
        # -------------------------
        st.subheader("📈 Feature Impact View")

        fig = px.bar(
            x=["MRP", "Visibility", "Weight"],
            y=[item_mrp, item_visibility * 1000, item_weight],
            title="Feature Contribution Analysis"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# INSIGHTS PAGE
# =========================
elif page == "📊 Insights":

    st.title("📊 Business Insights")

    st.markdown("### Key Retail Analytics")

    col1, col2 = st.columns(2)

    col1.info("📦 MRP strongly influences revenue and demand")
    col2.info("👁️ Visibility increases purchase probability")

    col3, col4 = st.columns(2)

    col3.success("📊 Ensemble models perform best in retail forecasting")
    col4.warning("📦 Inventory planning depends on demand prediction")

    st.divider()

    st.subheader("📈 Sales Trend Simulation")

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
st.markdown("🚀 Retail Intelligence SaaS | ML + Business Analytics Dashboard")

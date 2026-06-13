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
st.sidebar.info("AI-powered Retail Demand + Inventory Optimization System")

# =========================
# HOME PAGE
# =========================
if page == "🏠 Dashboard":

    st.title("📦 Retail Intelligence SaaS Platform")
    st.markdown("### AI Demand Forecasting + Inventory Optimization System")

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("🤖 Model", "Gradient Boosting")
    col2.metric("⚡ Status", "Active")
    col3.metric("📊 System", "Production Mode")

    st.success("Use sidebar to generate predictions and inventory insights.")

# =========================
# PREDICTION PAGE
# =========================
elif page == "📦 Predict Sales":

    st.title("📦 Sales Prediction Engine")
    st.markdown("### AI-Based Demand Forecasting System")

    st.divider()

    # -------------------------
    # INPUTS
    # -------------------------
    st.subheader("🧾 Product Inputs")

    colA, colB = st.columns([1, 2])

    with colA:
        item_mrp = st.slider("Item MRP", 10.0, 300.0, 150.0)
        item_visibility = st.slider("Item Visibility", 0.0, 0.3, 0.05)
        item_weight = st.slider("Item Weight", 1.0, 30.0, 10.0)

    with colB:
        st.markdown("### Input Overview")

        st.dataframe(pd.DataFrame({
            "Feature": ["MRP", "Visibility", "Weight"],
            "Value": [item_mrp, item_visibility, item_weight]
        }), use_container_width=True)

    st.divider()

    # -------------------------
    # PREDICTION
    # -------------------------
    if st.button("🚀 Predict Demand"):

        input_df = pd.DataFrame([{
            "Item_MRP": item_mrp,
            "Item_Visibility": item_visibility,
            "Item_Weight": item_weight
        }])

        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]

        # =========================
        # BUSINESS KPIs
        # =========================
        revenue = prediction * item_mrp
        lower = prediction * 0.85
        upper = prediction * 1.15

        # =========================
        # INDUSTRY INVENTORY MODEL
        # =========================

        lead_time_days = 7
        service_level = 1.65  # ~95% confidence

        demand_std = prediction * 0.25
        avg_daily_demand = prediction / 30

        lead_time_demand = avg_daily_demand * lead_time_days
        safety_stock = service_level * demand_std

        reorder_point = lead_time_demand + safety_stock
        recommended_stock = lead_time_demand + (1.5 * safety_stock)

        # =========================
        # STOCK DECISION ENGINE
        # =========================
        if prediction < reorder_point:
            stock_status = "🔴 RESTOCK IMMEDIATELY"
            risk = "HIGH STOCKOUT RISK"
            color = "#EF4444"
        elif prediction < recommended_stock:
            stock_status = "🟡 MONITOR & REPLENISH SOON"
            risk = "MEDIUM RISK"
            color = "#F59E0B"
        else:
            stock_status = "🟢 STOCK LEVEL HEALTHY"
            risk = "LOW RISK"
            color = "#10B981"

        # =========================
        # KPI DASHBOARD
        # =========================
        st.subheader("📊 Prediction Results")

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("📦 Predicted Sales", f"{prediction:.0f}")
        k2.metric("💰 Revenue", f"₹ {revenue:,.0f}")
        k3.metric("📉 Min Demand", f"{lower:.0f}")
        k4.metric("📈 Max Demand", f"{upper:.0f}")

        st.divider()

        # =========================
        # INVENTORY DASHBOARD
        # =========================
        st.subheader("📦 Inventory Intelligence System")

        c1, c2, c3 = st.columns(3)

        c1.metric("📍 Reorder Point", f"{reorder_point:.0f}")
        c2.metric("📦 Recommended Stock", f"{recommended_stock:.0f}")
        c3.metric("⚠️ Risk Level", risk)

        st.markdown(
            f"""
            <div style="
                padding:20px;
                border-radius:12px;
                background-color:#111827;
                border-left:6px solid {color};
            ">
                <h3 style="color:white;">Inventory Decision</h3>
                <h2 style="color:{color};">{stock_status}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # =========================
        # VISUALIZATION
        # =========================
        st.subheader("📈 Feature Impact Analysis")

        fig = px.bar(
            x=["MRP", "Visibility", "Weight"],
            y=[item_mrp, item_visibility * 1000, item_weight],
            title="Feature Contribution View"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# INSIGHTS PAGE
# =========================
elif page == "📊 Insights":

    st.title("📊 Business Insights")

    st.markdown("### Retail Analytics Overview")

    col1, col2 = st.columns(2)

    col1.info("📦 Demand depends heavily on MRP and visibility")
    col2.info("📊 Inventory optimization reduces stockout risk")

    col3, col4 = st.columns(2)

    col3.success("🤖 Ensemble models perform best in retail forecasting")
    col4.warning("📦 Safety stock improves supply chain reliability")

    st.divider()

    st.subheader("📈 Sales Trend Simulation")

    days = np.arange(1, 31)
    sales = np.random.randint(1500, 3000, 30)

    fig = px.line(x=days, y=sales, markers=True)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Retail Intelligence SaaS | ML + Inventory Optimization System")

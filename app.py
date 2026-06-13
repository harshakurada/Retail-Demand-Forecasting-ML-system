import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Retail Demand AI System",
    layout="wide",
    page_icon="📦"
)

# =========================
# LOAD MODEL + COLUMNS
# =========================
model = joblib.load("best_model.pkl")
columns = joblib.load("columns.pkl")

# =========================
# HEADER
# =========================
st.title("📦 Retail Demand Forecasting AI System")
st.markdown("### Industry-Level ML Dashboard for Sales Prediction")

st.divider()

# =========================
# SIDEBAR INPUT MODE
# =========================
st.sidebar.header("⚙️ Mode Selection")

mode = st.sidebar.radio(
    "Choose Input Mode",
    ["Single Prediction", "Batch CSV Prediction"]
)

# =========================
# SINGLE PREDICTION
# =========================
if mode == "Single Prediction":

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

    if st.button("🚀 Predict Sales"):
        

        prediction = model.predict(input_encoded)[0]
        # 💰 REVENUE IMPACT
        avg_price = item_mrp
        revenue = prediction * avg_price
        # 📊 DEMAND RANGE (UNCERTAINTY SIMULATION)
        lower_bound = prediction * 0.85
        upper_bound = prediction * 1.15
        # 📦 STOCK RECOMMENDATION
        if prediction < 1000:
            stock = "LOW STOCK ALERT 📉"
        elif prediction < 2500:
            stock = "MODERATE STOCK 📦"
        else:
            stock = "HIGH STOCK REQUIRED 📈"

        # KPI ROW
        col1, col2, col3 = st.columns(3)

        col1.metric("📊 Predicted Sales", f"{prediction:.2f}")
        col2.metric("🏆 Model", "Gradient Boosting")
        col3.metric("📉 RMSE", "1035")
        col4 = st.columns(1)[0]
        col4.metric("💰 Revenue Impact", f"₹ {revenue:,.0f}")

        st.success(f"Predicted Sales: {prediction:.2f} units")

        st.divider()

        # Feature impact visualization
        st.subheader("📈 Feature Influence View")

        fig = px.bar(
            x=["MRP", "Visibility", "Weight"],
            y=[item_mrp, item_visibility * 1000, item_weight],
            title="Input Feature Contribution (Scaled)"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# BATCH PREDICTION (CSV)
# =========================
elif mode == "Batch CSV Prediction":

    st.subheader("📂 Upload Dataset for Bulk Prediction")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        st.write("📊 Input Data Preview")
        st.dataframe(df.head())

        # preprocess
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

        preds = model.predict(df_encoded)

        df["Predicted_Sales"] = preds

        st.success("Prediction Completed!")

        st.dataframe(df.head())

        # chart
        fig = px.histogram(df, x="Predicted_Sales", nbins=20)
        st.plotly_chart(fig, use_container_width=True)

        # download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )

# =========================
# MODEL INSIGHTS SECTION
# =========================
st.divider()

st.subheader("📌 Business Insights")

st.markdown("""
- Higher MRP increases revenue potential  
- Visibility impacts purchase probability  
- Ensemble models give best accuracy in retail forecasting  
- Batch prediction helps inventory planning  
""")

# =========================
# MODEL COMPARISON (STATIC)
# =========================
st.subheader("🏆 Model Performance Comparison")

models = ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "Gradient Boosting"]
rmse = [1136, 1513, 1082, 1061, 1070, 1035]

fig2 = px.bar(
    x=models,
    y=rmse,
    title="Model RMSE Comparison (Lower is Better)"
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Built with Streamlit | End-to-End ML System | Retail Demand Forecasting")

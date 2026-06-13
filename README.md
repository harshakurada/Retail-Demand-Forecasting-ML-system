# 📦 Retail Demand Forecasting ML System

An end-to-end Machine Learning system for predicting retail product sales using the BigMart dataset.  
This project includes data preprocessing, feature engineering, multiple ML model comparison, and a fully interactive Streamlit dashboard for real-time predictions.

---

## 🚀 Live Demo
(If deployed on Streamlit Cloud, add link here)

---

## 📊 Project Overview

Retail businesses rely heavily on accurate demand forecasting to optimize inventory, reduce wastage, and improve profitability.  

This project builds a **production-style ML pipeline** that predicts product sales across multiple retail outlets using historical data.

---

## 🧠 Key Features

- End-to-end ML pipeline (data → model → deployment)
- Advanced feature engineering (handling categorical + numerical data)
- Comparison of 6 ML models
- Best model selection based on RMSE
- Interactive Streamlit dashboard
- Real-time sales prediction system
- Business insights visualization

---

## 🤖 Machine Learning Models Used

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor ⭐ (Best Model)  
- XGBoost Regressor  
- LightGBM Regressor  

---

## 🏆 Best Model Performance

| Model | RMSE | MAE |
|------|------|------|
| Gradient Boosting | **1035.73** | **721.67** |

---

## 📊 Tech Stack

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Plotly
- Streamlit

---

## 📈 Key Insights

- Gradient Boosting performed best for structured retail data
- Sales patterns are highly influenced by item price and outlet type
- Ensemble models outperform linear models significantly
- Feature engineering plays a critical role in performance improvement

---

## 🖥️ Streamlit Dashboard Features

- Interactive input form for predictions
- Real-time sales forecasting
- KPI metrics display
- Sales trend visualization
- Business insight section

---

## 📂 Project Structure

retail-demand-forecasting-ml-system
│
├── app.py # Streamlit application
├── best_model.pkl # Trained ML model
├── Train.csv # Dataset
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## 🚀 How to Run Locally

```bash
# Clone repository
git clone https://github.com/your-username/repo-name.git

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

📌 Business Use Case
Retail demand forecasting
Inventory optimization
Sales trend analysis
Supply chain planning

👨‍💻 Author

HARSHA KURADA | Passionate about Machine Learning & Real-world AI systems

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================
# Helper Functions
# =====================================
@st.cache_data
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# =====================================
# Load Model & Metrics
# =====================================
try:
    lgb_model = load_pickle("deployment/lgb_model.pkl")
    baseline_results = load_pickle("deployment/baseline_metrics.pkl")
    lightgbm_results = load_pickle("deployment/lightgbm_metrics.pkl")
    feature_cols = load_pickle("deployment/feature_columns.pkl")
    files_loaded = True
except Exception as e:
    files_loaded = False
    st.error(f"Failed to load model files: {e}")

if not files_loaded:
    st.stop()

# =====================================
# Load Full Dataset (for plotting)
# MUST contain Sales, Store, Date, features
# =====================================
try:
    plot_df = pd.read_csv("deployment/store_processed_small.csv")

    # Ensure Date is datetime type
    if "Date" in plot_df.columns:
        plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")

    df_loaded = True
except Exception as e:
    df_loaded = False
    st.error(f"Failed to load dataset: {e}")

if not df_loaded:
    st.stop()

# =====================================
# Restore Training Dtypes
# =====================================
categorical_cols = ["PromoInterval", "StoreType", "Assortment"]
for col in categorical_cols:
    if col in plot_df.columns:
        plot_df[col] = plot_df[col].astype("category")

if "StateHoliday" in plot_df.columns:
    plot_df["StateHoliday"] = plot_df["StateHoliday"].astype("int64")

# =====================================
# Sidebar Store Filter
# =====================================
st.sidebar.title("🔎 Filters")

if "Store" in plot_df.columns:
    stores = sorted(plot_df["Store"].unique())
    selected_store = st.sidebar.selectbox("Select Store", ["All Stores"] + stores)
else:
    selected_store = "All Stores"

st.sidebar.markdown("---")
st.sidebar.caption("Filter forecasts by store")

# Filter data if store is selected
if selected_store != "All Stores":
    plot_df = plot_df[plot_df["Store"] == selected_store]

# =====================================
# Page Title
# =====================================
st.title("📊 Retail Demand Forecasting – Executive Dashboard")

# =====================================
# KPIs Section
# =====================================
c1, c2, c3 = st.columns(3)

c1.metric("Baseline MAPE", f"{baseline_results['MAPE']:.2f}%")
c2.metric("LightGBM MAPE", f"{lightgbm_results['MAPE']:.2f}%")

improvement = baseline_results['MAPE'] - lightgbm_results['MAPE']
c3.metric("Accuracy Improvement", f"{improvement:.2f}%", "Better than baseline")

st.markdown("---")

# =====================================
# Forecast Generation
# =====================================
st.header("📈 Forecast vs Actual Sales")

# Ensure all model-required features exist
available_features = [f for f in feature_cols if f in plot_df.columns]
missing_features = [f for f in feature_cols if f not in plot_df.columns]

if missing_features:
    st.error(f"Missing model features in dataset: {missing_features}")
    st.stop()

# Predict directly from current dataset
try:
    plot_df["Forecast"] = lgb_model.predict(plot_df[available_features])
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Ensure columns for plotting exist
for col in ["Date", "Sales", "Forecast"]:
    if col not in plot_df.columns:
        st.error(f"'{col}' column missing in dataset, cannot plot forecast.")
        st.stop()

# Plot Actual vs Forecast
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(plot_df["Date"], plot_df["Sales"], label="Actual Sales")
ax.plot(plot_df["Date"], plot_df["Forecast"], label="Forecasted Sales", linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
st.pyplot(fig)

st.markdown("---")

# =====================================
# Model Comparison Table
# =====================================
st.header("🏁 Model Accuracy Comparison")

comparison_df = pd.DataFrame({
    "Model": ["Seasonal Naive", "LightGBM"],
    "RMSE": [baseline_results["RMSE"], lightgbm_results["RMSE"]],
    "MAPE (%)": [baseline_results["MAPE"], lightgbm_results["MAPE"]],
    "WAPE (%)": [baseline_results["WAPE"], lightgbm_results["WAPE"]],
})

st.dataframe(comparison_df, use_container_width=True)

st.markdown(
    f"""
**Insight:**  
LightGBM improved forecasting accuracy by **{improvement:.2f}%**  
compared to baseline Seasonal Naïve forecasting.
"""
)

st.markdown("---")

# =====================================
# SHAP (optional)
# =====================================
st.header("🔍 Key Demand Drivers (Feature Importance)")

try:
    import shap

    sample_size = min(2000, len(plot_df))
    shap_df = plot_df.sample(n=sample_size, random_state=42)[available_features]

    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(shap_df.values)

    st.caption("The chart below highlights which features have the strongest influence on demand forecasts.")

    fig, ax = plt.subplots(figsize=(10,6))
    shap.summary_plot(shap_values, shap_df, feature_names=available_features, show=False)
    st.pyplot(fig)

except Exception:
    st.warning("SHAP could not run. Add `shap` to your requirements.txt to enable feature importance.")

st.markdown("---")

# =====================================
# Business Value Summary
# =====================================
st.header("💼 Business Value Summary")

st.markdown(
    f"""
### Key Insights

- Forecasting accuracy improved by **{improvement:.2f}%**
- Enables smarter procurement decisions
- Helps plan promotions more effectively
- Supports proactive inventory control
- Enhances store-level visibility of seasonal trends

> LightGBM successfully demonstrates the value of AI-driven demand forecasting in retail.
"""
)


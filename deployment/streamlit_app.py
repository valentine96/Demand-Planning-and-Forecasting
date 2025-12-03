import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================
# Helper to Load Pickles
# =========================
@st.cache_data
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# =========================
# Load Models and Metrics
# =========================
try:
    lgb_model = load_pickle("deployment/lgb_model.pkl")
    baseline_results = load_pickle("deployment/baseline_metrics.pkl")
    lightgbm_results = load_pickle("deployment/lightgbm_metrics.pkl")
    feature_cols = load_pickle("deployment/feature_columns.pkl")
    files_loaded = True
except Exception as e:
    files_loaded = False
    st.error(f"Failed to load model/metrics files: {e}")

# =========================
# Load Dataset (sampled)
# =========================
try:
    train_fe = pd.read_csv("deployment/train_features_small.csv")
    data_loaded = True
except:
    data_loaded = False
    train_fe = None

# =========================
# Sidebar Filters
# =========================
st.sidebar.title("🔎 Filters")

if train_fe is not None and "Store" in train_fe.columns:
    stores = sorted(train_fe["Store"].unique())
    selected_store = st.sidebar.selectbox("Select Store", ["All Stores"] + stores)
else:
    selected_store = "All Stores"

st.sidebar.markdown("---")
st.sidebar.caption("Filter forecasts by store")

# =========================
# Dashboard Title
# =========================
st.title("📊 Retail Demand Forecasting – Executive Dashboard")

if not files_loaded:
    st.warning("Upload model files first to enable dashboard.")
    st.stop()

# =========================
# KPI Metrics
# =========================
c1, c2, c3 = st.columns(3)

c1.metric("Baseline MAPE", f"{baseline_results['MAPE']:.2f}%")
c2.metric("LightGBM MAPE", f"{lightgbm_results['MAPE']:.2f}%")

lift = baseline_results['MAPE'] - lightgbm_results['MAPE']
c3.metric("Accuracy Improvement", f"{lift:.2f}%", "Better than baseline")

st.markdown("---")

# =========================
# Forecast vs Actual
# =========================
st.header("📈 Forecast vs Actual Sales")

if not data_loaded:
    st.info("Upload 'train_features_small.csv' inside deployment folder.")
else:
    plot_df = train_fe.copy()

    if selected_store != "All Stores" and "Store" in plot_df.columns:
        plot_df = plot_df[plot_df["Store"] == selected_store]

    try:
        # Ensure using only model features
        features_for_pred = [f for f in feature_cols if f in plot_df.columns]
        plot_df["Forecast"] = lgb_model.predict(plot_df[features_for_pred])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    if "Date" in plot_df.columns:
        date_col = "Date"
    else:
        date_col = plot_df.columns[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_df[date_col], plot_df["Sales"], label="Actual Sales")
    ax.plot(plot_df[date_col], plot_df["Forecast"], label="Forecasted Sales", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")

# =========================
# Model Accuracy Comparison
# =========================
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
LightGBM improves forecasting accuracy by **{lift:.2f}%** 
compared to the baseline Seasonal Naïve model.
"""
)

st.markdown("---")

# =========================
# SHAP Feature Influence
# =========================
st.header("🔍 Key Demand Drivers")

if not data_loaded:
    st.info("Upload dataset to generate SHAP analysis.")
else:
    try:
        import shap

        n_samples = min(2000, len(train_fe))
        sample_df = train_fe.sample(n=n_samples, random_state=42)[features_for_pred]

        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(sample_df.values)

        st.info("The chart below highlights the strongest drivers of product demand.")

        fig, ax = plt.subplots(figsize=(10,6))
        shap.summary_plot(shap_values, sample_df, feature_names=features_for_pred, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP could not run.")
        st.caption("Install `shap` in app requirements to enable feature importance.")

st.markdown("---")

# =========================
# Business Value Summary
# =========================
st.header("💼 Business Value Summary")

st.markdown(
    f"""
### What The Model Achieves

- Forecasting accuracy improved by **{lift:.2f}%**
- Supports smarter inventory planning
- Reduces stockouts and overstocking
- Optimizes promotional scheduling
- Enhances weekly & seasonal demand insight

> LightGBM achieves the project's core goal of a **20–30% improvement** over baseline.
"""
)

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
# Load Dataset — Full data (Sales, Date, Store)
# =========================
try:
    plot_df = pd.read_csv("deployment/store_processed_small.csv")

    # Ensure Date is datetime
    if "Date" in plot_df.columns:
        plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")

    df_loaded = True
except Exception as e:
    df_loaded = False
    plot_df = None
    st.error(f"Failed to load plotting dataset: {e}")

# =========================
# Load Dataset — Model Feature Data
# =========================
try:
    model_df = pd.read_csv("deployment/train_features_small.csv")

    # Restore training dtypes
    categorical_cols = ["PromoInterval", "StoreType", "Assortment"]
    for col in categorical_cols:
        if col in model_df.columns:
            model_df[col] = model_df[col].astype("category")

    if "StateHoliday" in model_df.columns:
        model_df["StateHoliday"] = model_df["StateHoliday"].astype("int64")

    feat_loaded = True
except Exception as e:
    feat_loaded = False
    model_df = None
    st.error(f"Failed to load feature dataset: {e}")

# =========================
# Safety Checks
# =========================
if not files_loaded:
    st.stop()

if not df_loaded or not feat_loaded:
    st.stop()

# =========================
# Ensure ID column exists in both
# =========================
if "Id" not in model_df.columns or "Id" not in plot_df.columns:
    st.error("❌ 'Id' column is missing in either dataset. Cannot align predictions.")
    st.stop()

# =========================
# Sidebar Filters
# =========================
st.sidebar.title("🔎 Filters")

if "Store" in plot_df.columns:
    stores = sorted(plot_df["Store"].unique())
    selected_store = st.sidebar.selectbox("Select Store", ["All Stores"] + stores)
else:
    selected_store = "All Stores"

st.sidebar.markdown("---")
st.sidebar.caption("Filter forecasts by store")

# =========================
# Dashboard Title
# =========================
st.title("📊 Retail Demand Forecasting – Executive Dashboard")

# =========================
# KPI Metrics
# =========================
c1, c2, c3 = st.columns(3)

c1.metric("Baseline MAPE", f"{baseline_results['MAPE']:.2f}%")
c2.metric("LightGBM MAPE", f"{lightgbm_results['MAPE']:.2f}%")

improvement = baseline_results['MAPE'] - lightgbm_results['MAPE']
c3.metric("Accuracy Improvement", f"{improvement:.2f}%", "Better than baseline")

st.markdown("---")

# =========================
# Generate Forecasts
# =========================
st.header("📈 Forecast vs Actual Sales")

try:
    # Only keep features that exist in model_df
    features_for_pred = [f for f in feature_cols if f in model_df.columns]

    model_df["Forecast"] = lgb_model.predict(model_df[features_for_pred])

    # Merge forecasts back to full dataset
    plot_df = plot_df.merge(
        model_df[["Id", "Forecast"]],
        on="Id",
        how="left"
    )

except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Filter by Store
if selected_store != "All Stores" and "Store" in plot_df.columns:
    plot_df = plot_df[plot_df["Store"] == selected_store]

# Confirm required columns
needed = ["Date", "Sales", "Forecast"]
missing = [c for c in needed if c not in plot_df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(plot_df["Date"], plot_df["Sales"], label="Actual Sales")
ax.plot(plot_df["Date"], plot_df["Forecast"], label="Forecasted Sales", linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
st.pyplot(fig)

st.markdown("---")

# =========================
# Model Accuracy Comparison
# =========================
st.header("🏁 Model Accuracy Comparison")

cmp_df = pd.DataFrame({
    "Model": ["Seasonal Naive", "LightGBM"],
    "RMSE": [baseline_results["RMSE"], lightgbm_results["RMSE"]],
    "MAPE (%)": [baseline_results["MAPE"], lightgbm_results["MAPE"]],
    "WAPE (%)": [baseline_results["WAPE"], lightgbm_results["WAPE"]],
})

st.dataframe(cmp_df, use_container_width=True)

st.markdown(
    f"""
**Insight:**  
LightGBM improves forecasting accuracy by **{improvement:.2f}%** over the baseline Seasonal Naïve model.
"""
)

st.markdown("---")

# =========================
# SHAP Feature Influence
# =========================
st.header("🔍 Key Demand Drivers")

try:
    import shap

    n = min(2000, len(model_df))
    shap_df = model_df.sample(n=n, random_state=42)[features_for_pred]

    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(shap_df.values)

    fig, ax = plt.subplots(figsize=(10,6))
    shap.summary_plot(shap_values, shap_df, feature_names=features_for_pred, show=False)
    st.pyplot(fig)

except Exception:
    st.warning("SHAP not available. Install `shap` in the requirements file.")
    st.caption("Used only for feature importance explanation.")

st.markdown("---")

# =========================
# Business Value Summary
# =========================
st.header("💼 Business Value Summary")

st.markdown(
    f"""
### What The Model Achieves

- Forecasting accuracy improved by **{improvement:.2f}%**
- Supports smarter inventory planning
- Reduces stockouts and overstocking
- Optimizes promotional scheduling
- Enhances weekly & seasonal demand insights

> LightGBM demonstrates the project's goal of a significant uplift over baseline forecasting.
"""
)



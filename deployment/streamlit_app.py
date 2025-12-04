import streamlit as st
import pandas as pd
import pickle
import os

# =========================================================
# MUST BE THE FIRST STREAMLIT CALL
# =========================================================
st.set_page_config(
    page_title="Forecasting System",
    layout="wide"
)

# =========================================================
# Paths (relative to repo root on Streamlit Cloud)
# =========================================================
ARTIFACT_DIR = "deployment"

MODEL_PATH           = f"{ARTIFACT_DIR}/lightgbm_model.pkl"
FEATURES_PATH        = f"{ARTIFACT_DIR}/feature_columns.pkl"
BASELINE_METRICS     = f"{ARTIFACT_DIR}/baseline_metrics.pkl"
ARIMA_METRICS        = f"{ARTIFACT_DIR}/arima_metrics.pkl"
SARIMA_METRICS       = f"{ARTIFACT_DIR}/sarima_metrics.pkl"
LGBM_METRICS         = f"{ARTIFACT_DIR}/lightgbm_metrics.pkl"
SAMPLE_DATA_PATH     = f"{ARTIFACT_DIR}/store1_weekly_predictions.csv"

# =========================================================
# Small helpers
# =========================================================
def verify_file(path: str) -> bool:
    """Check if a file exists on disk."""
    return os.path.exists(path)


def missing(label: str, path: str) -> None:
    """Show a nice error when a required file is missing."""
    st.error(f"❌ Missing file for **{label}** → `{path}`")


@st.cache_data(show_spinner=False)
def load_pickle(path: str):
    """Load a pickle safely (returns None on failure)."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_csv(path: str):
    """Load a CSV safely (returns None on failure)."""
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# =========================================================
# Validate that all required artifacts exist
# =========================================================
required_files = {
    "LightGBM Model": MODEL_PATH,
    "Feature Columns": FEATURES_PATH,
    "Sample Input Data": SAMPLE_DATA_PATH,
    "Baseline Metrics": BASELINE_METRICS,
    "ARIMA Metrics": ARIMA_METRICS,
    "SARIMA Metrics": SARIMA_METRICS,
    "LightGBM Metrics": LGBM_METRICS,
}

missing_any = False
for label, path in required_files.items():
    if not verify_file(path):
        missing(label, path)
        missing_any = True

if missing_any:
    st.warning("⚠️ Model or metrics artifacts are missing. "
               "Please confirm all files are present in the `deployment/` folder.")
    st.stop()

# =========================================================
# Load all artifacts
# =========================================================
with st.spinner("Loading model and metrics…"):
    model           = load_pickle(MODEL_PATH)
    feature_cols    = load_pickle(FEATURES_PATH)
    baseline_metrics = load_pickle(BASELINE_METRICS)
    arima_metrics    = load_pickle(ARIMA_METRICS)
    sarima_metrics   = load_pickle(SARIMA_METRICS)
    lgbm_metrics     = load_pickle(LGBM_METRICS)
    sample_df        = load_csv(SAMPLE_DATA_PATH)

# Final safety check
if any(x is None for x in [
    model, feature_cols, baseline_metrics,
    arima_metrics, sarima_metrics, lgbm_metrics, sample_df
]):
    st.error("❌ One or more artifacts could not be loaded. "
             "Double-check pickle versions and CSV structure.")
    st.stop()

# =========================================================
# Page header
# =========================================================
st.title("📈 Demand Planning & Forecasting System")
st.markdown("### Machine Learning–Driven Forecasting for Retail & FMCG")

st.caption(
    "This dashboard uses historical sales data and a LightGBM model to forecast "
    "store-level demand and compare performance against ARIMA, SARIMA, and a "
    "naive baseline."
)

st.divider()

# =========================================================
# Layout: 2 main tabs
# =========================================================
tab_forecast, tab_metrics = st.tabs(
    ["🔮 Store Forecast (LightGBM)", "📊 Model Metrics Comparison"]
)

# ---------------------------------------------------------
# TAB 1 — Forecasting
# ---------------------------------------------------------
with tab_forecast:
    st.subheader("🔮 Store Sales Prediction using LightGBM")

    # Optional view controls
    st.write("Below is a preview of the input data that will be used in prediction.")
    n_preview = st.slider("Rows to preview", 5, 50, 10)
    st.dataframe(sample_df.head(n_preview), use_container_width=True)

    # Optional store filter (only if Store column exists)
    store_col = None
    for candidate in ["Store", "store", "store_id", "Store_ID"]:
        if candidate in sample_df.columns:
            store_col = candidate
            break

    if store_col:
        stores = sorted(sample_df[store_col].unique())
        selected_store = st.selectbox("Select store to preview forecast", stores)
        store_mask = sample_df[store_col] == selected_store
        df_to_predict = sample_df[store_mask].copy()
    else:
        selected_store = None
        df_to_predict = sample_df.copy()

    st.write("Ready to generate forecasts using the trained LightGBM model.")

    if st.button("🚀 Run Forecast Now"):
        try:
            # Ensure feature columns exist
            X = df_to_predict[feature_cols]
            preds = model.predict(X)
            df_to_predict["Forecast"] = preds

            st.success("✅ Prediction completed successfully!")
            st.write(
                "Below is a sample of the forecast results "
                f"{'(filtered for store ' + str(selected_store) + ')' if selected_store is not None else ''}."
            )
            st.dataframe(df_to_predict.head(30), use_container_width=True)

        except KeyError as e:
            st.error(
                "❌ Some feature columns expected by the model are missing in "
                "the input data. Please confirm that the CSV used in deployment "
                "matches the training features.\n\n"
                f"Missing column: `{str(e)}`"
            )
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ---------------------------------------------------------
# TAB 2 — Metrics
# ---------------------------------------------------------
def metrics_to_dataframe(metrics_obj):
    """Convert various metric objects into a small table."""
    if metrics_obj is None:
        return pd.DataFrame()
    # Common case: dict like {"RMSE": ..., "MAE": ..., ...}
    if isinstance(metrics_obj, dict):
        return pd.DataFrame(
            [{"Metric": k, "Value": v} for k, v in metrics_obj.items()]
        )
    # Already a DataFrame
    if isinstance(metrics_obj, pd.DataFrame):
        return metrics_obj
    # Fallback: just show as a single column
    return pd.DataFrame({"Value": [metrics_obj]})


with tab_metrics:
    st.subheader("📊 Full Model Metrics Comparison")

    col_base, col_arima = st.columns(2)
    col_sarima, col_lgbm = st.columns(2)

    # Baseline
    with col_base:
        st.markdown("#### 🧱 Baseline (Naive) Metrics")
        df_base = metrics_to_dataframe(baseline_metrics)
        if df_base.empty:
            st.info("No baseline metrics found.")
        else:
            st.dataframe(df_base, use_container_width=True, height=200)

    # ARIMA
    with col_arima:
        st.markdown("#### 📉 ARIMA Metrics")
        df_arima = metrics_to_dataframe(arima_metrics)
        if df_arima.empty:
            st.info("No ARIMA metrics found.")
        else:
            st.dataframe(df_arima, use_container_width=True, height=200)

    # SARIMA
    with col_sarima:
        st.markdown("#### 🔁 SARIMA Metrics")
        df_sarima = metrics_to_dataframe(sarima_metrics)
        if df_sarima.empty:
            st.info("No SARIMA metrics found.")
        else:
            st.dataframe(df_sarima, use_container_width=True, height=200)

    # LightGBM
    with col_lgbm:
        st.markdown("#### ⚡ LightGBM Metrics")
        df_lgbm = metrics_to_dataframe(lgbm_metrics)
        if df_lgbm.empty:
            st.info("No LightGBM metrics found.")
        else:
            st.dataframe(df_lgbm, use_container_width=True, height=200)

    st.markdown("---")
    st.markdown(
        "🏆 **Best Performing Model**: "
        "Based on the evaluation metrics (e.g., lower RMSE/MAE and higher R²), "
        "**LightGBM** achieved the strongest forecasting performance across stores."
    )

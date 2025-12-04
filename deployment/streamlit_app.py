import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ==============================================
# Load Model + Feature Columns + Metrics
# ==============================================
@st.cache_data
def load_model():
    with open("lightgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_features():
    with open("feature_columns.pkl", "rb") as f:
        features = pickle.load(f)
    return features

@st.cache_data
def load_metrics():
    try:
        with open("lightgbm_metrics.pkl", "rb") as f:
            metrics = pickle.load(f)
    except:
        metrics = None
    return metrics


# ==============================================
# Load Sample Data (store processed small)
# This ensures real alignment when predicting
# ==============================================
@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv("store_processed_small.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except:
        return None


# ==============================================
# Helper: Make Forecast on any date/store
# ==============================================
def build_input_row(df, store_id, target_date):
    """
    We extract the latest record before the target_date,
    and reuse its engineered features.
    """
    base = df[df["Store"] == store_id].copy()

    if base.empty:
        return None

    # pick latest record
    latest = base.sort_values("Date").iloc[-1]

    new_row = latest.copy()
    new_row["Date"] = target_date

    # If lag features exist (lag_7, lag_14,...), estimate them
    date_diffs = (target_date - latest["Date"]).days
    for col in df.columns:
        if col.startswith("lag_"):
            try:
                lag_days = int(col.split("_")[1])
                new_row[col] = base.sort_values("Date")["Sales"].iloc[-lag_days] \
                               if len(base) > lag_days else new_row["Sales"]
            except:
                pass

    return new_row.to_frame().T


# ==============================================
# STREAMLIT UI
# ==============================================
st.set_page_config(
    page_title="Demand Forecasting System",
    page_icon="📈",
    layout="centered"
)

st.title("📊 Demand Planning & Forecasting System")
st.markdown("Machine Learning Driven Forecasting for Retail & FMCG")

model = load_model()
features = load_features()
metrics = load_metrics()
sample_df = load_sample_data()


# ==============================================
# Sidebar Inputs
# ==============================================
with st.sidebar:
    st.header("🔧 Forecast Inputs")

    if sample_df is not None:
        store_options = sorted(sample_df["Store"].unique().tolist())
    else:
        store_options = []

    store_id = st.selectbox("Select Store ID", store_options)

    target_date = st.date_input(
        "Select Forecast Date",
        datetime(2015, 9, 1),
        min_value=datetime(2015, 7, 1),
        max_value=datetime(2016, 12, 31)
    )


# ==============================================
# Predict Button
# ==============================================
if st.button("Generate Forecast"):
    if sample_df is None:
        st.error("❗ Missing sample data file: store_processed_small.csv")
    else:
        input_row = build_input_row(sample_df, store_id, pd.to_datetime(target_date))

        if input_row is None:
            st.error("No historical data found for this store.")
        else:
            X = input_row[features]
            pred = model.predict(X)[0]

            st.success(f"💡 Forecasted Sales for Store **{store_id}** on **{target_date}**:")
            st.metric(label="Predicted Sales", value=f"{pred:,.0f}")

            # Add small trend series
            history = (
                sample_df[sample_df["Store"] == store_id]
                .sort_values("Date")
                .tail(20)
            )
            history["Forecast"] = np.nan
            history.loc[history.index[-1], "Forecast"] = pred

            st.line_chart(
                history.set_index("Date")[["Sales", "Forecast"]],
                height=350
            )


# ==============================================
# Display Metrics
# ==============================================
st.subheader("📌 Model Performance Summary")

if metrics is not None:
    m1 = round(float(metrics["RMSE"]), 2)
    m2 = round(float(metrics["MAPE (%)"]), 2)
    m3 = round(float(metrics["WAPE (%)"]), 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", m1)
    col2.metric("MAPE %", m2)
    col3.metric("WAPE %", m3)

else:
    st.info("Metrics file not found — add lightgbm_metrics.pkl to deployment directory.")

# ==============================================
# Final Notes
# ==============================================
st.markdown("---")
st.markdown("✔ Built using ARIMA, SARIMA & LightGBM Models")
st.markdown("✔ SHAP Explainability applied to ensure trust in predictions")
st.markdown("📌 Final Deployment Bundle: `deployment/` directory")
st.markdown("👤 Developed by **Valentine (Group Black - Ngao Labs)**")

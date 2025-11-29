import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from datetime import timedelta

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# LOAD MODEL + DATA
# -----------------------------
@st.cache_resource
def load_model_and_features():
    model = pickle.load(open("deployment/lightgbm_model.pkl", "rb"))
    feature_cols = pickle.load(open("deployment/feature_columns.pkl", "rb"))
    return model, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv("deployment/store_processed_small.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

model, feature_cols = load_model_and_features()
data = load_data()

# -----------------------------
# SMALL HELPERS
# -----------------------------
LAG_LIST = [1, 2, 3, 7, 14, 28, 56]

def forecast_store(store_id: int, horizon: int = 14) -> pd.DataFrame:
    """
    Roll-forward multi-step forecast for a single store using the trained LightGBM model.
    Uses last 90 days of sales per store (already in store_processed_small.csv).
    """
    hist = data[data["Store"] == store_id].sort_values("Date").copy()
    last_date = hist["Date"].max()

    forecasts = []

    for step in range(horizon):
        next_date = last_date + timedelta(days=1)

        # Start from last known row as template
        base = hist.iloc[-1].copy()
        base["Date"] = next_date

        # Calendar features (if present)
        if "DayOfWeek" in hist.columns:
            base["DayOfWeek"] = next_date.weekday() + 1
        if "Year" in hist.columns:
            base["Year"] = next_date.year
        if "Month" in hist.columns:
            base["Month"] = next_date.month
        if "Day" in hist.columns:
            base["Day"] = next_date.day
        if "WeekOfYear" in hist.columns:
            base["WeekOfYear"] = int(next_date.isocalendar().week)
        if "Quarter" in hist.columns:
            base["Quarter"] = (next_date.month - 1) // 3 + 1
        if "DayOfYear" in hist.columns:
            base["DayOfYear"] = next_date.timetuple().tm_yday
        if "IsWeekend" in hist.columns:
            base["IsWeekend"] = 1 if next_date.weekday() >= 5 else 0

        # Lags – always based on true/predicted Sales in history
        for lag in LAG_LIST:
            col = f"lag_{lag}"
            if col in hist.columns or col in feature_cols:
                if len(hist) > lag:
                    base[col] = hist["Sales"].iloc[-lag]
                else:
                    base[col] = hist["Sales"].mean()

        # Rolling means
        if "rolling_mean_7" in hist.columns or "rolling_mean_7" in feature_cols:
            base["rolling_mean_7"] = hist["Sales"].tail(7).mean()
        if "rolling_mean_14" in hist.columns or "rolling_mean_14" in feature_cols:
            base["rolling_mean_14"] = hist["Sales"].tail(14).mean()
        if "rolling_mean_30" in hist.columns or "rolling_mean_30" in feature_cols:
            base["rolling_mean_30"] = hist["Sales"].tail(30).mean()

        # Promo / holidays – keep last known situation
        for col in ["Promo", "SchoolHoliday", "StateHoliday"]:
            if col in hist.columns:
                base[col] = hist[col].iloc[-1]

        # Ensure all required feature columns exist
        row_df = pd.DataFrame([base])
        for col in feature_cols:
            if col not in row_df.columns:
                # fallback: use last known value from history if exists, else 0
                if col in hist.columns:
                    row_df[col] = hist[col].iloc[-1]
                else:
                    row_df[col] = 0

        X = row_df[feature_cols]
        y_pred = model.predict(X)[0]

        forecasts.append({"Date": next_date, "Forecast": y_pred})

        # Append prediction to history for next lags
        base["Sales"] = y_pred
        hist = pd.concat([hist, pd.DataFrame([base])], ignore_index=True)
        last_date = next_date

    forecast_df = pd.DataFrame(forecasts)
    return forecast_df


def get_overall_stats(df: pd.DataFrame):
    n_stores = df["Store"].nunique()
    start = df["Date"].min().date()
    end = df["Date"].max().date()
    total_obs = len(df)
    return n_stores, start, end, total_obs

# -----------------------------
# TABS LAYOUT
# -----------------------------
st.markdown("### 🧠 Executive Dashboard – Machine Learning for Retail Demand Planning")

tab_overview, tab_forecast, tab_compare, tab_features = st.tabs(
    ["📊 Overview", "📈 Store Forecast", "⚖️ Model Comparison", "🧬 Feature Importance"]
)

# -----------------------------
# TAB 1 – OVERVIEW
# -----------------------------
with tab_overview:
    n_stores, start_date, end_date, total_obs = get_overall_stats(data)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stores", n_stores)
    col2.metric("Period Start", start_date)
    col3.metric("Period End", end_date)
    col4.metric("Records Used", f"{total_obs:,}")

    st.markdown("#### Recent Portfolio Trend (Average Sales Across All Stores)")
    recent = (
        data.sort_values("Date")
            .groupby("Date")["Sales"]
            .mean()
            .tail(60)
    )
    st.line_chart(recent, height=300)

    st.markdown(
        """
        **Insight:** The portfolio-level trend plot shows how the model’s input data
        behaves over the most recent two months, forming the basis for the forecasts
        generated in later tabs.
        """
    )

# -----------------------------
# TAB 2 – STORE FORECAST
# -----------------------------
with tab_forecast:
    st.markdown("### Store-level Forecast")

    store_id = st.selectbox(
        "Select Store",
        sorted(data["Store"].unique()),
        key="forecast_store_select"
    )

    horizon = st.slider("Forecast horizon (days)", 7, 30, 14, key="forecast_horizon")

    if st.button("Generate Forecast", key="run_forecast_btn"):
        hist = data[data["Store"] == store_id].sort_values("Date").copy()
        hist_tail = hist.tail(60)[["Date", "Sales"]].rename(columns={"Sales": "Actual"})

        forecast_df = forecast_store(store_id, horizon)
        forecast_df = forecast_df.rename(columns={"Forecast": "Predicted"})

        combined = pd.concat(
            [hist_tail.set_index("Date"), forecast_df.set_index("Date")],
            axis=0
        )

        st.markdown(f"#### Store {store_id} – Last 60 days + {horizon} days forecast")
        st.line_chart(combined, height=350)

        st.markdown("#### Forecast Table")
        st.dataframe(forecast_df, use_container_width=True)

        st.info(
            "The solid historical part shows actual sales; the continuation represents "
            "model forecasts based on recent patterns, promotions, and calendar effects."
        )

# -----------------------------
# TAB 3 – MODEL COMPARISON
# -----------------------------
with tab_compare:
    st.markdown("### Baseline vs SARIMA vs LightGBM")

    # Use your previously computed metrics (hard-coded from notebook)
    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Baseline (7-day Lag)",
                "SARIMA (Store 1, weekly)",
                "LightGBM (All stores, daily)",
            ],
            "RMSE": [2614.59, 445.01, 751.22],
            "MAPE (%)": [31.82, 7.14, 8.11],
            "WAPE (%)": [31.18, 8.00, 7.86],
        }
    )

    st.dataframe(comparison_df.style.format({"RMSE": "{:,.2f}", "MAPE (%)": "{:.2f}", "WAPE (%)": "{:.2f}"}), use_container_width=True)

    st.markdown(
        """
        **Reading this table:**

        - The **baseline** seasonal naïve model has very high error (~32% MAPE).
        - **SARIMA** performs extremely well for a single weekly series (Store 1).
        - **LightGBM** offers near-SARIMA accuracy *while scaling to all stores on a daily basis*,
          making it more suitable as the production model.
        """
    )

# -----------------------------
# TAB 4 – FEATURE IMPORTANCE
# -----------------------------
with tab_features:
    st.markdown("### What Drives the Forecasts? (Global Feature Importance)")

    importances = model.feature_importance()
    fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(20)

    st.bar_chart(fi_df.set_index("Feature"))

    st.markdown(
        """
        The chart ranks the top drivers of demand according to the LightGBM model.
        Typically, lag features (recent sales history), **DayOfWeek**, and **Promo**
        dominate, confirming that the model has learned intuitive and explainable
        relationships that align with business understanding in retail.
        """
    )


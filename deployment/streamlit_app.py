import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import lightgbm as lgb

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide"
)

st.markdown("## 🧠 Retail Demand Forecasting – Executive Dashboard")


# =========================================================
# LOAD MODEL + DATA
# =========================================================
@st.cache_resource
def load_model_and_features():
    """Load LightGBM model and feature column list."""
    model = pickle.load(open("deployment/lightgbm_model.pkl", "rb"))
    feature_cols = pickle.load(open("deployment/feature_columns.pkl", "rb"))
    return model, feature_cols


@st.cache_data
def load_data():
    """Load reduced dataset with last 90 days per store."""
    df = pd.read_csv("deployment/store_processed_small.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


model, feature_cols = load_model_and_features()
data = load_data()

LAG_LIST = [1, 2, 3, 7, 14, 28, 56]


# =========================================================
# FORECAST FUNCTION (HARDENED)
# =========================================================
def forecast_store(store_id: int, horizon: int = 14) -> pd.DataFrame:
    """
    Roll-forward multi-step forecast for a selected store.
    Includes safety checks for missing features.
    """
    hist = data[data["Store"] == store_id].sort_values("Date").copy()
    hist = hist.dropna(subset=["Sales"])  # Ensure valid starting point
    last_date = hist["Date"].max()

    forecasts = []

    for step in range(horizon):
        next_date = last_date + timedelta(days=1)

        # Start with last known row
        base = hist.iloc[-1].copy()
        base["Date"] = next_date

        # ----------------------------
        # Calendar features
        # ----------------------------
        if "DayOfWeek" in feature_cols:
            base["DayOfWeek"] = next_date.weekday() + 1
        if "Month" in feature_cols:
            base["Month"] = next_date.month
        if "Day" in feature_cols:
            base["Day"] = next_date.day
        if "WeekOfYear" in feature_cols:
            base["WeekOfYear"] = int(next_date.isocalendar()[1])
        if "Quarter" in feature_cols:
            base["Quarter"] = (next_date.month - 1) // 3 + 1
        if "DayOfYear" in feature_cols:
            base["DayOfYear"] = next_date.timetuple().tm_yday
        if "IsWeekend" in feature_cols:
            base["IsWeekend"] = 1 if next_date.weekday() >= 5 else 0

        # ----------------------------
        # Lag features
        # ----------------------------
        for lag in LAG_LIST:
            col = f"lag_{lag}"
            if col in feature_cols:
                if len(hist) > lag:
                    base[col] = hist["Sales"].iloc[-lag]
                else:
                    base[col] = hist["Sales"].tail(7).mean()

        # ----------------------------
        # Rolling means
        # ----------------------------
        if "rolling_mean_7" in feature_cols:
            base["rolling_mean_7"] = hist["Sales"].tail(7).mean()
        if "rolling_mean_14" in feature_cols:
            base["rolling_mean_14"] = hist["Sales"].tail(14).mean()
        if "rolling_mean_30" in feature_cols:
            base["rolling_mean_30"] = hist["Sales"].tail(30).mean()

        # ----------------------------
        # Static features (Promo / Holidays)
        # ----------------------------
        for col in ["Promo", "StateHoliday", "SchoolHoliday"]:
            if col in feature_cols:
                base[col] = hist[col].iloc[-1] if col in hist.columns else 0

        # ----------------------------
        # Build prediction row
        # ----------------------------
        row_df = pd.DataFrame([base])

        # Ensure ALL model features exist
        for col in feature_cols:
            if col not in row_df.columns:
                row_df[col] = 0  # safe fallback

        X = row_df[feature_cols]
        prediction = model.predict(X)[0]

        forecasts.append({"Date": next_date, "Forecast": prediction})

        # Append back to history
        new_row = base.copy()
        new_row["Sales"] = prediction
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        last_date = next_date

    return pd.DataFrame(forecasts)


# =========================================================
# OVERVIEW TAB
# =========================================================
tab_overview, tab_forecast, tab_compare, tab_features = st.tabs(
    ["📊 Overview", "📈 Store Forecast", "⚖️ Model Comparison", "🧬 Feature Importance"]
)

with tab_overview:
    st.markdown("### Dataset Summary")

    n_stores = data["Store"].nunique()
    start_date = data["Date"].min()
    end_date = data["Date"].max()
    total_obs = len(data)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stores", n_stores)
    col2.metric("Period Start", start_date.strftime("%Y-%m-%d"))
    col3.metric("Period End", end_date.strftime("%Y-%m-%d"))
    col4.metric("Records Used", f"{total_obs:,}")

    st.markdown("#### Recent Trend (Average Sales Across All Stores)")
    recent = (
        data.groupby("Date")["Sales"].mean().sort_index().tail(60)
    )
    st.line_chart(recent)


# =========================================================
# STORE FORECAST TAB
# =========================================================
with tab_forecast:
    st.markdown("### Forecast Future Sales for a Store")

    store_id = st.selectbox("Select Store", sorted(data["Store"].unique()))
    horizon = st.slider("Forecast Horizon (days)", 7, 30, 14)

    if st.button("Run Forecast"):
        hist = data[data["Store"] == store_id].sort_values("Date").tail(60)
        hist = hist.rename(columns={"Sales": "Actual"})

        forecast_df = forecast_store(store_id, horizon)
        forecast_df = forecast_df.rename(columns={"Forecast": "Predicted"})

        combined = pd.concat(
            [
                hist.set_index("Date")[["Actual"]],
                forecast_df.set_index("Date")[["Predicted"]]
            ],
            axis=0
        )

        st.line_chart(combined)

        st.markdown("#### Forecast Table")
        st.dataframe(forecast_df, use_container_width=True)


# =========================================================
# MODEL COMPARISON TAB
# =========================================================
with tab_compare:
    st.markdown("### Model Performance Comparison")

    comparison_df = pd.DataFrame({
        "Model": [
            "Baseline (7-day Lag)",
            "SARIMA (Store 1, Weekly)",
            "LightGBM (All Stores)"
        ],
        "RMSE": [2614.59, 445.01, 751.22],
        "MAPE (%)": [31.82, 7.14, 8.11],
        "WAPE (%)": [31.18, 8.00, 7.86],
    })

    st.dataframe(
        comparison_df.style.format({
            "RMSE": "{:,.2f}",
            "MAPE (%)": "{:.2f}",
            "WAPE (%)": "{:.2f}"
        }),
        use_container_width=True
    )


# =========================================================
# FEATURE IMPORTANCE TAB
# =========================================================
with tab_features:
    st.markdown("### Top Features Driving Forecasts")

    importances = model.feature_importance()
    fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(20)

    st.bar_chart(fi_df.set_index("Feature"))

    st.info(
        "Lag features, DayOfWeek, and Promo are the strongest drivers — consistent "
        "with typical retail demand behavior."
    )

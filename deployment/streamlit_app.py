import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import lightgbm as lgb
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide"
)

st.markdown("## 🧠 Retail Demand Forecasting Dashboard")


# =========================================================
# PATH HELPER
# =========================================================
def get_local_path(filename: str) -> str:
    """Return absolute path to a file that sits beside this script."""
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, filename)


# =========================================================
# LOAD MODEL + DATA
# =========================================================
@st.cache_resource
def load_model_and_features():
    """Load trained LightGBM model and feature column list."""
    model_path = get_local_path("lgb_model.pkl")
    features_path = get_local_path("feature_columns.pkl")

    model = pickle.load(open(model_path, "rb"))
    feature_cols = pickle.load(open(features_path, "rb"))

    try:
        cat_cols = list(model.pandas_categorical)
    except AttributeError:
        cat_cols = []

    return model, feature_cols, cat_cols


@st.cache_data
def load_data():
    """Load reduced training dataset."""
    csv_path = get_local_path("store_processed_small.csv")
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


model, feature_cols, cat_cols = load_model_and_features()
df_all = load_data()

LAG_LIST = [1, 2, 3, 7, 14, 28, 56]


# =========================================================
# FORECAST FUNCTION
# =========================================================
def forecast_store(store_id: int, horizon: int = 14) -> pd.DataFrame:
    """
    Multi-step LightGBM forecast for a store.
    Uses the last known data and rolls forward day by day,
    recreating lag and rolling features.
    """
    hist = df_all[df_all["Store"] == store_id].sort_values("Date").copy()
    hist = hist.dropna(subset=["Sales"])
    last_date = hist["Date"].max()

    forecasts = []

    for step in range(horizon):
        next_date = last_date + timedelta(days=1)
        base = hist.iloc[-1].copy()
        base["Date"] = next_date

        # Calendar features
        cal_map = {
            "DayOfWeek": next_date.weekday() + 1,
            "Year": next_date.year,
            "Month": next_date.month,
            "Day": next_date.day,
            "WeekOfYear": int(next_date.isocalendar()[1]),
            "Quarter": (next_date.month - 1) // 3 + 1,
            "DayOfYear": next_date.timetuple().tm_yday,
            "IsWeekend": 1 if next_date.weekday() >= 5 else 0,
        }

        for col, val in cal_map.items():
            if col in feature_cols:
                base[col] = val

        # Lag features
        for lag in LAG_LIST:
            col = f"lag_{lag}"
            if col in feature_cols:
                if len(hist) > lag:
                    base[col] = hist["Sales"].iloc[-lag]
                else:
                    base[col] = hist["Sales"].mean()

        # Rolling means
        if "rolling_mean_7" in feature_cols:
            base["rolling_mean_7"] = hist["Sales"].tail(7).mean()
        if "rolling_mean_14" in feature_cols:
            base["rolling_mean_14"] = hist["Sales"].tail(14).mean()
        if "rolling_mean_30" in feature_cols:
            base["rolling_mean_30"] = hist["Sales"].tail(30).mean()

        # Promo / holiday metadata
        for col in ["Promo", "StateHoliday", "SchoolHoliday"]:
            if col in feature_cols:
                base[col] = hist[col].iloc[-1] if col in hist.columns else 0

        # Build prediction row
        row_df = pd.DataFrame([base])

        for col in feature_cols:
            if col not in row_df.columns:
                row_df[col] = 0

        X = row_df[feature_cols]

        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")

        X = X.fillna(0)
        y_pred = model.predict(X)[0]
        forecasts.append({"Date": next_date, "Forecast": y_pred})

        new_row = base.copy()
        new_row["Sales"] = y_pred
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        last_date = next_date

    return pd.DataFrame(forecasts)


# =========================================================
# TABS
# =========================================================
tab_overview, tab_forecast, tab_compare, tab_features = st.tabs(
    ["📊 Data Overview", "📈 Store Forecast", "⚖️ Model Comparison", "🧬 Feature Importance"]
)


# =========================================================
# TAB 1 – OVERVIEW
# =========================================================
with tab_overview:
    st.markdown("### Dataset Summary")

    n_stores = df_all["Store"].nunique()
    start_date = df_all["Date"].min()
    end_date = df_all["Date"].max()
    total_obs = len(df_all)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stores", n_stores)
    c2.metric("Period Start", start_date.strftime("%Y-%m-%d"))
    c3.metric("Period End", end_date.strftime("%Y-%m-%d"))
    c4.metric("Records Used", f"{total_obs:,}")

    st.markdown("#### Trend (Average Sales Across All Stores)")
    recent = (
        df_all.groupby("Date")["Sales"]
        .mean()
        .sort_index()
        .tail(60)
    )
    st.line_chart(recent, height=300)

    st.info(
        "This overview shows the historical demand behaviour that LightGBM was trained on. "
        "Average daily sales clearly show weekly cycles, spikes, and low-demand periods."
    )


# =========================================================
# TAB 2 – FORECAST
# =========================================================
with tab_forecast:
    st.markdown("### Store-Level Forecast (Future Predictions)")

    store_id = st.selectbox(
        "Select Store",
        sorted(df_all["Store"].unique())
    )
    horizon = st.slider("Forecast horizon (days)", 7, 30, 14)

    if st.button("Run Forecast"):
        hist = (
            df_all[df_all["Store"] == store_id]
            .sort_values("Date")
            .tail(60)[["Date", "Sales"]]
            .rename(columns={"Sales": "Actual"})
        )

        forecast_df = forecast_store(store_id, horizon)
        forecast_df = forecast_df.rename(columns={"Forecast": "Predicted"})

        combined = pd.concat(
            [
                hist.set_index("Date"),
                forecast_df.set_index("Date")
            ],
            axis=0
        )

        st.markdown(f"#### Store {store_id} – Last 60 Days + {horizon}-Day Forecast")
        st.line_chart(combined, height=350)

        st.markdown("#### Forecast Table")
        st.dataframe(forecast_df, use_container_width=True)

        st.success("Forecast generated successfully!")


# =========================================================
# TAB 3 – MODEL COMPARISON
# =========================================================
with tab_compare:
    st.markdown("### Model Performance Summary")

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Baseline (7-day Lag)",
                "SARIMA (Store 1, Weekly)",
                "LightGBM (All Stores, Daily)",
            ],
            "RMSE": [2614.59, 445.01, 751.22],
            "MAPE (%)": [31.82, 7.14, 8.11],
            "WAPE (%)": [31.18, 8.00, 7.86],
            "Notes": [
                "Simple moving lag average – weak benchmark",
                "High accuracy, but limited to one store / weekly data",
                "Scalable daily forecasting across all stores"
            ],
        }
    )

    st.dataframe(
        comparison_df.style.format(
            {"RMSE": "{:,.2f}", "MAPE (%)": "{:.2f}", "WAPE (%)": "{:.2f}"}
        ),
        use_container_width=True,
    )

    st.info(
        "**LightGBM** balances accuracy and scalability. While SARIMA performs slightly "
        "better on a single weekly series, LightGBM does so across the entire store network "
        "on a daily basis with automated feature engineering—making it the strongest "
        "deployment option."
    )


# =========================================================
# TAB 4 – FEATURE IMPORTANCE
# =========================================================
with tab_features:
    st.markdown("### Feature Importance (Top 20)")

    importances = model.feature_importance()
    fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(20)

    st.bar_chart(fi_df.set_index("Feature"))

    st.info(
        "Lag features, DayOfWeek, and Promo have the highest importance. "
        "This confirms that the model has learned business-aligned patterns—"
        "customer volume, promotional periods, and temporal cycles."
    )


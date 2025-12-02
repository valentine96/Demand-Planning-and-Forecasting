import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Retail Demand Forecasting Dashboard",
    page_icon="🛒",
    layout="wide"
)

st.markdown("## 🧠 Retail Demand Forecasting")


# =========================================================
# LOAD MODEL + FEATURES
# =========================================================
@st.cache_resource
def load_model_and_features():
    """Load LightGBM model and feature column list."""
    model = pickle.load(open("lgb_model.pkl", "rb"))
    feature_cols = pickle.load(open("feature_columns.pkl", "rb"))

    try:
        cat_cols = list(model.pandas_categorical)
    except AttributeError:
        cat_cols = []

    return model, feature_cols, cat_cols


@st.cache_data
def load_data():
    """Load historical data used to generate lag/rolling features."""
    df = pd.read_csv("store_processed_small.csv")
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
    Roll-forward multi-step forecast for a selected store.

    Uses the final historical period to generate future days,
    including rebuilding lag & rolling features at each step.
    """

    hist = df_all[df_all["Store"] == store_id].sort_values("Date").copy()
    hist = hist.dropna(subset=["Sales"])
    last_date = hist["Date"].max()

    forecasts = []

    for step in range(horizon):
        next_date = last_date + timedelta(days=1)

        base = hist.iloc[-1].copy()
        base["Date"] = next_date

        # -------- Calendar features --------
        if "DayOfWeek" in feature_cols:
            base["DayOfWeek"] = next_date.weekday() + 1
        if "Year" in feature_cols:
            base["Year"] = next_date.year
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

        # -------- Lag features --------
        for lag in LAG_LIST:
            col = f"lag_{lag}"
            if col in feature_cols:
                if len(hist) > lag:
                    base[col] = hist["Sales"].iloc[-lag]
                else:
                    base[col] = hist["Sales"].mean()

        # -------- Rolling means --------
        if "rolling_mean_7" in feature_cols:
            base["rolling_mean_7"] = hist["Sales"].tail(7).mean()
        if "rolling_mean_14" in feature_cols:
            base["rolling_mean_14"] = hist["Sales"].tail(14).mean()
        if "rolling_mean_30" in feature_cols:
            base["rolling_mean_30"] = hist["Sales"].tail(30).mean()

        # -------- Promo / holidays --------
        for col in ["Promo", "StateHoliday", "SchoolHoliday"]:
            if col in feature_cols:
                base[col] = hist[col].iloc[-1] if col in hist.columns else 0

        # Create single-row dataframe for prediction
        row_df = pd.DataFrame([base])

        # Ensure all training features exist
        for c in feature_cols:
            if c not in row_df.columns:
                row_df[c] = 0

        # Select & order columns
        X = row_df[feature_cols]

        # Cast categorical columns
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")

        X = X.fillna(0)

        # Predict
        y_pred = model.predict(X)[0]
        forecasts.append({"Date": next_date, "Predicted": y_pred})

        # Append prediction back into history for next lag calculations
        new_row = base.copy()
        new_row["Sales"] = y_pred
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        last_date = next_date

    return pd.DataFrame(forecasts)


# =========================================================
# TABS
# =========================================================
tab_overview, tab_forecast, tab_compare, tab_features = st.tabs(
    ["📊 Overview", "📈 Store Forecast", "⚖️ Model Comparison", "🧬 Feature Importance"]
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
    c2.metric("Start Date", start_date.strftime("%Y-%m-%d"))
    c3.metric("End Date", end_date.strftime("%Y-%m-%d"))
    c4.metric("Total Records", f"{total_obs:,}")

    recent = (
        df_all.groupby("Date")["Sales"]
        .mean()
        .sort_index()
        .tail(60)
    )
    st.markdown("#### Recent Trend (Average Sales Across All Stores)")
    st.line_chart(recent, height=310)

    st.info(
        "This view summarises the historical training data used by LightGBM. "
        "It highlights recent demand patterns before reviewing forward forecasts."
    )


# =========================================================
# TAB 2 – STORE FORECAST
# =========================================================
with tab_forecast:
    st.markdown("### Store-level Forecast")

    store_id = st.selectbox(
        "Select Store",
        sorted(df_all["Store"].unique())
    )
    horizon = st.slider("Forecast horizon (days)", 7, 30, 14)

    if st.button("Generate Forecast"):
        hist = (
            df_all[df_all["Store"] == store_id]
            .sort_values("Date")
            .tail(60)[["Date", "Sales"]]
            .rename(columns={"Sales": "Actual"})
        )

        future = forecast_store(store_id, horizon)
        combined = pd.concat(
            [
                hist.set_index("Date"),
                future.set_index("Date")
            ],
            axis=0
        )

        st.markdown(
            f"#### Store {store_id}: Last 60 days + {horizon}-day Forecast"
        )
        st.line_chart(combined, height=350)

        st.markdown("#### Forecast Table")
        st.dataframe(future, use_container_width=True)

        st.success("Forecast generated successfully!")


# =========================================================
# TAB 3 – MODEL COMPARISON
# =========================================================
with tab_compare:
    st.markdown("### Model Accuracy Comparison")

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
        }
    )

    st.dataframe(
        comparison_df.style.format(
            {"RMSE": "{:,.2f}", "MAPE (%)": "{:.2f}", "WAPE (%)": "{:.2f}"}
        ),
        use_container_width=True,
    )

    st.info(
        "- **Baseline model** shows very high error.\n"
        "- **SARIMA** performs excellently but only on a single weekly store.\n"
        "- **LightGBM** achieves near-SARIMA accuracy **while scaling to all stores daily**, making it ideal for production deployment."
    )


# =========================================================
# TAB 4 – FEATURE IMPORTANCE
# =========================================================
with tab_features:
    st.markdown("### Top Features Driving the LightGBM Model")

    importances = model.feature_importance()
    fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(20)

    st.bar_chart(fi_df.set_index("Feature"))

    st.info(
        "**Lag features**, day-of-week patterns and **Promo** activity "
        "dominate, confirming the model has learned realistic retail demand behaviour."
    )



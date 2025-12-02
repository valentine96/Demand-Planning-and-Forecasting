import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide"
)

st.markdown("## 🧠 Retail Demand Forecasting App")


# =========================================================
# LOAD MODEL + DATA
# =========================================================
@st.cache_resource
def load_model_and_features():
    """
    Load trained LightGBM model and feature column list.
    Model & columns are stored in /models for clean deployment.
    """
    model = joblib.load("models/lgb_model.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")

    # Extract categorical columns remembered by LightGBM
    try:
        cat_cols = list(model.pandas_categorical)
    except AttributeError:
        cat_cols = []

    return model, feature_cols, cat_cols


@st.cache_data
def load_data():
    """
    Load reduced dataset containing the last ~90 days per store.
    Stored inside /data for lightweight deployment.
    """
    df = pd.read_csv("data/store_processed_small.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


model, feature_cols, cat_cols = load_model_and_features()
df_all = load_data()

# Lags engineered during training
LAG_LIST = [1, 2, 3, 7, 14, 28, 56]


# =========================================================
# FORECAST FUNCTION
# =========================================================
def forecast_store(store_id: int, horizon: int = 14) -> pd.DataFrame:
    """
    Roll-forward multi-step daily forecast for a selected store.
    Uses last known actuals, regenerates lag & rolling features,
    and predicts next N days using LightGBM.

    Ensures alignment with the trained model:
    - Same features
    - Same column order
    - Same categorical dtypes
    """
    hist = df_all[df_all["Store"] == store_id].sort_values("Date").copy()
    hist = hist.dropna(subset=["Sales"])
    last_date = hist["Date"].max()

    forecasts = []

    for _ in range(horizon):
        next_date = last_date + timedelta(days=1)

        # Start with the last known row
        base = hist.iloc[-1].copy()
        base["Date"] = next_date

        # ------- Calendar features -------
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

        # ------- Lag Features -------
        for lag in LAG_LIST:
            col = f"lag_{lag}"
            if col in feature_cols:
                base[col] = (
                    hist["Sales"].iloc[-lag]
                    if len(hist) > lag else hist["Sales"].mean()
                )

        # ------- Rolling Means -------
        if "rolling_mean_7" in feature_cols:
            base["rolling_mean_7"] = hist["Sales"].tail(7).mean()
        if "rolling_mean_14" in feature_cols:
            base["rolling_mean_14"] = hist["Sales"].tail(14).mean()
        if "rolling_mean_30" in feature_cols:
            base["rolling_mean_30"] = hist["Sales"].tail(30).mean()

        # ------- Promo / Holiday Features -------
        for col in ["Promo", "StateHoliday", "SchoolHoliday"]:
            if col in feature_cols:
                base[col] = hist[col].iloc[-1] if col in hist.columns else 0

        row_df = pd.DataFrame([base])

        # Guarantee full feature alignment
        for col in feature_cols:
            if col not in row_df.columns:
                row_df[col] = 0

        X = row_df[feature_cols]  # preserve order

        # Cast categories
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")

        X = X.fillna(0)

        y_pred = model.predict(X)[0]
        forecasts.append({"Date": next_date, "Forecast": y_pred})

        # Add prediction back into history
        new_row = base.copy()
        new_row["Sales"] = y_pred
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        last_date = next_date

    return pd.DataFrame(forecasts)


# =========================================================
# STREAMLIT TABS
# =========================================================
tab_overview, tab_forecast, tab_compare, tab_features = st.tabs([
    "📊 Overview",
    "📈 Store Forecast",
    "⚖️ Model Comparison",
    "🧬 Feature Importance"
])


# =========================================================
# TAB: OVERVIEW
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

    st.markdown("#### Recent Trend (Average Sales Across All Stores)")
    recent = (
        df_all.groupby("Date")["Sales"]
        .mean()
        .sort_index()
        .tail(60)  # most recent 60 days
    )
    st.line_chart(recent, height=300)

    st.info(
        "This view summarises the dataset used to train the LightGBM model. "
        "The recent trend helps contextualise behaviour before forecasting."
    )


# =========================================================
# TAB: STORE FORECAST
# =========================================================
with tab_forecast:
    st.markdown("### Store-level Forecast")

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
            [hist.set_index("Date"), forecast_df.set_index("Date")],
            axis=0
        )

        st.markdown(f"#### Store {store_id}: Recent Actuals + {horizon}-day Forecast")
        st.line_chart(combined, height=350)

        st.markdown("#### Forecast Table")
        st.dataframe(forecast_df, use_container_width=True)

        st.success(
            "Forecast generated successfully. Historical values appear as solid lines, "
            "and future LightGBM predictions extend the demand curve."
        )


# =========================================================
# TAB: MODEL COMPARISON
# =========================================================
with tab_compare:
    st.markdown("### Performance Benchmarking")

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Baseline (7-day Lag)",
                "ARIMA (Store 1)",
                "SARIMA (Store 1, Weekly Seasonality)",
                "LightGBM (All Stores, Daily)"
            ],
            "RMSE": [2674.0, 1155.0, 445.0, 513.0],
            "MAPE (%)": [31.82, 13.20, 7.14, 5.81],
            "WAPE (%)": [31.18, 12.70, 8.00, 5.62],
        }
    )

    st.dataframe(
        comparison_df.style.format(
            {"RMSE": "{:,.2f}", "MAPE (%)": "{:.2f}", "WAPE (%)": "{:.2f}"}
        ),
        use_container_width=True,
    )

    st.info(
        "SARIMA provides strong seasonal accuracy on a single store, but LightGBM "
        "achieves the lowest percentage errors while generalising to all stores "
        "on daily forecasts — making it the best production model."
    )


# =========================================================
# TAB: FEATURE IMPORTANCE
# =========================================================
with tab_features:
    st.markdown("### Key Drivers Learned by the Model")

    importances = model.feature_importance()
    fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(20)

    st.bar_chart(fi_df.set_index("Feature"))

    st.info(
        "Lag features, promo activity, calendar structure and store attributes "
        "dominate the model's learning. SHAP analysis in the research notebook "
        "further confirmed these as the most influential demand drivers."
    )


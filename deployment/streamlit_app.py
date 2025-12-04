import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ==========================================
# Helper: Load Pickle safely
# ==========================================
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return None

# ==========================================
# Load Assets
# ==========================================
@st.cache_data
def load_all():
    data = {}

    # Models
    data["model"] = load_pickle("lightgbm_model.pkl")
    data["features"] = load_pickle("feature_columns.pkl")

    # Metrics
    data["baseline_m"] = load_pickle("baseline_metrics.pkl")
    data["arima_m"] = load_pickle("arima_metrics.pkl")
    data["sarima_m"] = load_pickle("sarima_metrics.pkl")
    data["lgb_m"] = load_pickle("lightgbm_metrics.pkl")

    # Predictions
    data["store_pred"] = pd.read_csv("store1_weekly_predictions.csv")
    data["test_pred"] = pd.read_csv("lightgbm_test_forecast.csv")
    data["baseline_pred"] = pd.read_csv("baseline_weekly_predictions.csv")

    return data

data = load_all()

# ==========================================
# Sidebar Navigation
# ==========================================
selected = st.sidebar.selectbox(
    "🔍 Navigate",
    [
        "📈 Weekly Forecast",
        "📊 Metrics Dashboard",
        "🧪 Test Forecast Evaluation",
        "🧠 SHAP Explainability"
    ]
)

# ==========================================
# Page: Weekly Forecast
# ==========================================
if selected == "📈 Weekly Forecast":
    st.title("📈 Weekly Demand Forecast — LightGBM")
    st.write("Predicting store-level weekly demand using trained LightGBM Model.")

    df = data["store_pred"]

    st.subheader("🔮 Forecast Output Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📅 Prediction Chart")
    fig, ax = plt.subplots()
    ax.plot(df["week"], df["prediction"], label="Predicted Demand")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units Sold")
    ax.set_title("Weekly Demand Predictions")
    ax.legend()
    st.pyplot(fig)



# ==========================================
# Page: Metrics Dashboard
# ==========================================
if selected == "📊 Metrics Dashboard":
    st.title("📊 Model Performance Metrics Dashboard")

    baseline = data["baseline_m"]
    arima = data["arima_m"]
    sarima = data["sarima_m"]
    lgb = data["lgb_m"]

    if not baseline:
        st.warning("Metrics could not be loaded. Ensure files exist.")
        st.stop()

    # KPIs
    st.subheader("⚡ RMSE Comparison — Lower is Better")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Baseline RMSE", f"{baseline['rmse']:.2f}")
    col2.metric("ARIMA RMSE", f"{arima['rmse']:.2f}")
    col3.metric("SARIMA RMSE", f"{sarima['rmse']:.2f}")
    col4.metric("LightGBM RMSE", f"{lgb['rmse']:.2f}")

    # Table
    metrics_df = pd.DataFrame({
        "Model": ["Baseline", "ARIMA", "SARIMA", "LightGBM"],
        "MAE": [baseline["mae"], arima["mae"], sarima["mae"], lgb["mae"]],
        "RMSE": [baseline["rmse"], arima["rmse"], sarima["rmse"], lgb["rmse"]],
        "MAPE": [baseline["mape"], arima["mape"], sarima["mape"], lgb["mape"]],
    })

    st.subheader("📊 Detailed Comparison Table")
    st.dataframe(metrics_df, use_container_width=True)

    # Best Model
    best_row = metrics_df.loc[metrics_df["RMSE"].idxmin()]
    st.success(
        f"🏆 Best Model: **{best_row['Model']}** "
        f"with RMSE: **{best_row['RMSE']:.2f}** and MAE: **{best_row['MAE']:.2f}**"
    )

    # RMSE Bar Plot
    st.subheader("📉 RMSE Visualization")
    fig2, ax2 = plt.subplots()
    ax2.bar(metrics_df["Model"], metrics_df["RMSE"])
    ax2.set_title("RMSE Comparison Chart")
    ax2.set_ylabel("RMSE")
    st.pyplot(fig2)



# ==========================================
# Page: Test Forecast Evaluation
# ==========================================
if selected == "🧪 Test Forecast Evaluation":
    st.title("🧪 LightGBM Test Forecast Results")

    df_test = data["test_pred"]
    st.dataframe(df_test.head(), use_container_width=True)

    st.subheader("📉 Forecast vs Actual Plot")
    fig3, ax3 = plt.subplots()
    ax3.plot(df_test["week"], df_test["actual"], label="Actual", color="green")
    ax3.plot(df_test["week"], df_test["forecast"], label="Predicted", linestyle="--")
    ax3.set_xlabel("Week")
    ax3.set_ylabel("Units Sold")
    ax3.set_title("Actual vs Predicted")
    ax3.legend()
    st.pyplot(fig3)



# ==========================================
# Page: SHAP Explainability
# ==========================================
if selected == "🧠 SHAP Explainability":
    st.title("🧠 SHAP Feature Importance (Global)")

    try:
        import shap
        shap.initjs()
    except:
        st.error("SHAP is not installed. Add `shap` in requirements.txt")
        st.stop()

    model = data["model"]
    cols = data["features"]

    if model is None or cols is None:
        st.error("Model or feature columns missing.")
        st.stop()

    st.info("Loading SHAP values... please wait ⏳")

    sample_data = data["store_pred"][cols].head(200)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_data)

    st.subheader("🔥 SHAP Summary Plot")
    fig_shap = shap.summary_plot(shap_values, sample_data, show=False)
    st.pyplot(bbox_inches="tight")

import streamlit as st
import pandas as pd
import joblib
import os

# ================================
# Resolve correct deployment path
# ================================

# Get directory of this app file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# All files live next to this file in same folder
def path(file_name):
    return os.path.join(BASE_DIR, file_name)


# ================================
# Cached Loaders
# ================================
@st.cache_resource
def load_model():
    return joblib.load(path("lightgbm_model.pkl"))

@st.cache_data
def load_feature_columns():
    return joblib.load(path("feature_columns.pkl"))

@st.cache_data
def load_weekly_predictions():
    return pd.read_csv(path("baseline_weekly_predictions.csv"))

@st.cache_data
def load_test_forecast():
    return pd.read_csv(path("lightgbm_test_forecast.csv"))


# ================================
# Load items
# ================================
try:
    model = load_model()
    feature_cols = load_feature_columns()
    weekly_forecast = load_weekly_predictions()
    test_forecast = load_test_forecast()
except Exception as e:
    st.error(f"⚠️ Failed to load model or files.\n\nDetails: {e}")
    st.stop()


# ================================
# UI
# ================================
st.title("📊 Demand Planning & Forecasting System")
st.caption("Machine Learning Driven Forecasting for Retail & FMCG")

st.sidebar.header("🔍 Store Selection")

stores = sorted(weekly_forecast["Store"].unique())
selected_store = st.sidebar.selectbox("Select Store:", stores)


# Filter results for chosen store
store_weekly = weekly_forecast[weekly_forecast["Store"] == selected_store]
store_test = test_forecast[test_forecast["Id"].astype(str).str.startswith(str(selected_store))]


# ================================
# Display Weekly Forecast
# ================================
st.subheader(f"📅 Model Weekly Forecast — Store {selected_store}")

st.line_chart(
    store_weekly.set_index("Date")["LGB_Pred"],
    use_container_width=True
)


# ================================
# Show sample forecasted test values
# ================================
st.subheader(f"🧠 Forecasted Values For Test Horizon — Store {selected_store}")

st.dataframe(
    store_test.head(10),
    use_container_width=True
)

st.success("Model and forecasts loaded successfully!")

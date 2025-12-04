# Deployment Bundle - Demand Forecasting System

This folder contains the deployment-ready artifacts for the Demand Planning and Forecasting Machine Learning System.

## Contents

- `lightgbm_model.pkl` → Trained LightGBM sales forecasting model  
- `feature_columns.pkl` → Saved list of engineered model features  
- `lightgbm_metrics.pkl` → Validation metrics (RMSE, MAPE, WAPE)  
- `lightgbm_val_predictions.csv` → Model predictions on validation set  
- `lightgbm_test_forecast.csv` → Final test set forecast file  
- `store1_weekly_predictions.csv` → Store-level weekly predictions  
- `baseline_weekly_predictions.csv` → Baseline (Naive) forecasts  
- `arima_metrics.pkl` → ARIMA model evaluation metrics  
- `sarima_metrics.pkl` → SARIMA model evaluation metrics  
- `baseline_metrics.pkl` → Metrics for the Naive baseline model  
- `requirements.txt` → Libraries needed to run the dashboard  

---

## Running the Streamlit Forecast Dashboard

1. Install dependencies:
```bash
pip install -r requirements.txt

**🌟 Demand Planning and Forecasting Using Machine Learning**

**Ngao Labs Bootcamp Capstone Project – Group Black**

**🧠 Project Overview**

This project aims to build a machine learning–driven demand forecasting system tailored for the retail and FMCG sector. The goal is to predict store-level daily/weekly product demand using historical sales data, promotions, holidays, and store attributes. Accurate demand forecasting directly improves inventory planning, reduces stockouts, and minimizes overstocking costs, supporting better operational and financial decision-making.

**🎯 Problem Statement**

Retail and FMCG companies struggle to predict demand accurately across different stores and time periods. Inaccurate forecasts lead to:

- Stockouts, causing lost sales and reduced customer satisfaction
- Overstocking, leading to high holding and warehousing costs
- Inefficient distribution planning
- Poor promo planning and missed opportunities
  
This project seeks to develop a robust forecasting system that can improve forecast accuracy by 20–30% compared to baseline methods and contribute to a 10–15% reduction in inventory-related losses.

**🎯 Project Objectives**
- Build a machine learning time-series model to forecast store-level demand
- Engineer meaningful features (time-based, promotions, store attributes)
- Evaluate multiple forecasting methods (ARIMA, Prophet, XGBoost, Hybrid models)
- Compare accuracy against a Seasonal Naïve baseline
- Generate insights on demand drivers (promotions, holidays, seasonality)
- Deploy the final solution via a dashboard or interactive tool
- Improve forecast performance by 20–30%
- Reduce stockout/overstock costs by 10–15%

**📊 Dataset Description (Rossmann Store Sales – Kaggle)**
The dataset contains daily sales data from over 1,000 Rossmann stores, including:
- Sales and customer counts
- Promotional activity (Promo, Promo2, PromoInterval)
- Store characteristics (StoreType, Assortment, CompetitionDistance)
- Calendar features (Dates, School Holidays, State Holidays)
- Trend and seasonality patterns

**🔗 Dataset Link:**
https://www.kaggle.com/c/rossmann-store-sales/data

**🧩 Modeling Approach**
**Baseline Model**
Seasonal Naïve Forecast
Uses previous seasonal values (e.g., same weekday last year) as a time-aware benchmark.
**Advanced Models**
- ARIMA / SARIMA – captures trend and seasonality
- Prophet – models trend, seasonality, and holiday impacts
- XGBoost / LightGBM (with lag features) – captures non-linear patterns influenced by promos, store types, and competition
- Hybrid Model (Prophet + XGBoost) – combines statistical trend modeling with ML-based residual learning

Models will be evaluated with:
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- WAPE (Weighted Absolute Percentage Error)
- Rolling-origin cross-validation

  **👥 Team**
1. Valentine Mbuthu
2. Evalyne Kagendo
   
Supervisor: Elsie

**📌 Project Status**
- ✔ Repository initialized
- ✔ Folder structure completed
- 🔄 Data cleaning & EDA in progress
- ⏳ Modeling and evaluation upcoming

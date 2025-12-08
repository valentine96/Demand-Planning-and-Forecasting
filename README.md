<p align="center">
  <img src="Banner.png" width="100%" />
</p>

# **🌟 Demand Planning and Forecasting Using Machine Learning**
## **Ngao Labs Bootcamp – Capstone Project**

 ## **🧠 Project Overview**

This project develops a machine learning–driven demand forecasting system tailored for the retail and FMCG sector. Using historical sales, promotions, holidays, and store-level attributes, the system predicts daily product demand to support:

- Efficient inventory planning
- Reduction of stockouts
- Lower overstocking and carrying costs
- Better distribution and replenishment decisions
Accurate forecasting ultimately improves operational efficiency, profitability, and customer satisfaction — all critical metrics for retail success.

## **🎯 Problem Statement**

Retail and FMCG companies face significant challenges in anticipating customer demand across numerous stores and time periods. Traditional forecasting approaches often fail to capture:
- Store-specific behaviour
- Promo-driven spikes
- Seasonality patterns
- Holiday effects
- Trend and competition dynamics
  
As a result, organizations experience:

 - Stockouts, causing lost sales and poor customer experiences
 -  Overstocking, leading to unnecessary holding and warehousing costs
 - Inefficient distribution, inflating logistics expenses
 -  Unoptimized promotional planning
This capstone project aims to deliver a robust, scalable forecasting system that improves accuracy by 20–30% compared to baseline models, contributing to an estimated 10–15% reduction in inventory-related losses.

## **🎯 Project Objectives**

✔ Build a machine learning time-series forecasting model for store-level demand

✔ Engineer meaningful features (temporal lags, promos, store attributes, holidays)

✔ Evaluate multiple forecasting approaches, including ARIMA,SARIMA and gradient boosting models

✔ Benchmark performance against a Seasonal Naïve baseline

✔ Identify key demand drivers such as seasonality and promotional activity

✔ Deploy the final model through an interactive dashboard (Streamlit)

✔ Achieve at least a 20–30% improvement in forecast accuracy

✔ Support inventory optimization with an estimated 10–15% cost reduction

### **📊 Dataset Description — Rossmann Store Sales (Kaggle)**

The project uses the Rossmann Store Sales Dataset, a widely studied real-world retail forecasting dataset. It contains:

🛒 Daily sales and customer counts

🎯 Promotional activity (Promo, Promo2, PromoInterval)

🏬 Store characteristics (StoreType, Assortment, CompetitionDistance)

📅 Calendar features (dates, school holidays, state holidays)

📈 Strong trends and seasonality patterns

🔗 Dataset Link

https://www.kaggle.com/c/rossmann-store-sales/data

### **🧩 Modeling Approach**
**1. Baseline Model**

Seasonal Naïve Forecast

Uses the previous season’s value (same weekday last year)

Provides a time-aware benchmark that models must outperform

**2. Advanced Statistical Models**

ARIMA – Captures auto-correlations and short-term trends

SARIMA – Adds seasonal patterns such as weekly/monthly cycles

Prophet – Effective for trend + seasonality + holidays

**3. Machine Learning Models**

LightGBM

Incorporates lag features, rolling windows, promo effects, and store metadata

Captures complex nonlinear relationships

Hybrid Model (Prophet + XGBoost)

Prophet models overall trend + seasonality

Lightgbm learns residuals and nonlinear promo-driven patterns

### **4. Model Evaluation Metrics**

RMSE – Penalizes large errors

MAPE – Measures percentage error

WAPE – Weighted accuracy across stores

SMAPE – Symmetric, stable error comparison

Rolling-origin cross-validation – Time-aware validation

📈 Expected Business Impact

⭐ 20–30% improvement in forecast accuracy

📉 10–15% reduction in inventory losses

🚚 Improved distribution and replenishment planning

🛒 Better promo and campaign planning

💰 Reduced operational costs and enhanced profitability

## **👥 Team**

Valentine Mbuthu

Evalyne Kagendo

Supervisor: Elsie Kiprop

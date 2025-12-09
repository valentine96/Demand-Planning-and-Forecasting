<p align="center">
  <img src="Banner.png" width="100%" />
</p>

# 🌟 Machine Learning for Retail Demand Planning & Forecasting  
### **Ngao Labs Bootcamp – Capstone Project**

---

## 🧭 **Executive Summary**

This project develops a **machine learning–driven demand forecasting system** designed for the retail and FMCG sector. Using historical sales, promotions, store attributes, and calendar patterns, we built a forecasting workflow that significantly improves accuracy over baseline models. The final solution is deployed as a **live interactive Streamlit dashboard**, enabling planners to analyze forecasts at the store and chain level for improved replenishment, promo planning, and inventory alignment.

---

## 🔍 **Problem Statement**

Retail businesses often struggle with:

- ❌ **Stockouts**, leading to lost sales  
- 📉 **Overstocking**, causing storage & holding cost increases  
- 🚚 **Inefficient distribution planning**  
- 🎯 **Uninformed promotional planning**

Accurate forecasting helps reduce these inefficiencies and can cut inventory-related losses by **10–15%**.

This project builds a scalable forecasting engine capable of improving prediction accuracy by **20–30%** over traditional methods.

---

## 🎯 **Project Objectives**

- Develop a **store-level time-series forecasting model**
- Engineer robust **time-based & promo-driven features**
- Compare statistical and ML models:
  - Seasonal Naïve (baseline)
  - ARIMA / SARIMA
  - LightGBM with lag features
- Evaluate models using:
  - RMSE  
  - MAPE   
  - WAPE  
- Deploy a **fully interactive dashboard**
- Enhance forecast reliability for real business applications

---

## 📊 **Dataset Description (Rossmann Store Sales – Kaggle)**

This project uses the **Rossmann Store Sales dataset**, containing over 1,000 stores with:

- 🏬 Store-level daily sales  
- 👥 Customer counts  
- 🎉 Promotion information (Promo, Promo2)  
- 🏷 Store attributes (Type, Assortment)  
- 🗺 Competition data  
- 📅 Holidays & seasonality patterns  

🔗 **Dataset Link:**  
https://www.kaggle.com/c/rossmann-store-sales/data

---

## 🧠 **Modeling Approach**

### **1️⃣ Baseline Model — Seasonal Naïve**
- Uses last year's same-day sales to predict current sales  
- Provides a benchmark for ML & statistical models  

### **2️⃣ Statistical Models**
- **ARIMA / SARIMA** for capturing trend + seasonality

### **3️⃣ Machine Learning Models**
- **LightGBM** (best-performing model)
  - Leverages lag features
  - Captures non-linear relationships
  - Handles promotions & store effects

### **4️⃣ Model Evaluation**
Metrics used:

| Metric | Purpose |
|-------|---------|
| RMSE | Overall error magnitude |
| MAPE | Percentage error |
| WAPE | Weighted error across varying demand levels |

---

## 🏆 **Final Results**

| Model | RMSE | MAPE (%) | WAPE (%) |
|-------|-------|-----------|-----------|
| Baseline | 2614.59 | 31.82 | 31.18 |
| ARIMA | 781.22 | 13.85 | 15.27 |
| SARIMA | 445.01 | 7.14 | 8.00 |
| **LightGBM (Best)** | **514.95** | **5.82** | **5.64** |

👉 **LightGBM outperformed all models and was selected for deployment.**

---

## 🌐 **Live Dashboard**

🎛 **Streamlit App:**  
https://demand-planning-and-forecasting-3jrfrdbgv79yeafhxrwcvl.streamlit.app/

---

## 🖥 **Dashboard Features**

### ✔ Model Metrics Summary  
View RMSE, MAPE, WAPE comparisons for all models.

### ✔ Chain Forecast  
Compare actual vs. forecasted demand across the entire store network.

### ✔ Store-Level Forecast Explorer  
Drill into individual stores and evaluate SMAPE performance.

### ✔ Data Preview & Download  
Inspect datasets and export for further analysis.

---

## 📁 **Project Structure**

Demand-Planning-and-Forecasting/
│
├── .devcontainer/          # Codespaces environment config
├── data/                   # Dataset and raw CSV files
├── deployment/             # Legacy deployment folder (can be removed)
├── notebooks/              # Jupyter/Colab notebooks for EDA & modeling
├── Banner.png              # Project banner for README
├── forecast_results.csv    # Model forecast output (move to /data if needed)
├── requirements.txt        # Python dependencies for Streamlit deployment
├── streamlit_app.py        # Main Streamlit dashboard application
├── README.md               # Project documentation (this file)
└── .gitignore              # Git ignore rules


---

## 🔧 How to Run the Project Locally

Follow the steps below to set up and run the Streamlit dashboard on your machine.


### 1️⃣ Clone the Repository
git clone https://github.com/valentine96/Demand-Planning-and-Forecasting.git
cd Demand-Planning-and-Forecasting

### 2️⃣ Create a Virtual Environment
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Run the Streamlit App
streamlit run streamlit_app.py


| **Role**          | **Name**                                          |
| ----------------- | ------------------------------------------------- |
| Project Lead      | Valentine Mbuthu                                  |
| Peer Collaborator | Evalyne Kagendo                                   |
| Mentor            | Elsie Kiprop                                      |
| Bootcamp          | Ngao Labs – Data Science & AI Foundations Program |


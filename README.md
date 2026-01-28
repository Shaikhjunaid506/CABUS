# 🚖 Cabus — AI-Powered Ride Price Intelligence  
> *Know the best ride before you book.*


---

## 🚀 What is Cabus?

**Cabus** is an **AI-powered ride intelligence platform** that predicts **cab prices, ETA, surge behavior, and traffic impact** for **Uber and Ola rides across Delhi–NCR (Delhi ↔ Noida)**.

Instead of manually checking multiple apps and guessing surge prices, Cabus uses **machine learning trained on 100,000+ realistic ride samples** to recommend the **best possible ride in real time**.

> Think of it as **Google Flights — but for cabs**.

---

## 🎯 Why Cabus?

### ❌ The Problem
- Surge pricing is unpredictable  
- Traffic drastically affects price and ETA  
- Users manually compare Uber & Ola  
- No visibility into future surge windows  

### ✅ The Solution
Cabus learns from historical ride behavior to:
- Predict **realistic prices**
- Estimate **accurate ETA**
- Detect **rush-hour & surge patterns**
- Recommend the **optimal ride** (price × time)

---

## ✨ Key Features

### 🧠 AI & Machine Learning
- Price Prediction (₹ INR)
- ETA Prediction (minutes)
- Surge Probability Modeling
- Traffic Severity Learning
- Time-aware learning:
  - Hour of day
  - Day of week
  - Cyclical encoding (sin/cos)
- Distance-aware pricing using geospatial logic

---

### ⚡ Smart Ride Comparison
- Compare **Uber vs Ola**
- Compare **Mini / Sedan / Prime**
- Automatic **Best Ride Highlighting**
- Balanced scoring: *Price × ETA*

---

### 📊 Real-Time Analytics
- Price trends (last 10 hours)
- Booking surge trends
- Traffic & delay analytics
- Route-level insights (pickup → drop)

---

### 📈 Exploratory Data Analysis (EDA)
- Platform distribution
- Ride category trends
- Distance vs price correlation
- Traffic severity impact
- Peak booking hour detection

---

## 🖥️ Screenshots

> Screenshots are available inside the `/plots` folder.<br>
├── dashboard_main.png<br>
├── price_trends.png<br>
├── traffic_trends.png<br>
├── eda_overview.png<br>
├── eda_distance_price.png<br>
├── eda_traffic_distribution.png<br>



<img width="1910" height="915" alt="Screenshot 2025-11-28 162311" src="https://github.com/user-attachments/assets/75cd712f-1f03-4d50-b892-d9ff567e22eb" />

<img width="1920" height="914" alt="Screenshot 2025-11-28 162358" src="https://github.com/user-attachments/assets/eb7c6ccc-93e2-485d-b546-16e44b1931fd" />

<img width="1918" height="678" alt="Screenshot 2025-11-28 162430" src="https://github.com/user-attachments/assets/29914ba6-7f63-4283-b3f2-4d9dd42d1df4" />

<img width="1920" height="883" alt="Screenshot 2025-11-28 162440" src="https://github.com/user-attachments/assets/6cf48d01-ab7c-44f5-99a1-71b2d7ec611d" />

<img width="1890" height="406" alt="Screenshot 2025-11-28 162452" src="https://github.com/user-attachments/assets/5fd43c66-43c7-4be1-8f9b-43608226b8ff" />


<img width="1920" height="733" alt="Screenshot 2025-11-28 162458" src="https://github.com/user-attachments/assets/dbbf55d2-82b5-4ee8-bc65-7426e17729a0" />







---
## 🏗️ Architecture Overview
Data (100k rides)<br>
↓<br>
Feature Engineering<br>
(distance, time, surge, traffic)<br>
↓<br>
ML Models<br>
(RandomForest → XGBoost)<br>
↓<br>
Prediction Engine<br>
(price + ETA)<br>
↓<br>
Streamlit Dashboard<br>
(real-time insights)

---

## 📁 Project Structure

cabus/<br>
│
├── data/<br>
│ ├── uber_ola_100k_train.csv<br>
│ └── uber_ola_100k_test.csv<br>
│<br>
├── models/<br>
│ ├── model_price_xgb.pkl<br>
│ └── model_eta_xgb.pkl<br>
│
├── data_utils.py # Feature engineering & helpers<br>
├── eda.py # Exploratory Data Analysis<br>
├── train.py # Baseline ML training<br>
├── train_xgb.py # Optimized XGBoost pipeline<br>
├── dashboard.py # Main Streamlit dashboard<br>
├── requirements.txt<br>
└── README.md

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **Pandas, NumPy**
- **Scikit-Learn**
- **XGBoost**
- **Plotly**
- **Streamlit**
- **Joblib**

---

## 🚀 How to Run Locally

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train Optimized Models
``` bash
python train_xgb.py --data data/uber_ola_100k_train.csv
```
### 3️⃣ Launch Dashboard
``` bash
python -m streamlit run dashboard.py
``` 


## 📊 Model Performance (Approx.)
- Task	Model	Performance
- Price Prediction	XGBoost	⭐⭐⭐⭐☆
- ETA Prediction	XGBoost	⭐⭐⭐⭐☆
- Surge Detection	Hybrid (Rules + ML)	⭐⭐⭐⭐☆

✔ Log-scaled targets<br>
✔ Hyperparameter tuning<br>
✔ Early stopping<br>
✔ Edge-case handling (same pickup & drop)

## 💡 Key Insights Discovered

- Peak booking hours: 8–10 AM & 6–8 PM

- Prime rides cost 30–40% more than Mini

- Strong distance–price correlation (R² ≈ 0.85)

- Heavy traffic can increase surge by up to 60%

- Uber & Ola pricing differs by <5% on average

## 🧪 Dataset Highlights

- 100,000+ synthetic but research-backed rides

- Delhi ↔ Noida realistic routing

### Includes:

- Surge multipliers

- Traffic severity

- Ride categories

- Payment methods

- Driver ratings

## 🛣️ Roadmap

🔌 Real-time traffic API integration

🌦️ Weather-aware surge modeling

🗓️ Holiday & event detection

🧠 SHAP-based explainability

☁️ Cloud deployment (AWS / GCP)

📱 Mobile-friendly UI

## 👤 Author

Junaid Hussain<br>
B.Tech (Information Technology) — Delhi NCR<br>
Machine Learning • Data Science • AI Systems



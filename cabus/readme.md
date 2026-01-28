# Delhi-Noida Ride Prediction Project

This project contains code to analyze, train, and serve ML models that predict ride price and ETA
for Uber and Ola rides in the Delhi-Noida region. The dashboard allows a user to pick pickup and drop
locations and a booking time (clock). The app compares predicted prices and ETAs across multiple ride
categories and platforms and highlights the best option (lowest price + ETA tradeoff).

Files:
- data_utils.py: helpers for preprocessing, distance and time feature extraction.
- eda.py: exploratory data analysis, plots, and algorithm comparison (SVM, RandomForest).
- train.py: training pipelines for price & ETA models; saves trained pipelines to models/.
- dashboard.py: Streamlit app for real-time prediction and comparison.
- requirements.txt: Python dependencies.
- data/: expected place for CSV datasets (a copy is included if available).

How to run:
1. Place your CSV dataset (e.g., `uber_ola_100k_train.csv`) in the project folder `data/`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run EDA (optional): `python eda.py --data data/uber_ola_100k_train.csv`
4. Train models: `python train.py --data data/uber_ola_100k_train.csv`
5. Run dashboard: `streamlit run dashboard.py`

Notes:
- The repo expects the CSVs `uber_ola_100k_train.csv` and `uber_ola_100k_test.csv` in the `data/` directory.
- Trained models will be saved to the `models/` directory by `train.py`.


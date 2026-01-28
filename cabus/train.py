# train.py (upgraded)
import argparse, os
import pandas as pd
import numpy as np
from data_utils import add_time_features, add_distance
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from sklearn.metrics import mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/uber_ola_100k_train.csv')
parser.add_argument('--save_dir', type=str, default='models')
args = parser.parse_args()

print("Loading data...")
df = pd.read_csv(args.data)
df = add_time_features(df, 'datetime')
df = add_distance(df)
df = df.dropna(subset=['estimated_distance_km','price_inr','duration_min'])

# Features
cat_cols = ['platform','pickup','drop','ride_category','traffic_severity','payment_method']
num_cols = ['estimated_distance_km','surge_multiplier','hour','dayofweek','hour_sin','hour_cos','dow_sin','dow_cos']

X = df[cat_cols + num_cols]
y_price = df['price_inr']
y_eta = df['duration_min']

# Preprocessor
cat_pipe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
num_pipe = StandardScaler()
preproc = ColumnTransformer([
    ('cat', cat_pipe, cat_cols),
    ('num', num_pipe, num_cols)
], remainder='drop')

# Model pipelines
price_pipeline = Pipeline([
    ('pre', preproc),
    ('hgb', HistGradientBoostingRegressor(random_state=42))
])
eta_pipeline = Pipeline([
    ('pre', preproc),
    ('hgb', HistGradientBoostingRegressor(random_state=42))
])

# Quick split
X_train, X_test, y_price_train, y_price_test = train_test_split(X, y_price, test_size=0.2, random_state=42)
_, _, y_eta_train, y_eta_test = train_test_split(X, y_eta, test_size=0.2, random_state=42)

# Hyperparam ranges for RandomizedSearch (keeps runtime reasonable)
param_dist = {
    'hgb__max_iter': [200, 400],
    'hgb__max_leaf_nodes': [15, 31, 63],
    'hgb__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'hgb__max_depth': [3,5,7,None]
}

print("Tuning & training price model (randomized search, small sample for speed)...")
rs_price = RandomizedSearchCV(price_pipeline, param_distributions=param_dist, n_iter=10,
                              cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42, verbose=1)
rs_price.fit(X_train, y_price_train)
print("Best params (price):", rs_price.best_params_)
price_preds = rs_price.predict(X_test)
print("Price MAE:", mean_absolute_error(y_price_test, price_preds))

print("Tuning & training ETA model...")
rs_eta = RandomizedSearchCV(eta_pipeline, param_distributions=param_dist, n_iter=8,
                              cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42, verbose=1)
rs_eta.fit(X_train, y_eta_train)
eta_preds = rs_eta.predict(X_test)
print("ETA MAE:", mean_absolute_error(y_eta_test, eta_preds))

# Save the best estimators
os.makedirs(args.save_dir, exist_ok=True)
joblib.dump(rs_price.best_estimator_, os.path.join(args.save_dir, 'model_price.pkl'))
joblib.dump(rs_eta.best_estimator_, os.path.join(args.save_dir, 'model_eta.pkl'))
print("Models saved to", args.save_dir)

# Optional: feature importance via permutation (can be added later)

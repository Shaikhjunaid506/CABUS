# train_xgb.py
import os, joblib, argparse, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from data_utils import add_time_features, add_distance
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
import category_encoders as ce
from sklearn.metrics import mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/uber_ola_100k_train.csv')
parser.add_argument('--save_dir', type=str, default='models')
args = parser.parse_args()

print("Loading data...")
df = pd.read_csv(args.data)
df = add_time_features(df, 'datetime')
df = add_distance(df)
df = df.dropna(subset=['distance_km','price_inr','duration_min'])

# features and target
cat_cols = ['platform','ride_category','traffic_severity','payment_method']
high_card_cols = ['pickup','drop']   # target-encode these
num_cols = ['distance_km','surge_multiplier','hour','dayofweek','hour_sin','hour_cos','dow_sin','dow_cos']

# Prepare X,y
X = df[cat_cols + high_card_cols + num_cols].copy()
y_price = df['price_inr'].copy()
y_eta = df['duration_min'].copy()

# Target encoder for pickup/drop
te = ce.TargetEncoder(cols=high_card_cols, smoothing=0.3)

# Column transformer: encode cat_cols (low-card) via OHE, numeric via scaler
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
num_pipe = StandardScaler()

pre = ColumnTransformer(transformers=[
    ('ohe', ohe, cat_cols),
    ('num', num_pipe, num_cols)
], remainder='passthrough')  # passthrough so target-encoded cols remain

# Build pipeline: target-encode, then preproc, then XGB
def build_pipeline():
    xgb = XGBRegressor(tree_method='hist', n_jobs=-1, random_state=42, verbosity=0)
    pipe = Pipeline(steps=[
        ('te', te),
        ('pre', pre),
        ('xgb', xgb)
    ])
    return pipe

# Hyperparam ranges
param_dist = {
    'xgb__n_estimators': [200,400,600],
    'xgb__max_depth': [4,6,8],
    'xgb__learning_rate': [0.01,0.03,0.05,0.08],
    'xgb__colsample_bytree': [0.6,0.8,1.0],
    'xgb__subsample': [0.6,0.8,1.0]
}

# Train price model
print("Training price model (randomized search)...")
price_pipe = build_pipeline()
rs_price = RandomizedSearchCV(price_pipe, param_distributions=param_dist, n_iter=12, cv=3,
                              scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42, verbose=1)
X_train, X_test, y_train, y_test = train_test_split(X, y_price, test_size=0.2, random_state=42)
rs_price.fit(X_train, y_train)
best_price = rs_price.best_estimator_
print("Best price params:", rs_price.best_params_)
preds_price = best_price.predict(X_test)
print("Price MAE:", mean_absolute_error(y_test, preds_price))

# Train ETA model (warm start)
print("Training ETA model (randomized search)...")
eta_pipe = build_pipeline()
rs_eta = RandomizedSearchCV(eta_pipe, param_distributions=param_dist, n_iter=10, cv=3,
                              scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42, verbose=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_eta, test_size=0.2, random_state=42)
rs_eta.fit(X_train2, y_train2)
best_eta = rs_eta.best_estimator_
preds_eta = best_eta.predict(X_test2)
print("ETA MAE:", mean_absolute_error(y_test2, preds_eta))

# Save models
os.makedirs(args.save_dir, exist_ok=True)
joblib.dump(best_price, os.path.join(args.save_dir,'model_price_xgb.pkl'))
joblib.dump(best_eta, os.path.join(args.save_dir,'model_eta_xgb.pkl'))
print("Saved models to", args.save_dir)

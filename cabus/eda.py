import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import add_time_features, add_distance

sns.set(style='whitegrid')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/uber_ola_100k_train.csv')
args = parser.parse_args()

print('Loading', args.data)
df = pd.read_csv(args.data)

# Basic preprocessing
print('Adding time features...')
df = add_time_features(df, 'datetime')
print('Adding estimated distances...')
df = add_distance(df)

# Create output folder
import os
os.makedirs('plots', exist_ok=True)

# Quick stats
print(df[['price_inr','duration_min','estimated_distance_km']].describe())

# 1) Surge frequency by hour
plt.figure(figsize=(10,5))
df['surge'] = df['surge_multiplier']>1.0
hour_group = df.groupby('hour')['surge'].mean()
hour_group.plot(kind='bar')
plt.title('Surge Probability by Hour')
plt.ylabel('Fraction of rides with surge')
plt.tight_layout()
plt.savefig('plots/surge_by_hour.png')

# 2) Average price by trip type
plt.figure(figsize=(8,5))
df.groupby('trip_type')['price_inr'].median().sort_values().plot(kind='barh')
plt.title('Median Price by Trip Type')
plt.tight_layout()
plt.savefig('plots/median_price_by_trip_type.png')

# 3) Popular routes (top origin-destination pairs)
plt.figure(figsize=(10,6))
route_counts = df.groupby(['pickup','drop']).size().reset_index(name='count')
route_counts = route_counts.sort_values('count', ascending=False).head(15)
import seaborn as sns
sns.barplot(y=route_counts.apply(lambda r: f"{r['pickup']} -> {r['drop']}", axis=1), x='count', data=route_counts)
plt.title('Top 15 Pickup->Drop Pairs')
plt.tight_layout()
plt.savefig('plots/top_routes.png')

# 4) Price distribution
plt.figure(figsize=(8,5))
sns.histplot(df['price_inr'], bins=80, kde=True)
plt.title('Price Distribution')
plt.tight_layout()
plt.savefig('plots/price_distribution.png')

# 5) Compare simple algorithms for price prediction (SVM vs RandomForest)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# small sample for speed
sample = df.dropna(subset=['estimated_distance_km']).sample(min(15000, len(df)), random_state=1)
X = sample[['platform','pickup','drop','estimated_distance_km','hour','dayofweek']]
y = sample['price_inr']
# simple encoding
X_enc = pd.get_dummies(X, columns=['platform','pickup','drop'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)

print('Training RandomForest...')
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)

print('Training SVR (RBF)...')
svr = SVR(kernel='rbf', C=10, epsilon=5)
svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)
svr_mae = mean_absolute_error(y_test, svr_pred)

print('RF MAE:', rf_mae, 'SVR MAE:', svr_mae)

# Save comparison
with open('plots/model_comparison.txt', 'w') as f:
    f.write(f"RF MAE: {rf_mae}\nSVR MAE: {svr_mae}\n")

print('EDA complete. Plots saved to plots/ folder.')

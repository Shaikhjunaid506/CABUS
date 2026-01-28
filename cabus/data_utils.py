# data_utils.py
import pandas as pd
import numpy as np
from math import radians, sin, cos, asin, sqrt
from datetime import timedelta

# ---------- Coordinates ----------
COORDS = {
    "Connaught Place": (28.6315,77.2167),
    "Saket": (28.5245,77.2100),
    "Rohini": (28.7234,77.1115),
    "Dwarka": (28.5793,77.0421),
    "Lajpat Nagar": (28.5672,77.2431),
    "AIIMS": (28.5672,77.2100),
    "Anand Vihar": (28.6505,77.3151),
    "Rajouri Garden": (28.6456,77.1218),
    "Noida Sector 18": (28.5716,77.3256),
    "Noida Sector 62": (28.6156,77.3774),
    "Noida Sector 137": (28.4959,77.4310),
    "Noida Alpha 1": (28.5704,77.3170),
    "Knowledge Park II": (28.5155,77.3885),
    "Noida Atta Market": (28.5690,77.3200),
    "Greater Noida": (28.4744,77.5036)
}

# ---------- Haversine ----------
def haversine_km(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    return 6371 * c

# ---------- Time features ----------
def add_time_features(df, dt_col='datetime'):
    dts = pd.to_datetime(df[dt_col])
    df['hour'] = dts.dt.hour
    df['minute'] = dts.dt.minute
    df['dayofweek'] = dts.dt.dayofweek
    df['is_weekend'] = dts.dt.dayofweek >= 5
    # cyclical
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['dow_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['dayofweek']/7)
    df['date'] = dts.dt.date
    df[dt_col] = dts
    return df

# ---------- Distance ----------
def add_distance(df, pickup_col='pickup', drop_col='drop', trip_type_col='trip_type'):
    def compute(row):
        p = row[pickup_col]; d = row[drop_col]
        if p in COORDS and d in COORDS:
            lat1,lon1 = COORDS[p]; lat2,lon2 = COORDS[d]
            straight = haversine_km(lat1,lon1,lat2,lon2)
            rf = 1.1 if row.get(trip_type_col,'')!='Delhi-Noida' else 1.25
            return round(straight*rf,2)
        return np.nan
    df['distance_km'] = df.apply(compute, axis=1)
    return df

# ---------- Historical aggregates (fixed) ----------
def historical_aggregates(df, pickup, drop, center_dt, window_hours=10):
    """Return hourly aggregates (past window_hours up to center_dt) for the route or nearby fallback."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    end = pd.Timestamp(center_dt)
    start = end - pd.Timedelta(hours=window_hours)
    # exact route
    mask = (df['datetime'] >= start) & (df['datetime'] <= end) & (df['pickup']==pickup) & (df['drop']==drop)
    slice_df = df.loc[mask].copy()
    if slice_df.empty:
        # fallback: same pickup OR same drop
        mask2 = (df['datetime'] >= start) & (df['datetime'] <= end) & ((df['pickup']==pickup) | (df['drop']==drop))
        slice_df = df.loc[mask2].copy()
    if slice_df.empty:
        # fallback: any rides in window
        slice_df = df.loc[(df['datetime'] >= start) & (df['datetime'] <= end)].copy()
    if slice_df.empty:
        # return empty hour buckets
        hours = pd.date_range(start=end - pd.Timedelta(hours=window_hours), end=end, freq='H')
        platforms = df['platform'].unique().tolist()
        rows = []
        for h in hours:
            for p in platforms:
                rows.append({'hour_bucket':h,'platform':p,'avg_price':np.nan,'avg_surge':np.nan,'surge_rate':0.0,'avg_eta':np.nan,'count':0})
        return pd.DataFrame(rows)

    slice_df['hour_bucket'] = slice_df['datetime'].dt.floor('H')
    agg = slice_df.groupby(['hour_bucket','platform']).agg(
        avg_price=('price_inr','mean'),
        avg_surge=('surge_multiplier','mean'),
        surge_rate=('surge_multiplier', lambda x: float((x>1.0).mean())),
        avg_eta=('duration_min','mean'),
        count=('price_inr','size')
    ).reset_index()

    hours = pd.date_range(start=start.ceil('H'), end=end.floor('H'), freq='H')
    platforms = df['platform'].unique().tolist()
    full = []
    for h in hours:
        for p in platforms:
            rec = agg[(agg['hour_bucket']==h) & (agg['platform']==p)]
            if rec.empty:
                full.append({'hour_bucket':h,'platform':p,'avg_price':np.nan,'avg_surge':np.nan,'surge_rate':0.0,'avg_eta':np.nan,'count':0})
            else:
                full.append(rec.iloc[0].to_dict())
    return pd.DataFrame(full)

# ---------- Simple surge model (for UI & data augmentation) ----------
def surge_probability_by_hour(hour):
    if 8 <= hour < 10: return 0.38
    if 17 <= hour < 21: return 0.42
    if 13 <= hour < 15: return 0.12
    if 6 <= hour < 8: return 0.18
    return 0.06

def sample_surge(hour, distance_km, weekday=True):
    p = surge_probability_by_hour(hour)
    if distance_km > 20: p += 0.05
    if np.random.rand() < p:
        if 8 <= hour < 10 or 17 <= hour < 21:
            return round(np.random.uniform(1.2,2.0),2)
        else:
            return round(np.random.uniform(1.05,1.5),2)
    else:
        if np.random.rand() < 0.03:
            return round(np.random.uniform(0.5,0.95),2)
        return 1.0

# ---------- Live ETA simulator (simple) ----------
def simulate_eta(distance_km, hour, traffic_severity='moderate'):
    # base speeds by severity (km/h)
    speeds = {'low':40, 'moderate':28, 'heavy':15, 'gridlock':8}
    speed = speeds.get(traffic_severity, 28)
    # add hour-based slowdown (commute hours)
    if 8 <= hour < 10 or 17 <= hour < 21:
        speed *= 0.7
    # random jitter
    speed = speed * np.random.uniform(0.85,1.12)
    duration_min = max(3, int((distance_km / speed) * 60 + np.random.normal(0,5)))
    return duration_min


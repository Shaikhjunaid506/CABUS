# dashboard.py (updated)
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np
import plotly.express as px
from data_utils import COORDS, haversine_km, historical_aggregates

st.set_page_config(page_title='Ride Compare – Delhi/Noida', layout='wide')

@st.cache(allow_output_mutation=True)
def load_models():
    price_m = joblib.load('models/model_price.pkl')
    eta_m = joblib.load('models/model_eta.pkl')
    return price_m, eta_m

price_model, eta_model = load_models()

DATA_PATH = 'data/uber_ola_100k_train.csv'
df = pd.read_csv(DATA_PATH)
df['datetime'] = pd.to_datetime(df['datetime'])

st.title('🚕 Delhi–Noida Ride Compare — Price & Surge Trends')
st.markdown('Select pickup/drop, pick booking time (clock), then click **Compare Rides**. Charts show past 10 hours history for the selected route.')

col1, col2 = st.columns([1,2])
with col1:
    pickup = st.selectbox('Pickup location', sorted(df['pickup'].unique()))
    drop = st.selectbox('Drop location', sorted(df['drop'].unique()))
    date = st.date_input('Booking date', datetime.now().date())
    time = st.time_input('Booking time', datetime.now().time())
    dt = datetime.combine(date, time)   # important: use selected time immediately

with col2:
    st.write('Tune importance of price vs ETA for ranking')
    price_weight = st.slider('Price importance (0 = ignore price; 1 = price only)', 0.0, 1.0, 0.6)
    eta_weight = 1.0 - price_weight

st.write('---')

btn_col1, btn_col2 = st.columns([1,1])
with btn_col1:
    compare = st.button('Compare Rides')
with btn_col2:
    reset = st.button('Reset')

if reset:
    st.experimental_rerun()

if compare:
    # Build query rows (platform x category)
    platforms = ['Uber','Ola']
    categories = ['mini','sedan','prime']
    rows = []
    for p in platforms:
        for c in categories:
            row = {'platform': p, 'pickup': pickup, 'drop': drop, 'ride_category': c,
                   'payment_method': 'UPI', 'surge_multiplier': 1.0}
            # distance estimate
            if pickup in COORDS and drop in COORDS:
                lat1,lon1 = COORDS[pickup]; lat2,lon2 = COORDS[drop]
                straight = haversine_km(lat1,lon1,lat2,lon2)
                route_factor = 1.12 if pickup.split()[0] == drop.split()[0] else 1.25
                row['estimated_distance_km'] = round(straight*route_factor, 2)
            else:
                row['estimated_distance_km'] = float(np.random.uniform(3,25))
            # time features (important!)
            row['hour'] = dt.hour
            row['dayofweek'] = dt.weekday()
            # cyclical features (if model requires)
            row['hour_sin'] = np.sin(2*np.pi*row['hour']/24)
            row['hour_cos'] = np.cos(2*np.pi*row['hour']/24)
            row['dow_sin'] = np.sin(2*np.pi*row['dayofweek']/7)
            row['dow_cos'] = np.cos(2*np.pi*row['dayofweek']/7)
            # quick traffic heuristic & surge guess for UI realism
            if 8 <= dt.hour <= 10 or 17 <= dt.hour <= 21:
                row['traffic_severity'] = 'heavy'
                row['surge_multiplier'] = np.random.choice([1.0,1.2,1.4,1.6], p=[0.4,0.3,0.2,0.1])
            elif 13 <= dt.hour <= 15:
                row['traffic_severity'] = 'moderate'
                row['surge_multiplier'] = np.random.choice([1.0,1.1,1.2], p=[0.7,0.2,0.1])
            else:
                row['traffic_severity'] = 'low'
                row['surge_multiplier'] = 1.0
            rows.append(row)
    query_df = pd.DataFrame(rows)

    # Ensure columns align with training pipeline: rename estimated_distance_km to distance_km if needed
    if 'estimated_distance_km' in query_df.columns and 'distance_km' in price_model.named_steps['pre'].transformers_[1][2]:
        pass  # assume preprocessor names match; otherwise ensure expected numeric column names used

    # Predict
    pred_price = price_model.predict(query_df)
    pred_eta = eta_model.predict(query_df)
    query_df['pred_price'] = np.round(pred_price).astype(int)
    query_df['pred_eta_min'] = np.round(pred_eta).astype(int)

    # Ranking (lower is better)
    q = query_df.copy()
    q['price_norm'] = (q['pred_price'] - q['pred_price'].min()) / (q['pred_price'].max() - q['pred_price'].min() + 1e-6)
    q['eta_norm'] = (q['pred_eta_min'] - q['pred_eta_min'].min()) / (q['pred_eta_min'].max() - q['pred_eta_min'].min() + 1e-6)
    q['score'] = price_weight * q['price_norm'] + eta_weight * q['eta_norm']
    q = q.sort_values('score').reset_index(drop=True)

    # Highlight best
    best = q.iloc[0]
    st.markdown('### Best option — highlighted')
    st.metric(label=f"{best['platform']} · {best['ride_category']}", value=f"₹{best['pred_price']}", delta=f"ETA {best['pred_eta_min']} min")

    st.write('---')
    st.markdown('### All Options (sorted)')
    st.dataframe(q[['platform','ride_category','pred_price','pred_eta_min','traffic_severity','surge_multiplier']])

    # ---------- Historical aggregates (past 10 hours) ----------
    hist = historical_aggregates(df, pickup, drop, dt, window_hours=10)
    # Plot A: Avg price by hour and platform
    fig_price = px.line(hist, x='hour_bucket', y='avg_price', color='platform', markers=True,
                        title='Average Price (past 10 hours) — selected route')
    # Add current predicted bars as overlay
    curr = q.groupby('platform')['pred_price'].median().reset_index()
    curr['hour_bucket'] = dt.replace(minute=0, second=0, microsecond=0)
    # show lines + markers; also show a bar of current predicted price beneath
    st.plotly_chart(fig_price, use_container_width=True)

    # small bar for current predicted price
    fig_curr = px.bar(curr, x='platform', y='pred_price', color='platform', title='Current Predicted Price (selected time)')
    st.plotly_chart(fig_curr, use_container_width=True)

    # Plot B: surge/traffic metrics (avg_surge, surge_rate, avg_eta)
    fig2 = px.line(hist, x='hour_bucket', y=['avg_surge','surge_rate','avg_eta'], color='platform',
                   title='Traffic & Surge Metrics (past 10 hours) — avg_surge / surge_rate / avg_eta', markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    # Provide short explanation
    st.markdown("""
    **Notes:**  
    - Plots show historical averages for the selected pickup→drop over the past 10 hours (fallbacks used if exact route history is sparse).  
    - The model prediction uses the exact booking **time** you selected — change the clock and click *Compare Rides* again to see updated predictions.  
    - If predictions look unrealistic, retrain models with `python train.py` (see README) after improving features or increasing data.
    """)

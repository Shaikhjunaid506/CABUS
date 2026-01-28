# dashboard_ui.py
import streamlit as st
import pandas as pd, numpy as np, joblib
from datetime import datetime
import plotly.express as px
from data_utils import COORDS, haversine_km, historical_aggregates, simulate_eta, sample_surge

st.set_page_config(page_title="Cabus — Compare Rides", layout='wide', initial_sidebar_state='expanded')

@st.cache_resource
def load_models():
    price_m = joblib.load('models/model_price_xgb.pkl')
    eta_m = joblib.load('models/model_eta_xgb.pkl')
    return price_m, eta_m

price_model, eta_model = load_models()

# load dataset for choices and history
DATA_PATH = 'data/uber_ola_100k_train.csv'
df = pd.read_csv(DATA_PATH)
df['datetime'] = pd.to_datetime(df['datetime'])

st.title("🚖 Cabus — Best Ride Compare (Delhi ↔ Noida)")
st.markdown("Select pickup & drop, choose booking time (clock), then click **Compare Rides**.")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    pickup = st.selectbox("Pickup", sorted(df['pickup'].unique()))
    drop = st.selectbox("Drop", sorted(df['drop'].unique()))
    date = st.date_input("Booking date", datetime.now().date())
    time = st.time_input("Booking time", datetime.now().time())
    dt = datetime.combine(date, time)
    price_weight = st.slider("Price importance", 0.0, 1.0, 0.6)
    show_raw_history = st.checkbox("Show raw historical points", value=False)

if st.button("Compare Rides"):
    # Build queries for combinations
    platforms = ['Uber','Ola']
    categories = ['mini','sedan','prime']
    rows = []
    for p in platforms:
        for c in categories:
            d = {}
            d['platform'] = p; d['ride_category'] = c
            d['pickup'] = pickup; d['drop'] = drop
            # estimate distance
            if pickup in COORDS and drop in COORDS:
                lat1,lon1 = COORDS[pickup]; lat2,lon2 = COORDS[drop]
                straight = haversine_km(lat1,lon1,lat2,lon2)
                route_factor = 1.12 if pickup.split()[0]==drop.split()[0] else 1.25
                d['distance_km'] = round(straight*route_factor,2)
            else:
                d['distance_km'] = float(np.random.uniform(3,25))
            d['surge_multiplier'] = sample_surge(dt.hour, d['distance_km'])
            d['payment_method'] = 'UPI'
            d['traffic_severity'] = 'heavy' if (8<=dt.hour<=10 or 17<=dt.hour<=21) else ('moderate' if 13<=dt.hour<=15 else 'low')
            d['hour'] = dt.hour; d['dayofweek'] = dt.weekday()
            d['hour_sin'] = np.sin(2*np.pi*d['hour']/24); d['hour_cos'] = np.cos(2*np.pi*d['hour']/24)
            d['dow_sin'] = np.sin(2*np.pi*d['dayofweek']/7); d['dow_cos'] = np.cos(2*np.pi*d['dayofweek']/7)
            rows.append(d)
    qdf = pd.DataFrame(rows)

    # Predict price & ETA
    pred_price = price_model.predict(qdf)
    pred_eta = eta_model.predict(qdf)
    qdf['pred_price'] = np.round(pred_price).astype(int)
    qdf['pred_eta'] = np.round(pred_eta).astype(int)

    # Live ETA simulator override/augment (show both)
    qdf['sim_eta'] = qdf.apply(lambda r: simulate_eta(r['distance_km'], int(r['hour']), r['traffic_severity']), axis=1)

    # Scoring & ranking
    qdf['price_norm'] = (qdf['pred_price'] - qdf['pred_price'].min()) / (qdf['pred_price'].max()-qdf['pred_price'].min()+1e-6)
    qdf['eta_norm']  = (qdf['sim_eta'] - qdf['sim_eta'].min()) / (qdf['sim_eta'].max()-qdf['sim_eta'].min()+1e-6)
    qdf['score'] = price_weight*qdf['price_norm'] + (1-price_weight)*qdf['eta_norm']
    qdf = qdf.sort_values('score').reset_index(drop=True)

    # UI top card
    best = qdf.iloc[0]
    st.markdown("### 🔥 Recommended option")
    st.markdown(f"**{best['platform']} — {best['ride_category']}**  •  **₹{best['pred_price']}**  •  ETA: **{best['sim_eta']} min**")
    st.write("---")

    # Show table
    st.dataframe(qdf[['platform','ride_category','pred_price','pred_eta','sim_eta','traffic_severity','surge_multiplier']])

    # Historical aggregates (past 10 hours)
    hist = historical_aggregates(df, pickup, drop, dt, window_hours=10)
    # Plot A: avg price line + current predicted bars
    fig_price = px.line(hist, x='hour_bucket', y='avg_price', color='platform', markers=True,
                        labels={'hour_bucket':'Hour','avg_price':'Avg Price (INR)'}, title='Avg Price — Past 10 hours')
    if show_raw_history:
        # overlay scatter of raw points
        raw_mask = (df['datetime'] >= pd.Timestamp(dt) - pd.Timedelta(hours=10)) & (df['datetime'] <= pd.Timestamp(dt)) & ((df['pickup']==pickup)&(df['drop']==drop))
        raw = df.loc[raw_mask]
        if not raw.empty:
            fig_price.add_scatter(x=raw['datetime'], y=raw['price_inr'], mode='markers', name='raw prices', marker=dict(size=6, opacity=0.6))

    st.plotly_chart(fig_price, use_container_width=True)

    # Current predicted price bar
    cur = qdf.groupby('platform')['pred_price'].min().reset_index()
    fig_cur = px.bar(cur, x='platform', y='pred_price', color='platform', title='Current predicted price')
    st.plotly_chart(fig_cur, use_container_width=True)

    # Plot B: avg_surge, surge_rate, avg_eta (line chart)
    if not hist.empty:
        # melt for multi-series
        hist_long = hist.copy()
        hist_long['hour'] = hist_long['hour_bucket']
        fig2 = px.line(hist_long, x='hour', y='avg_surge', color='platform', markers=True, title='Avg Surge (past 10h)')
        fig3 = px.line(hist_long, x='hour', y='surge_rate', color='platform', markers=True, title='Surge Rate (past 10h)')
        fig4 = px.line(hist_long, x='hour', y='avg_eta', color='platform', markers=True, title='Avg ETA (past 10h)')
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No historical data found for this route/time window.")

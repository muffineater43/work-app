import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

st.set_page_config(layout="wide")
st.title("3-Month Fly vs Outright Dashboard")

# ----------------------
# 1) Upload two CSVs
# ----------------------
fly_file = st.sidebar.file_uploader("Upload butterfly CSV", type="csv")
leg_file = st.sidebar.file_uploader("Upload outright CSV", type="csv")

if not fly_file or not leg_file:
    st.sidebar.info("Please upload both a butterfly and an outright CSV.")
    st.stop()

# ----------------------
# 2) Load & align series
# ----------------------
@st.cache_data
def load_series(file):
    df = pd.read_csv(file, parse_dates=["Timestamp (UTC)"])
    df.set_index("Timestamp (UTC)", inplace=True)
    # drop any timezone info
    df.index = pd.to_datetime(df.index).tz_localize(None)
    # pick the Close column if present, otherwise first numeric
    if "Close" in df.columns:
        return df["Close"]
    return df.select_dtypes(include="number").iloc[:, 0]

fly = load_series(fly_file).rename("fly")
leg = load_series(leg_file).rename("leg")

# intersect timestamps
common_idx = fly.index.intersection(leg.index)
fly = fly.loc[common_idx]
leg = leg.loc[common_idx]

# build aligned DataFrame
df = pd.DataFrame({"fly": fly, "leg": leg}).dropna().sort_index()

# ----------------------
# 3) Rolling 3-month regression
# ----------------------
betas, alphas = [], []
min_periods   = 50
window_months = 3

for t in df.index:
    start  = t - pd.DateOffset(months=window_months)
    window = df.loc[(df.index >= start) & (df.index <= t)]
    if len(window) < min_periods:
        betas.append(np.nan)
        alphas.append(np.nan)
    else:
        m, b0 = np.polyfit(window["leg"], window["fly"], 1)
        betas.append(m)
        alphas.append(b0)

df["beta"]      = betas
df["intercept"] = alphas
df["predicted"] = df["beta"] * df["leg"] + df["intercept"]
df["residual"]  = df["fly"] - df["predicted"]

# ----------------------
# 4) Compute metrics
# ----------------------
res = df["residual"].dropna()
mu       = res.mean()
sigma    = res.std(ddof=1)
skw      = skew(res)
kurt_p   = kurtosis(res, fisher=False)
latest_z = (res.iloc[-1] - mu) / sigma

# ----------------------
# 5) Display results
# ----------------------
st.subheader("Residual Summary Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean",        f"{mu:.4f}")
c2.metric("Std Dev",     f"{sigma:.4f}")
c3.metric("Skewness",    f"{skw:.4f}")
c4.metric("Kurtosis",    f"{kurt_p:.4f}")
c5.metric("Latest Z-Score", f"{latest_z:.2f}")

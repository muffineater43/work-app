import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm
import warnings

# suppress irrelevant warnings
warnings.filterwarnings("ignore", message="Workbook contains no default style")

st.set_page_config(layout="wide")
st.title("3-Month Fly vs Outright Dashboard")

# 1) Upload two CSVs
fly_file = st.sidebar.file_uploader("Upload butterfly CSV", type="csv")
leg_file = st.sidebar.file_uploader("Upload outright CSV",   type="csv")
if not fly_file or not leg_file:
    st.sidebar.info("Please upload both a butterfly and an outright CSV.")
    st.stop()

# 2) Load, parse, and align on intersection of dates
@st.cache_data
def load_series(file):
    df = pd.read_csv(file, parse_dates=["Timestamp (UTC)"])
    df.set_index("Timestamp (UTC)", inplace=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    if "Close" in df.columns:
        return df["Close"]
    return df.select_dtypes(include=np.number).iloc[:, 0]

fly = load_series(fly_file).rename("fly")
leg = load_series(leg_file).rename("leg")

common_idx = fly.index.intersection(leg.index)
fly = fly.loc[common_idx]
leg = leg.loc[common_idx]
df = pd.DataFrame({"fly": fly, "leg": leg}).dropna()
df.sort_index(inplace=True)

# 3) Rolling 3-month regression
betas, alphas = [], []
min_periods = 50
window_months = 3
for t in df.index:
    start = t - pd.DateOffset(months=window_months)
    window = df.loc[(df.index >= start) & (df.index <= t)]
    if len(window) < min_periods:
        betas.append(np.nan)
        alphas.append(np.nan)
    else:
        m, b0 = np.polyfit(window["leg"], window["fly"], 1)
        betas.append(m)
        alphas.append(b0)

df["beta"] = betas
df["intercept"] = alphas
df["predicted"] = df["beta"] * df["leg"] + df["intercept"]
df["residual"] = df["fly"] - df["predicted"]

# 4) Compute metrics
res = df["residual"].dropna()
mu = float(res.mean())
sigma = float(res.std(ddof=1))
skw = float(skew(res))
kurt_p = float(kurtosis(res, fisher=False))
latest_z = float((res.iloc[-1] - mu) / sigma)

# 5) Display metrics and save button
st.subheader("Residual Summary Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean",     f"{mu:.4f}")
c2.metric("Std Dev",  f"{sigma:.4f}")
c3.metric("Skewness", f"{skw:.4f}")
c4.metric("Kurtosis", f"{kurt_p:.4f}")
c5.metric("Latest Z", f"{latest_z:.2f}")

if "history" not in st.session_state:
    st.session_state.history = []
if st.button("Save Metrics"):
    st.session_state.history.append({
        "butterfly": fly_file.name,
        "mean": mu,
        "std": sigma,
        "skew": skw,
        "kurtosis": kurt_p,
        "z_score": latest_z
    })

# 6) Show and download history
if st.session_state.history:
    st.subheader("Saved Metrics History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)
    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Metrics CSV",
        data=csv,
        file_name="metrics_history.csv",
        mime="text/csv"
    )

# 7) Optional histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(res, bins=30, density=True, alpha=0.6)
x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
ax.plot(x_vals, norm.pdf(x_vals, mu, sigma), linewidth=2)
ax.set_title("Residuals Histogram")
st.pyplot(fig, use_container_width=True)

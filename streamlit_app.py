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

# 2) Load and align on common dates
df_fly = pd.read_csv(fly_file, parse_dates=["Timestamp (UTC)"])
df_leg = pd.read_csv(leg_file, parse_dates=["Timestamp (UTC)"])
common_times = set(df_fly["Timestamp (UTC)"]) & set(df_leg["Timestamp (UTC)"])
fly_common = df_fly[df_fly["Timestamp (UTC)"].isin(common_times)].sort_values("Timestamp (UTC)")
leg_common = df_leg[df_leg["Timestamp (UTC)"].isin(common_times)].sort_values("Timestamp (UTC)")

df = pd.DataFrame({
    "leg": leg_common.set_index("Timestamp (UTC)")["Close"],
    "fly": fly_common.set_index("Timestamp (UTC)")["Close"],
}).dropna()
df.index = pd.to_datetime(df.index)

# 3) Rolling 3-month regression
betas, alphas = [], []
min_periods = 50
for t in df.index:
    window_df = df.loc[(df.index >= t - pd.DateOffset(months=3)) & (df.index <= t)]
    if len(window_df) < min_periods:
        betas.append(np.nan)
        alphas.append(np.nan)
    else:
        m, b0 = np.polyfit(window_df["leg"], window_df["fly"], 1)
        betas.append(m)
        alphas.append(b0)

df["beta"]      = betas
df["intercept"] = alphas
df["predicted"] = df["beta"] * df["leg"] + df["intercept"]
df["residual"]  = df["fly"] - df["predicted"]

# 4) Compute metrics
residuals = df["residual"].dropna()
mu        = residuals.mean()
sigma     = residuals.std(ddof=1)
skw       = skew(residuals)
kurt_p    = kurtosis(residuals, fisher=False)
latest_z  = (residuals.iloc[-1] - mu) / sigma

# 5) Display summary metrics
st.subheader("Residual Summary Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean",        f"{mu:.4f}")
c2.metric("Std Dev",     f"{sigma:.4f}")
c3.metric("Skewness",    f"{skw:.4f}")
c4.metric("Kurtosis",    f"{kurt_p:.4f}")
c5.metric("Latest Z-Score", f"{latest_z:.2f}")

# 6) Save current metrics
if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []
if st.button("Save current metrics"):
    st.session_state.metrics_history.append({
        "butterfly": fly_file.name,
        "mean": float(mu),
        "std": float(sigma),
        "skew": float(skw),
        "kurtosis": float(kurt_p),
        "z_score": float(latest_z)
    })

# 7) Display saved history and download
if st.session_state.metrics_history:
    st.subheader("Saved Metrics History")
    history_df = pd.DataFrame(st.session_state.metrics_history)
    st.dataframe(history_df)
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download saved metrics",
        data=csv,
        file_name="metrics_history.csv",
        mime="text/csv"
    )

# 8) (Optional) Histogram with fitted normal curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=30, density=True, alpha=0.6)
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
ax.plot(x, norm.pdf(x, mu, sigma), linewidth=2)
ax.set_title("Residuals Histogram with Fitted Normal Curve")
st.pyplot(fig, use_container_width=True)

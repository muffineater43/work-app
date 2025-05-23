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
common = set(df_fly["Timestamp (UTC)"]) & set(df_leg["Timestamp (UTC)"])
fly_common = df_fly[df_fly["Timestamp (UTC)"].isin(common)].sort_values("Timestamp (UTC)")
leg_common = df_leg[df_leg["Timestamp (UTC)"].isin(common)].sort_values("Timestamp (UTC)")

df = pd.DataFrame({
    "leg": leg_common.set_index("Timestamp (UTC)")["Close"],
    "fly": fly_common.set_index("Timestamp (UTC)")["Close"],
}).dropna()
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

# 3) Rolling 3-month regression
betas, alphas = [], []
min_periods = 50
window_months = 3
for t in df.index:
    start = t - pd.DateOffset(months=window_months)
    window_df = df.loc[(df.index >= start) & (df.index <= t)]
    if len(window_df) < min_periods:
        betas.append(np.nan)
        alphas.append(np.nan)
    else:
        m, b0 = np.polyfit(window_df["leg"], window_df["fly"], 1)
        betas.append(m)
        alphas.append(b0)

# Attach regression results
df["beta"] = betas
df["intercept"] = alphas
df["predicted"] = df["beta"] * df["leg"] + df["intercept"]
df["residual"] = df["fly"] - df["predicted"]

# 4) Compute metrics
residuals = df["residual"].dropna()
mu = residuals.mean()
sigma = residuals.std(ddof=1)
skw = skew(residuals)
kurt_p = kurtosis(residuals, fisher=False)
latest_z = (residuals.iloc[-1] - mu) / sigma

# 5) Display metrics
st.subheader("Residual Summary Statistics")
st.write(f"Mean:     {mu:.4f}")
st.write(f"Std Dev:  {sigma:.4f}")
st.write(f"Skewness: {skw:.4f}")
st.write(f"Kurtosis: {kurt_p:.4f}")
st.write(f"Latest Z-score: {latest_z:.2f}")

# 6) Save feature
if "history" not in st.session_state:
    st.session_state.history = []

def save_metrics():
    st.session_state.history.append({
        "butterfly": fly_file.name,
        "mean": float(mu),
        "std": float(sigma),
        "skew": float(skw),
        "kurtosis": float(kurt_p),
        "z_score": float(latest_z)
    })

st.button("Save Metrics", on_click=save_metrics)

# 7) Show history
if st.session_state.history:
    st.subheader("Saved Metrics History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)
    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Metrics CSV", csv, "metrics_history.csv", "text/csv")

# 8) Optional histogram
fig, ax = plt.subplots()
ax.hist(residuals, bins=30, density=True, alpha=0.6)
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
ax.plot(x, norm.pdf(x, mu, sigma), linewidth=2)
ax.set_title("Residuals Histogram with Fitted Normal Curve")
st.pyplot(fig, use_container_width=True)

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

# 3) Rolling 3-month regression
betas, alphas = [], []
min_periods = 50
for t in df.index:
    window_df = df.loc[df.index >= (t - pd.DateOffset(months=3))]
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
mu      = residuals.mean()
sigma   = residuals.std(ddof=1)
skw     = skew(residuals)
kurt_p  = kurtosis(residuals, fisher=False)
latest_z = (residuals.iloc[-1] - mu) / sigma

# 5) Display summary metrics
st.subheader("Latest Residual Z-Score & Summary Stats")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Mean",        f"{mu:.4f}")
col2.metric("Std Dev",     f"{sigma:.4f}")
col3.metric("Skewness",    f"{skw:.4f}")
col4.metric("Kurtosis",    f"{kurt_p:.4f}")
col5.metric("Latest Z",    f"{latest_z:.2f}")

# 6) Show historical values in a table
st.subheader("Historical Regression Results")
# reset index so Timestamp becomes a column
df_display = df.reset_index().rename(columns={"index": "Timestamp"})
st.dataframe(df_display)

# 7) Allow download of full history as CSV
csv = df_display.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download full history as CSV",
    data=csv,
    file_name="fly_outright_history.csv",
    mime="text/csv"
)

# 8) (Optional) Histogram with fitted normal curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=30, density=True, alpha=0.6)
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
ax.plot(x, norm.pdf(x, mu, sigma), linewidth=2)
ax.set_title("Residuals Histogram with Fitted Normal Curve")
st.pyplot(fig, use_container_width=True)

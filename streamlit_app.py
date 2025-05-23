import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

st.set_page_config(layout="wide")
st.title("3-Month Fly vs Outright Dashboard")

# ----------------------
# 1) FILE UPLOAD / DATA LOAD
# ----------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload your contract CSVs (outrights and butterflies)",
    type=["csv"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.sidebar.info("Please upload one or more CSV files to get started.")
    st.stop()

@st.cache_data
def load_contracts(files):
    series = []
    for file in files:
        df = pd.read_csv(file, parse_dates=[0])
        df.columns = ["Timestamp (UTC)"] + list(df.columns[1:])
        df.set_index("Timestamp (UTC)", inplace=True)
        df.index = df.index.tz_localize(None)
        name = file.name.rsplit(".", 1)[0]
        price = df["Close"] if "Close" in df.columns else df.select_dtypes("number").iloc[:,0]
        series.append(price.rename(name))
    # INNER join: only timestamps common to ALL series
    return pd.concat(series, axis=1, join="inner")

raw_df = load_contracts(uploaded_files)


# ----------------------
# 2) SIDEBAR: CONTRACT DETECTION & SELECTION
# ----------------------
butterflies = [c for c in raw_df.columns if "butterfly" in c.lower()]
outrights   = [c for c in raw_df.columns if c not in butterflies]

st.sidebar.subheader("Contract selection")
out_contract = st.sidebar.selectbox("Choose outright:",   outrights or raw_df.columns)
fly_contract = st.sidebar.selectbox("Choose butterfly:", butterflies or raw_df.columns)

if out_contract == fly_contract:
    st.sidebar.error("Outright and butterfly must be different.")
    st.stop()

# ----------------------
# 3) PARAMETERS
# ----------------------
min_periods   = st.sidebar.number_input("Min points for regression", 10, 200, 50)
window_months = st.sidebar.slider("Rolling window (months)", 1, 12, 3)

# ----------------------
# 4) PREP & SHOW SELECTED
# ----------------------
df = raw_df[[out_contract, fly_contract]].copy()
df.sort_index(inplace=True)
st.subheader("Selected Contracts Aligned")
st.line_chart(df)

# ----------------------
# 5) ROLLING REGRESSION
# ----------------------
betas, alphas = [], []
for t in df.index:
    start  = t - pd.DateOffset(months=window_months)
    window = df.loc[(df.index >= start) & (df.index <= t)]
    if len(window) < min_periods:
        betas.append(np.nan)
        alphas.append(np.nan)
    else:
        x = window[out_contract]
        y = window[fly_contract]
        slope, intercept = np.polyfit(x, y, 1)
        betas.append(slope)
        alphas.append(intercept)

df["beta"]      = betas
df["intercept"] = alphas
df["predicted"] = df["beta"] * df[out_contract] + df["intercept"]
df["residual"]  = df[fly_contract] - df["predicted"]

# ----------------------
# 6) STATS & METRIC
# ----------------------
mu       = df["residual"].mean()
sigma    = df["residual"].std(ddof=1)
latest   = df["residual"].iloc[-1]
z_score  = (latest - mu) / sigma

st.subheader("Latest Residual Z-Score")
st.metric(label=f"As of {df.index[-1].date()}", value=f"{z_score:.2f}")

# ----------------------
# 7) DISTRIBUTION PLOT
# ----------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(df["residual"].dropna(), bins=50, density=True, alpha=0.6)
x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
ax.plot(x_vals, norm.pdf(x_vals, mu, sigma), linewidth=2)
ax.set_title("Residuals vs. Normal PDF")
st.pyplot(fig, use_container_width=True)

# ----------------------
# 8) SUMMARY & BETA CHART
# ----------------------
st.subheader("Residual Summary Statistics")
st.table(pd.DataFrame({
    "Mean":     [mu],
    "Std Dev":  [sigma],
    "Skew":     [skew(df["residual"].dropna())],
    "Kurtosis": [kurtosis(df["residual"].dropna(), fisher=False)]
}, index=["Value"]))

if st.checkbox("Show β over time"):
    st.subheader(f"Rolling β: {out_contract} vs {fly_contract}")
    st.line_chart(df["beta"].dropna())

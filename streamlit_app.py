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
    # Read each CSV, parse first column as datetime, pick 'Close' price
    series = {}
    for file in files:
        df = pd.read_csv(file, parse_dates=[0])
        df.columns = ["Timestamp (UTC)"] + list(df.columns[1:])
        df.set_index("Timestamp (UTC)", inplace=True)
        df.index = df.index.tz_localize(None)
        name = file.name.rsplit('.', 1)[0]
        # Prefer 'Close' column
        if 'Close' in df.columns:
            series[name] = df['Close'].rename(name)
        else:
            # fallback to first numeric column
            num_cols = df.select_dtypes(include=np.number).columns
            series[name] = df[num_cols[0]].rename(name)
    # combine into one DataFrame aligned on index
    return pd.concat(series.values(), axis=1)

raw_df = pd.concat(series.values(), axis=1, join="inner")

# ----------------------
# 2) SIDEBAR: CONTRACT TYPE DETECTION & SELECTION
# ----------------------
butterflies = [c for c in raw_df.columns if 'butterfly' in c.lower()]
outrights = [c for c in raw_df.columns if c not in butterflies]

st.sidebar.subheader("Contract selection")
if outrights:
    out_contract = st.sidebar.selectbox("Choose outright:", outrights)
else:
    out_contract = st.sidebar.selectbox("Choose outright:", raw_df.columns)
if butterflies:
    fly_contract = st.sidebar.selectbox("Choose butterfly:", butterflies)
else:
    fly_contract = st.sidebar.selectbox("Choose butterfly:", raw_df.columns)

if out_contract == fly_contract:
    st.sidebar.error("Outright and butterfly must be different.")
    st.stop()

# ----------------------
# 3) PARAMETERS
# ----------------------
min_periods = st.sidebar.number_input("Min points for regression", min_value=10, max_value=200, value=50)
window_months = st.sidebar.slider("Rolling window (months)", 1, 12, 3)

# Prepare analysis DataFrame
df = raw_df[[out_contract, fly_contract]].dropna()
df.sort_index(inplace=True)   
# ----------------------
# 4) ROLLING REGRESSION
# ----------------------
betas, alphas = [], []
for t in df.index:
    start = t - pd.DateOffset(months=window_months)
    # inclusive slice: between start AND t
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

# Attach results to df
df['beta'] = betas
df['intercept'] = alphas
df['predicted'] = df['beta'] * df[out_contract] + df['intercept']
df['residual'] = df[fly_contract] - df['predicted']

# ----------------------
# 5) STATS & METRICS
# ----------------------
mu = df['residual'].mean()
sigma = df['residual'].std(ddof=1)
latest = df['residual'].iloc[-1]
z_score = (latest - mu) / sigma

st.subheader("Latest Residual Z-Score")
st.metric(label=f"As of {df.index[-1].date()}", value=f"{z_score:.2f}")

# ----------------------
# 6) DISTRIBUTION PLOT
# ----------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(df['residual'].dropna(), bins=50, density=True, alpha=0.6)
x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
ax.plot(x_vals, norm.pdf(x_vals, mu, sigma), linewidth=2)
ax.set_title("Residuals vs. Normal PDF", fontsize=12)
st.pyplot(fig, use_container_width=True)

# ----------------------
# 7) SUMMARY STATISTICS
# ----------------------
st.subheader("Residual Summary Statistics")
st.table(pd.DataFrame({
    'Mean': [mu],
    'Std Dev': [sigma],
    'Skewness': [skew(df['residual'].dropna())],
    'Kurtosis': [kurtosis(df['residual'].dropna(), fisher=False)]
}, index=["Value"]))

# ----------------------
# 8) BETA OVER TIME
# ----------------------
if st.checkbox("Show β over time"):
    st.subheader(f"Rolling β for {out_contract} vs {fly_contract}")
    st.line_chart(df['beta'].dropna())

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
    "Upload your contract CSVs (outrights and butterfly)",
    type=["csv"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.sidebar.info("Please upload one or more CSV files to get started.")
    st.stop()

@st.cache_data
def load_contracts(files):
    contracts = {}
    for f in files:
        # read CSV, expect a Date-Time or Timestamp column plus OHLC columns
        df = pd.read_csv(f, parse_dates=[0])
        # ensure timestamp index
        df.columns = ["Date-Time"] + list(df.columns[1:])
        df.set_index("Date-Time", inplace=True)
        df.index = df.index.tz_localize(None)
        # derive a friendly name from filename
        name = f.name.replace(".csv", "")
        # explicitly pick the 'Close' price column
        if 'Close' in df.columns:
            contracts[name] = df['Close'].rename(name)
        else:
            # fallback to first numeric column
            numeric_cols = df.select_dtypes('number').columns
            contracts[name] = df[numeric_cols[0]].rename(name)
    return pd.concat(contracts.values(), axis=1)(contracts.values(), axis=1)

# build one master DataFrame of all series
raw_df = load_contracts(uploaded_files)

# ----------------------
# 2) SIDEBAR CONTRACT SELECTION WITH TYPE FILTERING
# ----------------------
# Automatically split between butterfly and outright series based on filename
butterflies = [c for c in raw_df.columns if 'butterfly' in c.lower()]
outrights = [c for c in raw_df.columns if c not in butterflies]

st.sidebar.subheader("Contract selection")
if outrights:
    out_contract = st.sidebar.selectbox(
        "Choose outright contract:", outrights,
        index=0
    )
else:
    out_contract = st.sidebar.selectbox(
        "Choose outright contract:", raw_df.columns,
        index=0
    )

if butterflies:
    fly_contract = st.sidebar.selectbox(
        "Choose butterfly contract:", butterflies,
        index=0
    )
else:
    fly_contract = st.sidebar.selectbox(
        "Choose butterfly contract:", raw_df.columns,
        index=len(raw_df.columns)-1
    )

# ensure two distinct
if out_contract == fly_contract:
    st.sidebar.error("Outright and butterfly must be different.")
    st.stop()
if out_contract == fly_contract:
    st.sidebar.error("Outright and butterfly must be different.")
    st.stop()

# ----------------------
# 3) PREP & PARAMETERS
# ----------------------
min_periods = st.sidebar.number_input(
    "Min points for regression", 10, 200, 50
)
window_months = st.sidebar.slider(
    "Rolling window (months)", 1, 12, 3
)

# form df for analysis
df = raw_df[[out_contract, fly_contract]].dropna()

# ----------------------
# 4) CALC: rolling regression
# ----------------------
betas, alphas = [], []
for t in df.index:
    window = df.loc[t - pd.DateOffset(months=window_months) : t]
    if len(window) < min_periods:
        betas.append(np.nan)
        alphas.append(np.nan)
    else:
        # fly ~ beta * outright + intercept
        slope, intercept = np.polyfit(window[out_contract], window[fly_contract], 1)
        betas.append(slope)
        alphas.append(intercept)

# attach results
df = df.assign(beta=betas, intercept=alphas)
df['predicted'] = df['beta'] * df[out_contract] + df['intercept']
df['residual'] = df[fly_contract] - df['predicted']

# ----------------------
# 5) STATS & METRICS
# ----------------------
mu, sigma = df['residual'].mean(), df['residual'].std(ddof=1)
latest_res = df['residual'].iloc[-1]
z_score = (latest_res - mu) / sigma

st.subheader("Latest Residual Z-Score")
st.metric(
    label=f"Z-score as of {df.index[-1].date()}",
    value=f"{z_score:.2f}"
)

# ----------------------
# 6) DISTRIBUTION PLOT
# ----------------------
fig, ax = plt.subplots()
ax.hist(df['residual'].dropna(), bins=50, density=True, alpha=0.6)
x_vals = np.linspace(mu-4*sigma, mu+4*sigma, 200)
ax.plot(x_vals, norm.pdf(x_vals, mu, sigma), linewidth=2)
ax.set_title("Residuals vs. Normal PDF")
st.pyplot(fig)

# ----------------------
# 7) SUMMARY TABLE
# ----------------------
st.subheader("Residuals Summary Statistics")
st.table(
    pd.DataFrame({
        'Mean': [mu],
        'Std Dev': [sigma],
        'Skewness': [skew(df['residual'].dropna())],
        'Kurtosis': [kurtosis(df['residual'].dropna(), fisher=False)]
    }, index=["Value"])
)

# ----------------------
# 8) BETA OVER TIME
# ----------------------
if st.checkbox("Show β over time"):
    st.subheader(f"Rolling β: {out_contract} vs {fly_contract}")
    st.line_chart(df['beta'].dropna())

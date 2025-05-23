import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm
warnings.filterwarnings("ignore", message="Workbook contains no default style")

df1 = pd.read_csv("SR3 Dec25 3mo Butterfly_Daily.csv")
df2 = pd.read_csv("SR3 Dec25_Daily.csv")
df3 = pd.read_csv("SR3 Jun26_Daily.csv")
df4 = pd.read_csv("SR3 Mar26_Daily.csv")

common_times = set(df1['Timestamp (UTC)']) & set(df2['Timestamp (UTC)']) & set(df3['Timestamp (UTC)']) & set(df4['Timestamp (UTC)'])
leg3_common = df4[df4['Timestamp (UTC)'].isin(common_times)]
leg2_common = df3[df3['Timestamp (UTC)'].isin(common_times)]
leg1_common = df2[df2['Timestamp (UTC)'].isin(common_times)]
fly_common = df1[df1['Timestamp (UTC)'].isin(common_times)]


df = pd.DataFrame({
    'leg3':   leg3_common.set_index('Timestamp (UTC)')['Close'],
    'leg2':   leg2_common.set_index('Timestamp (UTC)')['Close'],
    'leg1':   leg1_common.set_index('Timestamp (UTC)')['Close'],
    'fly': fly_common.set_index('Timestamp (UTC)')['Close'],
}).dropna()

df.index = pd.to_datetime(df.index)


#Fly vs Outright
betas, alphas = [], []

min_periods = 50

for t in df.index:
    window_df = df.loc[df.index >= t - pd.DateOffset(months=3)]
    
    if len(window_df) < min_periods:
        betas.append(np.nan)
        alphas.append(np.nan)
        continue

    slope, intercept = np.polyfit(window_df['leg2'], window_df['fly'], 1)
    betas.append(slope)
    alphas.append(intercept)


df['beta']      = betas
df['intercept'] = alphas

df['predicted'] = df['beta'] * df['leg2'] + df['intercept']
df['residual']  = df['fly'] - df['predicted']

mu, sigma = df['residual'].mean(), df['residual'].std(ddof=1)
latest_R   = df['residual'].iloc[-1]
z_score    = (latest_R - mu) / sigma

print(f"Used a true 3-month slice ending at {df.index[-1]}")
print(f"Latest Z-score = {z_score:.2f}")


residuals = df['residual'].dropna()
mu, sigma = residuals.mean(), residuals.std(ddof=1)

skewness = skew(residuals)
kurt = kurtosis(residuals, fisher=False)

print(f"Mean: {mu:.4f}")
print(f"Std Dev: {sigma:.4f}")
print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis (Pearson): {kurt:.4f}")

x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
pdf = norm.pdf(x, mu, sigma)

plt.figure()
plt.hist(residuals, bins=30, density=True)
plt.plot(x, pdf)
plt.title('Residuals Histogram with Fitted Normal Curve')
plt.xlabel('Residual')
plt.ylabel('Density')
plt.show()


#If the kurtosis is greater than 3, leptokurtic 
#If neat 0, normal distribution 
#If less than 0, platykurtic 

#positive skewness, tail on the right 
#negative skewness, tail on the left

#z-score -1.5 to 1.5 is 0.86639 in distribution 
#The probabilty of greater than 1.5 is 0.0668,6.68% (top)
#Less than -1.5 is 0.0668, 6.68% (bottom) 


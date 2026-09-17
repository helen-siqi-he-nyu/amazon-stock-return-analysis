# Install dependencies: python -m pip install pandas numpy yfinance matplotlib seaborn scipy statsmodels
# Adapted from the supplied course notebook; run the .ipynb for submission and visible tables.

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# %%
TICKER = "AMZN"  # Change this to your chosen publicly traded stock.
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"  # Exclusive end date, matching the supplied notebook.

# Download daily data (AMZN) and keep adjusted close
df = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    progress=False
)

if df.empty:
    raise ValueError("No price data downloaded. Check ticker and internet connection.")
# Keep adjusted close only
df = df.loc[:, ["Adj Close"]]

# yfinance often returns a MultiIndex (field, ticker). Flatten for a single-asset notebook.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.rename(columns={"Adj Close": "adj_close"})
df.index.name = "Date"

df


# %%
print(f'Downloaded {df.shape[0]} rows of data')
df.head()

# %%
# Compute returns (in decimal form; e.g., 0.01 = 1%)
df["simple_rtn"] = df["adj_close"].pct_change()
df["log_rtn"] = np.log(df["adj_close"] / df["adj_close"].shift(1))

df.head()


# %%
df = df.dropna(how="any").copy()
df.head()


# %%
# =========================
# Histogram comparison: price vs. returns (with color contrast)
# =========================

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Price histogram (neutral / muted)
ax[0].hist(
    df["adj_close"],
    bins=50,
    color="gray",
    alpha=0.7,
    edgecolor="black"
)
ax[0].set_title(f"Distribution of Prices ({TICKER})")
ax[0].set_xlabel("Price level")
ax[0].set_ylabel("Frequency")

# Return histogram (analytic focus)
ax[1].hist(
    df["simple_rtn"].dropna(),
    bins=50,
    color="steelblue",
    alpha=0.8,
    edgecolor="black"
)
ax[1].set_title(f"Distribution of Daily Returns ({TICKER})")
ax[1].set_xlabel("Daily return")
ax[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


# %%
fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Price
df["adj_close"].plot(ax=ax[0])
ax[0].set(
    title=f"{TICKER}: Adjusted close price",
    ylabel="Price ($)"
)

# Simple returns (percent)
(df["simple_rtn"] * 100).plot(ax=ax[1])
ax[1].set(ylabel="Simple return (%)")

# Log returns (percent)
(df["log_rtn"] * 100).plot(ax=ax[2])
ax[2].set(
    xlabel="Date",
    ylabel="Log return (%)"
)

ax[2].tick_params(axis="x", which="major", labelsize=12)
plt.tight_layout()
plt.show()


# %%
# =========================
# Descriptive statistics: price and returns
# (Returns shown in percent for readability)
# =========================

price = df["adj_close"]
simple_rtn_pct = df["simple_rtn"] * 100
log_rtn_pct = df["log_rtn"] * 100

stats = pd.DataFrame({
    "Price ($)": price,
    "Simple Return (%)": simple_rtn_pct,
    "Log Return (%)": log_rtn_pct
}).describe().T

stats["Skewness"] = [price.skew(), simple_rtn_pct.skew(), log_rtn_pct.skew()]
stats["Kurtosis"] = [price.kurtosis(), simple_rtn_pct.kurtosis(), log_rtn_pct.kurtosis()]

stats.round(4)


# %%
import seaborn as sns 
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.tsa.api as smt

# %%
# Normal overlay parameters for SIMPLE returns (percent)
simple_rtn_pct = df["simple_rtn"] * 100

r_range = np.linspace(simple_rtn_pct.min(), simple_rtn_pct.max(), 1000)
mu = simple_rtn_pct.mean()
sigma = simple_rtn_pct.std()
norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)


# %%
# Histogram + Q–Q plot: SIMPLE returns (percent)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Histogram (percent)
sns.histplot(simple_rtn_pct, bins=50, stat="density", ax=ax[0])
ax[0].set_title(f"{TICKER}: Simple daily returns", fontsize=16)
ax[0].set_xlabel("Return (%)")
ax[0].plot(r_range, norm_pdf, "g", lw=2, label=f"Normal fit: N({mu:.2f}, {sigma**2:.2f})")
ax[0].legend(loc="upper left")

# Q–Q plot (use the SAME series as the histogram)
sm.qqplot(simple_rtn_pct.values, line="s", ax=ax[1])
ax[1].set_title("Q–Q plot (simple returns, %)", fontsize=16)

plt.tight_layout()
plt.show()


# %%
# Summary statistics + normality test: SIMPLE returns

r = df["simple_rtn"].dropna()
r_pct = r * 100
jb_test = scs.jarque_bera(r.values)

print("---------- Simple Returns (Daily) ----------")
print("Range of dates:", r.index.min().date(), "-", r.index.max().date())
print("Number of observations:", r.shape[0])
print(f"Mean (%): {r_pct.mean():.4f}")
print(f"Median (%): {r_pct.median():.4f}")
print(f"Min (%): {r_pct.min():.4f}")
print(f"Max (%): {r_pct.max():.4f}")
print(f"Std Dev (%): {r_pct.std():.4f}")
print(f"Skewness: {r_pct.skew():.4f}")
print(f"Excess kurtosis: {r_pct.kurtosis():.4f}")
print(f"Jarque–Bera: {jb_test[0]:.2f}  |  p-value: {jb_test[1]:.3e}")


# %%
# Histogram + Q–Q plot: LOG returns (percent)

log_rtn_pct = df["log_rtn"] * 100

# Normal overlay parameters for log returns (percent)
r_range2 = np.linspace(log_rtn_pct.min(), log_rtn_pct.max(), 1000)
mu2 = log_rtn_pct.mean()
sigma2 = log_rtn_pct.std()
norm_pdf2 = scs.norm.pdf(r_range2, loc=mu2, scale=sigma2)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Histogram (percent)
sns.histplot(log_rtn_pct, bins=50, stat="density", ax=ax[0])
ax[0].set_title(f"{TICKER}: Log daily returns", fontsize=16)
ax[0].set_xlabel("Return (%)")
ax[0].plot(r_range2, norm_pdf2, "g", lw=2, label=f"Normal fit: N({mu2:.2f}, {sigma2**2:.2f})")
ax[0].legend(loc="upper left")

# Q–Q plot
sm.qqplot(log_rtn_pct.values, line="s", ax=ax[1])
ax[1].set_title("Q–Q plot (log returns, %)", fontsize=16)

plt.tight_layout()
plt.show()


# %%
# Summary statistics + normality test: LOG returns

r = df["log_rtn"].dropna()
r_pct = r * 100
jb_test = scs.jarque_bera(r.values)

print("---------- Log Returns (Daily) ----------")
print("Range of dates:", r.index.min().date(), "-", r.index.max().date())
print("Number of observations:", r.shape[0])
print(f"Mean (%): {r_pct.mean():.4f}")
print(f"Median (%): {r_pct.median():.4f}")
print(f"Min (%): {r_pct.min():.4f}")
print(f"Max (%): {r_pct.max():.4f}")
print(f"Std Dev (%): {r_pct.std():.4f}")
print(f"Skewness: {r_pct.skew():.4f}")
print(f"Excess kurtosis: {r_pct.kurtosis():.4f}")
print(f"Jarque–Bera: {jb_test[0]:.2f}  |  p-value: {jb_test[1]:.3e}")

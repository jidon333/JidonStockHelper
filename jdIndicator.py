# indicators.py
import numpy as np
import pandas as pd

def true_range(df: pd.DataFrame):
    """
    Calculate the True Range (TR) for a given OHLCV DataFrame.

    TR = max( High - Low,
              abs(High - Previous Close),
              abs(Low  - Previous Close) )
    """
    high = df['High']
    low = df['Low']
    prev_close = df['Close'].shift(1)

    d1 = high - low
    d2 = np.abs(high - prev_close)
    d3 = np.abs(low - prev_close)
    
    tr = np.maximum(d1, d2)
    tr = np.maximum(tr, d3)

    return tr

def atr(df: pd.DataFrame, n=14):
    return true_range(df).rolling(n).mean()

def sma(series: pd.Series, n):
    return series.rolling(n).mean()

def ema(series: pd.Series, span):
    return series.ewm(span=span, adjust=False).mean()

def adr(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    Average Daily Range (%) for the past *n* days.
    ADR% = (rolling_mean(High / Low) -1) x 100
    """
    # DR% (Daily Range)
    high = df['High']
    low = df['Low']
    daily_range_percentages = high / low

    # ADR% (20-days)
    adr = daily_range_percentages.rolling(n).mean()
    adr = 100 * (adr - 1)

    return adr

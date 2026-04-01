import yfinance as yf
import pandas as pd
import datetime

def fetch_historical_prices(tickers, years=5, market=None):
    """
    Fetch historical daily prices from yfinance.
    
    Args:
        tickers (list): List of stock ticker symbols.
        years (int): Number of years of historical data to fetch.
        market (str): Default market suffix to append if none present.
        
    Returns:
        pd.DataFrame: Cleaned daily closing prices.
    """
    processed_tickers = []
    for t in tickers:
        # If the ticker already has a suffix (like .TO or .NE), use it as is
        if "." in t:
            processed_tickers.append(t)
        elif market:
            processed_tickers.append(f"{t}.{market}")
        else:
            processed_tickers.append(t)
            
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=years*365)
    
    # Fetch data
    data = yf.download(processed_tickers, start=start_date, end=end_date)
    
    # In newer yfinance, 'Adj Close' is often 'Close' if auto_adjust=True (default)
    # The columns are a MultiIndex (Price, Ticker)
    if 'Adj Close' in data.columns.levels[0]:
        data = data['Adj Close']
    else:
        data = data['Close']
    
    # Handle cases where some tickers fail or return NaN
    data = data.dropna(axis=0, how='all')
    data = data.ffill().bfill()
    
    return data

if __name__ == "__main__":
    # Test fetch with mixed tickers
    test_tickers = ["BNS.TO", "AMD.NE", "AAPL"]
    prices = fetch_historical_prices(test_tickers, years=2)
    print(prices.head())
    print(f"Data shape: {prices.shape}")

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_forex_data(symbol, period="1mo"):
    try:
        forex = yf.Ticker(f"{symbol}=X")
        hist = forex.history(period=period)
        if not hist.empty:
            hist['current_price'] = hist['Close'].iloc[-1]
        return hist
    except Exception as e:
        print(f"Error fetching forex data: {e}")
        return None

def test_forex_data():
    symbol = "EURUSD"
    data = get_forex_data(symbol)
    
    if data is not None and not data.empty:
        print(f"Successfully fetched data for {symbol}")
        print("\nLast 5 rows of data:")
        print(data.tail())
        print(f"\nCurrent price: {data['current_price'].iloc[-1]:.4f}")
    else:
        print(f"Failed to fetch data for {symbol}")

if __name__ == "__main__":
    test_forex_data()
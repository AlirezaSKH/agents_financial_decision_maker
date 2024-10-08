import logging
import sys
import os
from datetime import datetime, timedelta
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import numpy as np
from sqlalchemy import create_engine, Column, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from flask import Flask, jsonify
import re
from typing import Any
import json
import psycopg2
from urllib.parse import quote, unquote


# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')#'postgresql://postgres:Ali011111@localhost/financial_analysis'
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(String, primary_key=True)
    asset = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    intraday_plan = Column(JSON)
    short_term_plan = Column(JSON)
    medium_term_plan = Column(JSON)

def create_database():
    Base.metadata.create_all(bind=engine)

def save_to_database(asset, intraday_plan, short_term_plan, medium_term_plan):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    timestamp = datetime.now().isoformat()
    
    # Ensure plans are dictionaries, not None
    intraday_plan = intraday_plan or {}
    short_term_plan = short_term_plan or {}
    medium_term_plan = medium_term_plan or {}
    
    cur.execute("""
        INSERT INTO analysis_results (id, asset, timestamp, intraday_plan, short_term_plan, medium_term_plan)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        f"{asset}_{timestamp}",
        asset,
        timestamp,
        json.dumps(intraday_plan),
        json.dumps(short_term_plan),
        json.dumps(medium_term_plan)
    ))
    
    conn.commit()
    cur.close()
    conn.close()

    print("Saved to database:")
    print(f"Intraday Plan: {json.dumps(intraday_plan, indent=2)}")
    print(f"Short-term Plan: {json.dumps(short_term_plan, indent=2)}")
    print(f"Medium-term Plan: {json.dumps(medium_term_plan, indent=2)}")




# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(process)d - %(filename)s:%(funcName)s:%(lineno)d - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set API Keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

class SharedMemory:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

def get_finnhub_data(symbol):
    try:
        base_url = "https://finnhub.io/api/v1/quote"
        params = {
            "symbol": symbol,
            "token": FINNHUB_API_KEY
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        logger.error(f"Error fetching Finnhub data: {e}")
        return None

def get_finnhub_forex_data(symbol):
    try:
        base_url = "https://finnhub.io/api/v1/forex/candle"
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=30)).timestamp())
        
        # Convert symbol to OANDA format
        oanda_symbol = f"OANDA:{symbol.replace('/', '_')}"
        
        params = {
            "symbol": oanda_symbol,
            "resolution": "D",
            "from": start_time,
            "to": end_time,
            "token": FINNHUB_API_KEY
        }
        logger.info(f"Fetching Finnhub forex data for symbol: {oanda_symbol}")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['s'] == 'ok':
            df = pd.DataFrame({
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            }, index=pd.to_datetime(data['t'], unit='s'))
            
            # Round all price columns to 5 decimal places
            price_columns = ['Open', 'High', 'Low', 'Close']
            df[price_columns] = df[price_columns].round(5)
            
            df['current_price'] = df['Close'].iloc[-1]
            return df
        else:
            logger.error(f"Error fetching Finnhub forex data: {data.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        logger.error(f"Error fetching Finnhub forex data: {e}")
        return None

def get_yfinance_data(symbol, period="1mo"):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if not hist.empty:
            hist['current_price'] = hist['Close'].iloc[-1]
            return hist
        else:
            logger.warning(f"No data available from yfinance for {symbol}")
            return None
    except Exception as e:
        logger.error(f"Error fetching yfinance data: {e}")
        return None

def get_stock_data(symbol, period="1mo"):
    try:
        normalized_symbol = normalize_asset(symbol)
        logger.info(f"Fetching data for: {symbol} (normalized: {normalized_symbol})")
        
        if '/' in normalized_symbol:  # This is a forex pair or commodity
            return get_forex_or_commodity_data(normalized_symbol, period)
        elif '-' in symbol:  # This is a crypto
            return get_crypto_data(symbol, period)
        else:  # This is a stock
            return get_stock_specific_data(symbol, period)
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def get_forex_or_commodity_data(symbol, period):
    try:
        finnhub_data = get_finnhub_forex_data(symbol)
        if finnhub_data is not None and not finnhub_data.empty:
            logger.info(f"Successfully fetched Finnhub data for {symbol}")
            return finnhub_data
        
        logger.warning(f"Fallback to yfinance for forex/commodity data: {symbol}")
        fallback_symbol = symbol.replace('/', '-')
        data = get_yfinance_data(fallback_symbol, period)
        if data is not None and not data.empty:
            price_columns = ['Open', 'High', 'Low', 'Close', 'current_price']
            data[price_columns] = data[price_columns].round(5)
            return data
        
        raise ValueError(f"No data available for {symbol}")
    except Exception as e:
        logger.error(f"Error fetching forex/commodity data for {symbol}: {str(e)}")
        return None

def get_crypto_data(symbol, period):
    try:
        finnhub_data = get_finnhub_data(symbol)
        if finnhub_data:
            logger.info(f"Successfully fetched Finnhub data for crypto {symbol}")
            hist = yf.Ticker(symbol).history(period=period)
            hist['current_price'] = finnhub_data['c']
            price_columns = ['Open', 'High', 'Low', 'Close', 'current_price']
            hist[price_columns] = hist[price_columns].round(8)
            return hist
    except Exception as e:
        logger.warning(f"Failed to fetch Finnhub data for crypto {symbol}: {str(e)}")

    logger.warning(f"Fallback to yfinance for crypto data: {symbol}")
    data = get_yfinance_data(symbol, period)
    if data is not None and not data.empty and data['Close'].iloc[-1] != 0:
        price_columns = ['Open', 'High', 'Low', 'Close', 'current_price']
        data[price_columns] = data[price_columns].round(8)
        return data
    
    logger.warning(f"Fallback to CoinGecko API for crypto data: {symbol}")
    coin_id = symbol.split('-')[0].lower()
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30&interval=daily"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['Open'] = df['Close'].shift(1)
        df['High'] = df['Close']
        df['Low'] = df['Close']
        df['Volume'] = [price[1] for price in data['total_volumes']]
        df['current_price'] = df['Close'].iloc[-1]
        price_columns = ['Open', 'High', 'Low', 'Close', 'current_price']
        df[price_columns] = df[price_columns].round(8)
        return df
    else:
        raise ValueError(f"No data available for {symbol}")

def get_stock_specific_data(symbol, period):
    finnhub_data = get_finnhub_data(symbol)
    if finnhub_data:
        logger.info(f"Successfully fetched Finnhub data for stock {symbol}")
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        hist['current_price'] = finnhub_data['c']
        price_columns = ['Open', 'High', 'Low', 'Close', 'current_price']
        hist[price_columns] = hist[price_columns].round(2)
        return hist
    
    logger.warning(f"Fallback to yfinance for stock data: {symbol}")
    data = get_yfinance_data(symbol, period)
    if data is not None and not data.empty:
        price_columns = ['Open', 'High', 'Low', 'Close', 'current_price']
        data[price_columns] = data[price_columns].round(2)
        return data
    
    raise ValueError(f"No data available for {symbol}")



def get_finnhub_forex_data(symbol, period="1mo"):
    try:
        # Convert period to number of days
        days = {
            "1d": 1, "1w": 7, "1mo": 30, "3mo": 90,
            "6mo": 180, "1y": 365, "2y": 730, "5y": 1825
        }.get(period, 30)  # Default to 30 days if period is not recognized

        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=days)).timestamp())
        
        # Convert symbol to OANDA format if it's not already
        if '/' not in symbol:
            symbol = symbol.replace('-', '/')
        oanda_symbol = f"OANDA:{symbol.replace('/', '_')}"
        
        base_url = "https://finnhub.io/api/v1/forex/candle"
        params = {
            "symbol": oanda_symbol,
            "resolution": "D",
            "from": start_time,
            "to": end_time,
            "token": FINNHUB_API_KEY
        }
        logger.info(f"Fetching Finnhub forex data for symbol: {oanda_symbol}")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['s'] == 'ok':
            df = pd.DataFrame({
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            }, index=pd.to_datetime(data['t'], unit='s'))
            
            # Round all price columns to 5 decimal places
            price_columns = ['Open', 'High', 'Low', 'Close']
            df[price_columns] = df[price_columns].round(5)
            
            df['current_price'] = df['Close'].iloc[-1]
            return df
        else:
            logger.error(f"Error fetching Finnhub forex data: {data.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        logger.error(f"Error fetching Finnhub forex data for {symbol}: {str(e)}")
        return None




def get_news(symbol):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        url = f"https://newsapi.org/v2/everything?q={symbol}&from={start_date.date()}&to={end_date.date()}&sortBy=popularity&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return None

def get_economic_indicators():
    # This is a placeholder. In a real-world scenario, you'd fetch this data from a reliable financial API
    return {
        "interest_rate": 0.25,
        "unemployment_rate": 3.6,
        "cash_rate": 0.1,
        "pmi": 52.6,
        "ism": 55.3,
        "nfp": 250000
    }

def calculate_volume_profile(data, n_bins=10):
    """
    Calculate the Volume Profile for the given data.
    
    :param data: DataFrame with 'Close' and 'Volume' columns
    :param n_bins: Number of bins to divide the price range into
    :return: DataFrame with volume profile data
    """
    # Ensure required columns are present
    if 'Close' not in data.columns or 'Volume' not in data.columns:
        raise ValueError("DataFrame must contain 'Close' and 'Volume' columns")

    # Calculate the price range and bin size
    price_range = data['Close'].max() - data['Close'].min()
    if price_range == 0:
        raise ValueError("Price range is zero, cannot create bins")

    bin_size = price_range / n_bins
    
    # Define bins and labels
    bins = [data['Close'].min() + i * bin_size for i in range(n_bins + 1)]
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(n_bins)]
    
    # Bin the data and calculate volume profile
    data['price_bin'] = pd.cut(data['Close'], bins=bins, labels=labels, include_lowest=True)
    volume_profile = data.groupby('price_bin')['Volume'].sum().sort_values(ascending=False)
    
    # Log the result for debugging
    print(f"Volume Profile Calculated:\n{volume_profile.head()}")

    return volume_profile


def calculate_ichimoku_cloud(data):
    """
    Calculate the Ichimoku Cloud indicators.
    
    :param data: DataFrame with 'high', 'low', 'close' columns
    :return: A dictionary with Ichimoku Cloud components
    """
    # Ensure required columns are present
    if not all(col in data.columns for col in ['high', 'low', 'close']):
        raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns")
    
    # Calculate Ichimoku Cloud components
    conversion_line = (data['high'].rolling(window=9).max() + data['low'].rolling(window=9).min()) / 2
    base_line = (data['high'].rolling(window=26).max() + data['low'].rolling(window=26).min()) / 2
    leading_span_a = ((conversion_line + base_line) / 2).shift(26)
    leading_span_b = ((data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2).shift(26)
    lagging_span = data['close'].shift(-26)
    
    # Ensure the components are not empty or all NaN
    if any(comp.isnull().all() for comp in [conversion_line, base_line, leading_span_a, leading_span_b, lagging_span]):
        raise ValueError("One or more Ichimoku components could not be calculated")

    # Log the result for debugging
    print("Ichimoku Cloud components calculated successfully.")

    return {
        'conversion_line': conversion_line,
        'base_line': base_line,
        'leading_span_a': leading_span_a,
        'leading_span_b': leading_span_b,
        'lagging_span': lagging_span
    }


def calculate_price_bins(data, n_bins=10):
    """
    Calculate price bins for the given data.
    
    :param data: DataFrame with 'Close' column
    :param n_bins: Number of bins to divide the price range into
    :return: DataFrame with 'price_bin' column added
    """
    price_range = data['Close'].max() - data['Close'].min()
    bin_size = price_range / n_bins
    bins = [data['Close'].min() + i * bin_size for i in range(n_bins + 1)]
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(n_bins)]
    data['price_bin'] = pd.cut(data['Close'], bins=bins, labels=labels, include_lowest=True)
    return data





def get_enhanced_technical_analysis(data):
    """
    Calculate a wide range of technical indicators for the given data.
    
    :param data: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns
    :return: Dictionary containing calculated technical indicators
    """
    indicators = {}

    if data is None or data.empty:
        logger.error("Input data is None or empty")
        return {}

    # Check for required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        logger.error(f"Input data is missing required columns. Required: {required_columns}, Got: {data.columns.tolist()}")
        return {}

    try:
        # Calculate price bins before volume profile
        data = calculate_price_bins(data)

        # Simple Moving Averages
        indicators['SMA_10'] = ta.sma(data['Close'], length=10)
        indicators['SMA_20'] = ta.sma(data['Close'], length=20)
        indicators['SMA_50'] = ta.sma(data['Close'], length=50)
        
        # Exponential Moving Averages
        indicators['EMA_10'] = ta.ema(data['Close'], length=10)
        indicators['EMA_20'] = ta.ema(data['Close'], length=20)
        indicators['EMA_50'] = ta.ema(data['Close'], length=50)
        
        # Relative Strength Index
        indicators['RSI'] = ta.rsi(data['Close'])
        
        # Moving Average Convergence Divergence
        try:
            fast = data['Close'].ewm(span=12, adjust=False).mean()
            slow = data['Close'].ewm(span=26, adjust=False).mean()
            indicators['MACD'] = fast - slow
            indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
            indicators['MACD_Hist'] = indicators['MACD'] - indicators['MACD_Signal']
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            indicators['MACD'] = pd.Series([np.nan] * len(data))
            indicators['MACD_Signal'] = pd.Series([np.nan] * len(data))
            indicators['MACD_Hist'] = pd.Series([np.nan] * len(data))
        
        # Bollinger Bands
        try:
            bbands = ta.bbands(data['Close'])
            indicators['Upper_BB'] = bbands['BBU_20_2.0']
            indicators['Middle_BB'] = bbands['BBM_20_2.0']
            indicators['Lower_BB'] = bbands['BBL_20_2.0']
        except KeyError:
            # Fallback method for Bollinger Bands calculation
            period = 20
            std_dev = 2
            rolling_mean = data['Close'].rolling(window=period).mean()
            rolling_std = data['Close'].rolling(window=period).std()
            indicators['Upper_BB'] = rolling_mean + (rolling_std * std_dev)
            indicators['Middle_BB'] = rolling_mean
            indicators['Lower_BB'] = rolling_mean - (rolling_std * std_dev)
        
        # Stochastic Oscillator
        try:
            stoch = ta.stoch(data['High'], data['Low'], data['Close'])
            indicators['STOCH_K'] = stoch['STOCHk_14_3_3']
            indicators['STOCH_D'] = stoch['STOCHd_14_3_3']
        except KeyError:
            # Fallback method for Stochastic Oscillator calculation
            period = 14
            k = 3
            d = 3
            low_min = data['Low'].rolling(window=period).min()
            high_max = data['High'].rolling(window=period).max()
            indicators['STOCH_K'] = ((data['Close'] - low_min) / (high_max - low_min)) * 100
            indicators['STOCH_D'] = indicators['STOCH_K'].rolling(window=d).mean()
        
        # Average Directional Index
        try:
            adx = ta.adx(data['High'], data['Low'], data['Close'])
            indicators['ADX'] = adx['ADX_14']
        except KeyError:
            # We'll skip ADX if it's not available, as it's complex to calculate manually
            indicators['ADX'] = pd.Series([np.nan] * len(data))
        
        # Commodity Channel Index
        try:
            indicators['CCI'] = ta.cci(data['High'], data['Low'], data['Close'])
        except Exception:
            # Fallback method for CCI calculation
            period = 20
            tp = (data['High'] + data['Low'] + data['Close']) / 3
            sma_tp = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            indicators['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        # On-Balance Volume
        indicators['OBV'] = ta.obv(data['Close'], data['Volume'])
        
        # Volume Profile
        try:
            volume_profile = data.groupby('price_bin')['Volume'].sum().sort_values(ascending=False)
            indicators['Volume_Profile'] = volume_profile
        except Exception as e:
            logger.error(f"Error calculating Volume Profile: {str(e)}")
            indicators['Volume_Profile'] = pd.Series([np.nan] * len(data))
        
        # Ichimoku Cloud
        try:
            conversion_line, base_line, leading_span_a, leading_span_b, lagging_span = calculate_ichimoku_cloud(data)
            indicators['Ichimoku_Conversion_Line'] = conversion_line
            indicators['Ichimoku_Base_Line'] = base_line
            indicators['Ichimoku_Leading_Span_A'] = leading_span_a
            indicators['Ichimoku_Leading_Span_B'] = leading_span_b
        except Exception as e:
            logger.error(f"Error calculating Ichimoku Cloud: {str(e)}")
            indicators['Ichimoku_Conversion_Line'] = pd.Series([np.nan] * len(data))
            indicators['Ichimoku_Base_Line'] = pd.Series([np.nan] * len(data))
            indicators['Ichimoku_Leading_Span_A'] = pd.Series([np.nan] * len(data))
            indicators['Ichimoku_Leading_Span_B'] = pd.Series([np.nan] * len(data))
        
        # Fibonacci Retracement
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        indicators['Fib_23.6'] = high - (diff * 0.236)
        indicators['Fib_38.2'] = high - (diff * 0.382)
        indicators['Fib_50.0'] = high - (diff * 0.5)
        indicators['Fib_61.8'] = high - (diff * 0.618)
        
        # Pivot Points
        indicators['Pivot_Point'] = (data['High'].iloc[-1] + data['Low'].iloc[-1] + data['Close'].iloc[-1]) / 3
        indicators['R1'] = 2 * indicators['Pivot_Point'] - data['Low'].iloc[-1]
        indicators['S1'] = 2 * indicators['Pivot_Point'] - data['High'].iloc[-1]
        
        # Average True Range (ATR)
        indicators['ATR'] = ta.atr(data['High'], data['Low'], data['Close'])

    except Exception as e:
        logger.error(f"Error in get_enhanced_technical_analysis: {str(e)}", exc_info=True)
        return {}  # Return an empty dictionary if there's an error
    
    return indicators


def run_analysis(asset):
    normalized_asset = normalize_asset(asset)
    logger.info(f"Starting analysis for asset: {asset}")
    try:
        shared_memory = SharedMemory()
        logger.info(f"Fetching stock data for {asset}")
        current_data = get_stock_data(asset)
        
        if current_data is None or current_data.empty:
            logger.error(f"No data available for {asset}")
            raise ValueError(f"No data available for {asset}")
        

        logger.info(f"Stock data fetched successfully for {asset}. Shape: {current_data.shape}")
        logger.debug(f"First few rows of data:\n{current_data.head()}")

        current_price = current_data['current_price'].iloc[-1] if 'current_price' in current_data else current_data['Close'].iloc[-1]
        news = get_news(asset)
        economic_indicators = get_economic_indicators()
        
        # Calculate enhanced technical indicators
        try:
            technical_indicators = get_enhanced_technical_analysis(current_data)
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            technical_indicators = {}  # Use an empty dict if calculation fails
        
        # Store all data in shared memory
        shared_memory.set('current_data', current_data)
        shared_memory.set('news', news)
        shared_memory.set('economic_indicators', economic_indicators)
        shared_memory.set('technical_indicators', technical_indicators)

        # Define agents
        data_collection_agent = Agent(
            role='Data Collection Specialist',
            goal='Collect and preprocess financial data',
            backstory="You are an expert in collecting and preprocessing financial data. Your job is to gather relevant information about the given asset and prepare it for analysis.",
            verbose=True,
            allow_delegation=False,
            llm=ChatOpenAI(model_name="gpt-4", temperature=0)
        )

        analysis_agent = Agent(
            role='Financial Analyst',
            goal='Analyze financial data and provide insights',
            backstory="You are a seasoned financial analyst with expertise in interpreting market trends, financial indicators, and economic data. Your task is to analyze the preprocessed data and provide valuable insights.",
            verbose=True,
            allow_delegation=False,
            llm=ChatOpenAI(model_name="gpt-4", temperature=0)
        )

        decision_making_agent = Agent(
            role='Investment Decision Maker',
            goal='Make informed investment decisions based on technical analysis and market structure',
            backstory="You are a professional investment decision maker with years of experience in the financial markets. Your expertise lies in analyzing market structure across multiple timeframes to identify key support and resistance levels. You excel at making trading decisions that balance technical analysis with sound risk management principles. Your recommendations are known for their logical stop loss and target levels based on significant market levels rather than arbitrary percentages.",
            verbose=True,
            allow_delegation=False,
            llm=ChatOpenAI(model_name="gpt-4", temperature=0)
        )

        # Define tasks
        data_collection_task = Task(
            description=f"Collect and preprocess financial data for {asset}. Use the get_stock_data function to fetch the data and store it in the shared memory. The current price is ${current_price:.5f}. Also, collect the latest news and economic indicators.",
            agent=data_collection_agent,
            expected_output="A summary of the collected and preprocessed financial data for the given asset, including the current price, recent news, and relevant economic indicators."
        )

        analysis_task = Task(
        description=f"""Analyze the preprocessed financial data for {asset}. The current price is ${current_price:.5f}. 
        Provide insights on market trends, key indicators, and potential risks across multiple timeframes (intraday, short-term, medium-term, and long-term). 
        Consider the following in your analysis:
        1. Technical analysis of price trends and patterns, including the enhanced technical indicators provided:
        - Moving Averages (Simple and Exponential)
        - Relative Strength Index (RSI)
        - Moving Average Convergence Divergence (MACD)
        - Bollinger Bands
        - Stochastic Oscillator
        - Average Directional Index (ADX)
        - Commodity Channel Index (CCI)
        - On-Balance Volume (OBV)
        - Volume Profile (to identify significant price levels based on trading volume)
        - Ichimoku Cloud
        - Fibonacci Retracement levels
        - Pivot Points
        - Average True Range (ATR)
        2. Recent news and its potential impact on the asset in different timeframes
        3. Economic indicators: interest rates, unemployment rate, cash rate, PMI, ISM, and NFP
        4. Any other relevant factors that could influence the asset's price
        
        Pay special attention to the Volume Profile indicator and the newly added indicators, as they can help identify key support and resistance levels. Use this information to reinforce or question other technical indicators and price action analysis.
        
        For each timeframe (intraday, short-term, medium-term, and long-term), provide:
        - The overall trend
        - Key support and resistance levels
        - Potential pivot points or areas of interest
        - Any divergences or conflicting signals between different indicators""",
        agent=analysis_agent,
        expected_output="A comprehensive analysis of the financial data, including identified trends, key indicators, potential risks, and the impact of fundamental factors across multiple timeframes. Provide insights based on the enhanced technical indicators, with a focus on how the new indicators (Ichimoku Cloud, Fibonacci Retracement, Pivot Points, ATR) support or conflict with other indicators and price action. Clearly differentiate between intraday, short-term, medium-term, and long-term outlooks."
    )

        decision_making_task = Task(
        description=f"""Based on the analysis provided and the current price of ${current_price:.5f} for {asset}, make informed investment decisions for three distinct timeframes. 
        Provide clear recommendations along with the rationale behind each. 
        Your recommendations should include:

        1. Intraday Plan (for today):
        - Whether to buy, sell, or hold the asset
        - Entry point price (consider a range if appropriate, based on key support/resistance levels)
        - Stop loss price (based on the nearest strong support level for long trades, or resistance level for short trades also ATR is really important for stoploss)
        - Target price(s) (based on key resistance levels for long trades, or support levels for short trades)
        - Risk management considerations, including position sizing recommendations

        2. Short-term Plan (until next week):
        - Whether to buy, sell, or hold the asset
        - Entry point price (consider a range if appropriate, based on key support/resistance levels)
        - Stop loss price (based on the nearest strong support level for long trades, or resistance level for short trades also ATR is really important for stoploss)
        - Target price(s) (based on key resistance levels for long trades, or support levels for short trades)
        - Risk management considerations, including position sizing recommendations

        3. Medium-term Plan (until next month):
        - Whether to buy, sell, or hold the asset
        - Entry point price (consider a range if appropriate, based on key support/resistance levels)
        - Stop loss price (based on the nearest strong support level for long trades, or resistance level for short trades also ATR is really important for stoploss)
        - Target price(s) (based on key resistance levels for long trades, or support levels for short trades)
        - Risk management considerations, including position sizing recommendations

        For each plan:
        - Use technical analysis, including the enhanced indicators provided, to identify key support and resistance levels.
        - Set stop loss at a logical level based on the market structure, not just a fixed percentage also ATR is really important for stoploss.
        - Determine target prices based on significant resistance or support levels, depending on the trade direction.
        - Ensure that the potential reward is greater than the risk. The distance to the target should be larger than the distance to the stop loss.
        - Consider using multiple targets if appropriate, based on different resistance/support levels.
        - Take into account the asset's volatility when determining stop loss and target levels.
        - Explain the rationale behind each level (entry, stop loss, and targets) in terms of technical analysis and market structure.
        - Incorporate insights from the Volume Profile indicator to validate or adjust your support and resistance levels.

        Important guidelines:
        - Clearly state any relevant conditions that might invalidate each recommendation.
        - Remember, while maintaining a favorable risk-reward ratio is important, the specific levels should be based on the asset's price action and key technical levels, not arbitrary percentages.
        - Be aware that recommendations may differ between timeframes due to different market dynamics and analysis perspectives.""",
        agent=decision_making_agent,
        expected_output="Three clear investment recommendations (intraday, short-term, and medium-term) with detailed rationales based on the provided analysis. Include structured plans with entry points, stop losses, and target prices that reflect key support and resistance levels for each timeframe. Explain the technical basis for each level and ensure the overall plans adhere to sound risk management principles."
    )

        # Create Crew
        crew = Crew(
            agents=[data_collection_agent, analysis_agent, decision_making_agent],
            tasks=[data_collection_task, analysis_task, decision_making_task],
            verbose=2,
            process=Process.sequential
        )

        result = crew.kickoff()
        
        print("Raw crew output:")
        print(result)
        
        # Parse the result to extract plans
        plans = parse_result(result)
        
        print("Parsed plans:")
        print(json.dumps(plans, indent=2))
        
        # Save to database
        save_to_database(normalized_asset, plans['intraday'], plans['short_term'], plans['medium_term'])
        logger.info(f"Analysis completed for {asset} (normalized: {normalized_asset})")
        
        # Denormalize asset name in the result
        if isinstance(result, dict) and 'asset' in result:
            result['asset'] = denormalize_asset(result['asset'])
        
        return result

    except Exception as e:
        logger.error(f"Error in run_analysis: {str(e)}", exc_info=True)
        raise  # Re-raise the exception after logging




def parse_result(crew_result: Any) -> dict:
    if hasattr(crew_result, 'final_output'):
        result_text = crew_result.final_output
    else:
        result_text = str(crew_result)

    plans = {
        'intraday': {},
        'short_term': {},
        'medium_term': {}
    }
    
    # Regular expressions to extract information
    plan_regex = r"(\d+\.\s+(?:Intraday Plan|Short-term Plan \(until next week\)|Medium-term Plan \(until next month\))):(.+?)(?=\d+\.\s+(?:Intraday Plan|Short-term Plan|Medium-term Plan)|$)"
    decision_regex = r"Decision:\s*([^.\n]+)"
    entry_regex = r"Entry Point:\s*([^.\n]+)"
    stop_loss_regex = r"Stop Loss:\s*([^.\n]+)"
    target_regex = r"Target Price:\s*([^.\n]+)"
    risk_management_regex = r"Risk Management:\s*([^.\n]+(?:\n(?!\s*-)[^.\n]+)*)"
    
    # Find all plans in the result
    found_plans = re.findall(plan_regex, result_text, re.DOTALL | re.IGNORECASE)
    
    for plan_type, plan_content in found_plans:
        if "Intraday Plan" in plan_type:
            key = 'intraday'
        elif "Short-term Plan" in plan_type:
            key = 'short_term'
        elif "Medium-term Plan" in plan_type:
            key = 'medium_term'
        else:
            continue  # Skip if not a recognized plan type

        decision = re.search(decision_regex, plan_content)
        entry = re.search(entry_regex, plan_content)
        stop_loss = re.search(stop_loss_regex, plan_content)
        target = re.search(target_regex, plan_content)
        risk_management = re.search(risk_management_regex, plan_content, re.DOTALL)
        
        plans[key] = {
            'recommendation': decision.group(1).strip() if decision else None,
            'entry': entry.group(1).strip() if entry else None,
            'stop_loss': stop_loss.group(1).strip() if stop_loss else None,
            'target': target.group(1).strip() if target else None,
            'risk_management': risk_management.group(1).strip() if risk_management else None
        }
    
    return plans


def get_latest_analysis(asset):
    normalized_asset = normalize_asset(asset)
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT * FROM analysis_results WHERE asset = %s ORDER BY timestamp DESC LIMIT 1", (normalized_asset,))
        result = cur.fetchone()
        cur.close()
        conn.close()

        if result:
            # Check if each field is a string before attempting to parse it as JSON
            intraday_plan = json.loads(result[3]) if isinstance(result[3], str) else result[3]
            short_term_plan = json.loads(result[4]) if isinstance(result[4], str) else result[4]
            medium_term_plan = json.loads(result[5]) if isinstance(result[5], str) else result[5]

            return {
                'id': result[0],
                'asset': denormalize_asset(result[1]),
                'timestamp': result[2].isoformat(),
                'intraday_plan': intraday_plan,
                'short_term_plan': short_term_plan,
                'medium_term_plan': medium_term_plan
            }
        else:
            return None
    except Exception as e:
        app.logger.error(f"Error retrieving latest analysis for {asset} (normalized: {normalized_asset}): {str(e)}", exc_info=True)
        return None



def create_table_if_not_exists():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_results (
            id VARCHAR(255) PRIMARY KEY,
            asset VARCHAR(50),
            timestamp TIMESTAMP,
            intraday_plan JSON,
            short_term_plan JSON,
            medium_term_plan JSON
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Table 'analysis_results' created or already exists.")

def normalize_asset(asset):
    """Normalize asset string to internal format."""
    if '-' in asset:
        parts = asset.split('-')
        if len(parts) == 2:
            return f"{parts[0]}/{parts[1]}"
    return asset  # Return as-is if it doesn't match the expected format


def denormalize_asset(asset):
    """Convert internal asset format to API format."""
    return asset.replace('/', '-')



app = Flask(__name__)
create_table_if_not_exists()

@app.route('/api/analysis/<asset>', methods=['GET'])
def get_analysis(asset):
    normalized_asset = normalize_asset(asset)
    app.logger.info(f"Received GET request for asset: {asset} (normalized: {normalized_asset})")
    try:
        result = get_latest_analysis(normalized_asset)
        if result:
            # Denormalize the asset name in the result
            if isinstance(result, dict) and 'asset' in result:
                result['asset'] = denormalize_asset(result['asset'])
            
            app.logger.info(f"Analysis retrieved for asset: {asset}")
            return jsonify({'message': 'Analysis retrieved', 'result': result}), 200
        else:
            app.logger.warning(f"No analysis found for asset: {asset}")
            return jsonify({'error': 'No analysis found for this asset'}), 404
    except Exception as e:
        app.logger.error(f"Error retrieving analysis for asset {asset}: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/test/<asset>', methods=['GET'])
def get_test_analysis(asset):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT * FROM analysis_results WHERE asset = %s ORDER BY timestamp DESC LIMIT 1", (asset,))
        result = cur.fetchone()
        cur.close()
        conn.close()

        if result:
            return jsonify({
                'id': result[0],
                'asset': result[1],
                'timestamp': result[2].isoformat(),
                'intraday_plan': json.loads(result[3]) if isinstance(result[3], str) else result[3],
                'short_term_plan': json.loads(result[4]) if isinstance(result[4], str) else result[4],
                'medium_term_plan': json.loads(result[5]) if isinstance(result[5], str) else result[5]
            }), 200
        else:
            return jsonify({'error': 'No analysis found for this asset'}), 404
    except Exception as e:
        app.logger.error(f"Error retrieving analysis for {asset}: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500



@app.route('/api/run-analysis/<asset>', methods=['POST'])
def run_new_analysis(asset):
    normalized_asset = normalize_asset(asset)
    app.logger.info(f"Received request for asset: {asset} (normalized: {normalized_asset})")
    try:
        result = run_analysis(normalized_asset)
        
        # Check if result is None or empty
        if not result:
            app.logger.warning(f"No analysis result for asset: {normalized_asset}")
            return jsonify({'message': 'No analysis result', 'asset': asset}), 404

        # Assuming result is a dictionary, let's denormalize the asset name in the result
        if isinstance(result, dict) and 'asset' in result:
            result['asset'] = denormalize_asset(result['asset'])

        app.logger.info(f"Analysis completed for asset: {asset}")
        return jsonify({'message': 'Analysis completed', 'result': result}), 200
    except Exception as e:
        app.logger.error(f"Error in run_analysis for asset {asset}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500




if __name__ == "__main__":
    create_database()
    
    if len(sys.argv) != 2:
        print("Usage: python financial_decision_ai.py <asset_symbol>")
        sys.exit(1)

    asset = sys.argv[1]
    
    try:
        logger.info(f"Starting analysis for {asset}")
        result = run_analysis(asset)
        print(result)
        
        # Start the Flask app
        app.run(debug=True)
    except Exception as e:
        logger.error(f"An error occurred on {datetime.now().date()}: {str(e)}")
        print(f"An error occurred: {str(e)}")
        logger.error("Error details:", exc_info=True)
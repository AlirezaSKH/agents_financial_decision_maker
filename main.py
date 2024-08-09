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
        if '/' in symbol:  # This is a forex pair or commodity
            logger.info(f"Fetching forex/commodity data for: {symbol}")
            finnhub_data = get_finnhub_forex_data(symbol)
            if finnhub_data is not None and not finnhub_data.empty:
                return finnhub_data
            else:
                logger.warning(f"Fallback to yfinance for forex/commodity data: {symbol}")
                fallback_symbol = symbol.replace('/', '-')
                data = get_yfinance_data(fallback_symbol)
                if data is not None and not data.empty:
                    price_columns = ['Open', 'High', 'Low', 'Close', 'current_price']
                    data[price_columns] = data[price_columns].round(5)
                    return data
                else:
                    raise ValueError(f"No data available for {symbol}")
        elif '-' in symbol:  # This is a crypto
            logger.info(f"Fetching crypto data for: {symbol}")
            finnhub_data = get_finnhub_data(symbol)
            if finnhub_data:
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period)
                hist['current_price'] = finnhub_data['c']
                price_columns = ['Open', 'High', 'Low', 'Close', 'current_price']
                hist[price_columns] = hist[price_columns].round(8)
                return hist
            else:
                logger.warning(f"Fallback to yfinance for crypto data: {symbol}")
                data = get_yfinance_data(symbol)
                if data is not None and not data.empty:
                    price_columns = ['Open', 'High', 'Low', 'Close', 'current_price']
                    data[price_columns] = data[price_columns].round(8)
                    return data
                else:
                    raise ValueError(f"No data available for {symbol}")
        else:  # This is a stock
            logger.info(f"Fetching stock data for: {symbol}")
            finnhub_data = get_finnhub_data(symbol)
            if finnhub_data:
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period)
                hist['current_price'] = finnhub_data['c']
                price_columns = ['Open', 'High', 'Low', 'Close', 'current_price']
                hist[price_columns] = hist[price_columns].round(2)
                return hist
            else:
                logger.warning(f"Fallback to yfinance for stock data: {symbol}")
                data = get_yfinance_data(symbol)
                if data is not None and not data.empty:
                    price_columns = ['Open', 'High', 'Low', 'Close', 'current_price']
                    data[price_columns] = data[price_columns].round(2)
                    return data
                else:
                    raise ValueError(f"No data available for {symbol}")
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
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
    price_range = data['Close'].max() - data['Close'].min()
    bin_size = price_range / n_bins
    
    bins = [data['Close'].min() + i * bin_size for i in range(n_bins + 1)]
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(n_bins)]
    
    data['price_bin'] = pd.cut(data['Close'], bins=bins, labels=labels, include_lowest=True)
    volume_profile = data.groupby('price_bin')['Volume'].sum().sort_values(ascending=False)
    
    return volume_profile

def get_enhanced_technical_analysis(data):
    """
    Calculate a wide range of technical indicators for the given data.
    
    :param data: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns
    :return: Dictionary containing calculated technical indicators
    """
    indicators = {}
    
    # Simple Moving Averages
    indicators['SMA_10'] = data.ta.sma(length=10)
    indicators['SMA_20'] = data.ta.sma(length=20)
    indicators['SMA_50'] = data.ta.sma(length=50)
    
    # Exponential Moving Averages
    indicators['EMA_10'] = data.ta.ema(length=10)
    indicators['EMA_20'] = data.ta.ema(length=20)
    indicators['EMA_50'] = data.ta.ema(length=50)
    
    # Relative Strength Index
    indicators['RSI'] = data.ta.rsi()
    
    # Moving Average Convergence Divergence
    try:
        macd = data.ta.macd()
        indicators['MACD'] = macd['MACD_12_26_9']
        indicators['MACD_Signal'] = macd['MACDs_12_26_9']
        indicators['MACD_Hist'] = macd['MACDh_12_26_9']
    except KeyError:
        # Fallback method for MACD calculation
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        indicators['MACD'] = exp1 - exp2
        indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
        indicators['MACD_Hist'] = indicators['MACD'] - indicators['MACD_Signal']
    
    # Bollinger Bands
    try:
        bbands = data.ta.bbands()
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
        stoch = data.ta.stoch()
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
        indicators['ADX'] = data.ta.adx()['ADX_14']
    except KeyError:
        # We'll skip ADX if it's not available, as it's complex to calculate manually
        indicators['ADX'] = np.nan
    
    # Commodity Channel Index
    try:
        indicators['CCI'] = data.ta.cci()
    except:
        # Fallback method for CCI calculation
        period = 20
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        indicators['CCI'] = (tp - sma_tp) / (0.015 * mad)
    
    # On-Balance Volume
    indicators['OBV'] = data.ta.obv()
    
    # Volume Profile
    indicators['Volume_Profile'] = calculate_volume_profile(data)
    
    return indicators

def run_analysis(asset):
    shared_memory = SharedMemory()
    current_data = get_stock_data(asset)
    
    if current_data is None or current_data.empty:
        raise ValueError(f"No data available for {asset}")
    
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
    2. Recent news and its potential impact on the asset in different timeframes
    3. Economic indicators: interest rates, unemployment rate, cash rate, PMI, ISM, and NFP
    4. Any other relevant factors that could influence the asset's price
    
    Pay special attention to the Volume Profile indicator, as it can help identify key support and resistance levels based on historical trading volume. Use this information to reinforce or question other technical indicators and price action analysis.
    
    For each timeframe (intraday, short-term, medium-term, and long-term), provide:
    - The overall trend
    - Key support and resistance levels
    - Potential pivot points or areas of interest
    - Any divergences or conflicting signals between different indicators""",
    agent=analysis_agent,
    expected_output="A comprehensive analysis of the financial data, including identified trends, key indicators, potential risks, and the impact of fundamental factors across multiple timeframes. Provide insights based on the enhanced technical indicators, with a focus on how the Volume Profile supports or conflicts with other indicators and price action. Clearly differentiate between intraday, short-term, medium-term, and long-term outlooks."
)

    decision_making_task = Task(
    description=f"""Based on the analysis provided and the current price of ${current_price:.5f} for {asset}, make an informed investment decision. 
    Provide a clear recommendation along with the rationale behind it. 
    Your recommendation should include:
    1. Whether to buy, sell, or hold the asset
    2. The specific timeframe for this recommendation (e.g., intraday, short-term (1-5 days), medium-term (1-4 weeks), or long-term (1-6 months))
    3. Support and resistance prices for multiple timeframes (e.g., 1 hour, 4 hour, daily)
    4. A structured plan including:
       - Entry point price (consider a range if appropriate, based on key support/resistance levels)
       - Stop loss price (based on the nearest strong support level for long trades, or resistance level for short trades)
       - Target price(s) (based on key resistance levels for long trades, or support levels for short trades)
    5. Risk management considerations, including position sizing recommendations
    6. Any relevant conditions that might invalidate this recommendation
    
    Important guidelines:
    - Clearly state the expected duration or timeframe for which this recommendation is valid.
    - Use technical analysis, including the enhanced indicators provided, to identify key support and resistance levels across multiple timeframes.
    - Set stop loss at a logical level based on the market structure, not just a fixed percentage.
    - Determine target prices based on significant resistance or support levels, depending on the trade direction.
    - Ensure that the potential reward is greater than the risk. The distance to the target should be larger than the distance to the stop loss.
    - Consider using multiple targets if appropriate, based on different resistance/support levels.
    - Take into account the asset's volatility when determining stop loss and target levels.
    - Explain the rationale behind each level (entry, stop loss, and targets) in terms of technical analysis and market structure.
    - Incorporate insights from the Volume Profile indicator to validate or adjust your support and resistance levels.

    Remember, while maintaining a favorable risk-reward ratio is important, the specific levels should be based on the asset's price action and key technical levels, not arbitrary percentages.""",
    agent=decision_making_agent,
    expected_output="A clear investment recommendation with a detailed rationale based on the provided analysis. Include the specific timeframe for the recommendation, a structured plan with entry point, stop loss, and target prices that reflect key support and resistance levels across multiple timeframes. Explain the technical basis for each level and ensure the overall plan adheres to sound risk management principles. Clearly state how long this recommendation is expected to be valid."
)

    # Create Crew
    crew = Crew(
        agents=[data_collection_agent, analysis_agent, decision_making_agent],
        tasks=[data_collection_task, analysis_task, decision_making_task],
        verbose=2,
        process=Process.sequential
    )

    result = crew.kickoff()
    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python financial_decision_ai.py <asset_symbol>")
        sys.exit(1)

    asset = sys.argv[1]
    
    try:
        logger.info(f"Starting analysis for {asset}")
        result = run_analysis(asset)
        print(result)
    except Exception as e:
        logger.error(f"An error occurred on {datetime.now().date()}: {str(e)}")
        print(f"An error occurred: {str(e)}")
        logger.error("Error details:", exc_info=True)
#!/usr/bin/env python3
"""
Simple Trading Strategy - Moving Average Crossover
A basic implementation for Indian stock market with Telegram notifications
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTradingStrategy:
    def __init__(self, telegram_bot_token=None, telegram_chat_id=None):
        """
        Initialize the trading strategy
        
        Args:
            telegram_bot_token (str): Telegram bot token for notifications
            telegram_chat_id (str): Telegram chat ID to send messages
        """
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.positions = {}  # Track current positions
        self.signals_history = []  # Store signal history
        
        # Strategy parameters
        self.short_window = 20  # Short-term moving average (20 days)
        self.long_window = 50   # Long-term moving average (50 days)
        
        # Indian stock symbols (NSE format for yfinance)
        self.watchlist = [
            'RELIANCE.NS',  # Reliance Industries
            'TCS.NS',       # Tata Consultancy Services
            'HDFCBANK.NS',  # HDFC Bank
            'INFY.NS',      # Infosys
            'HINDUNILVR.NS', # Hindustan Unilever
            'ICICIBANK.NS', # ICICI Bank
            'KOTAKBANK.NS', # Kotak Mahindra Bank
            'BHARTIARTL.NS', # Bharti Airtel
            'ITC.NS',       # ITC Limited
            'SBIN.NS'       # State Bank of India
        ]
    
    def send_telegram_message(self, message):
        """
        Send message to Telegram
        
        Args:
            message (str): Message to send
        """
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.info(f"Telegram not configured. Message: {message}")
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
            else:
                logger.error(f"Failed to send Telegram message: {response.text}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    def get_stock_data(self, symbol, period='3mo'):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE.NS')
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            pandas.DataFrame: Stock data with OHLCV
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_moving_averages(self, data):
        """
        Calculate short and long-term moving averages
        
        Args:
            data (pandas.DataFrame): Stock price data
        
        Returns:
            pandas.DataFrame: Data with moving averages added
        """
        data[f'MA_{self.short_window}'] = data['Close'].rolling(window=self.short_window).mean()
        data[f'MA_{self.long_window}'] = data['Close'].rolling(window=self.long_window).mean()
        return data
    
    def generate_signals(self, data):
        """
        Generate buy/sell signals based on moving average crossover
        
        Args:
            data (pandas.DataFrame): Stock data with moving averages
        
        Returns:
            pandas.DataFrame: Data with signals added
        """
        # Initialize signal column
        data['Signal'] = 0
        data['Position'] = 0
        
        # Generate signals where short MA crosses above/below long MA
        short_ma = data[f'MA_{self.short_window}']
        long_ma = data[f'MA_{self.long_window}']
        
        # Buy signal: short MA crosses above long MA
        data.loc[short_ma > long_ma, 'Signal'] = 1
        
        # Sell signal: short MA crosses below long MA
        data.loc[short_ma < long_ma, 'Signal'] = -1
        
        # Calculate position changes (actual buy/sell points)
        data['Position'] = data['Signal'].diff()
        
        return data
    
    def analyze_stock(self, symbol):
        """
        Analyze a single stock and generate signals
        
        Args:
            symbol (str): Stock symbol to analyze
        
        Returns:
            dict: Analysis results with current signal
        """
        logger.info(f"Analyzing {symbol}")
        
        # Get stock data
        data = self.get_stock_data(symbol)
        if data is None or len(data) < self.long_window:
            logger.warning(f"Insufficient data for {symbol}")
            return None
        
        # Calculate moving averages and signals
        data = self.calculate_moving_averages(data)
        data = self.generate_signals(data)
        
        # Get latest values
        latest = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else latest
        
        current_price = latest['Close']
        short_ma = latest[f'MA_{self.short_window}']
        long_ma = latest[f'MA_{self.long_window}']
        current_signal = latest['Signal']
        position_change = latest['Position']
        
        # Determine signal type
        signal_type = None
        if position_change == 2:  # Changed from -1 to 1
            signal_type = 'BUY'
        elif position_change == -2:  # Changed from 1 to -1
            signal_type = 'SELL'
        
        result = {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'short_ma': round(short_ma, 2),
            'long_ma': round(long_ma, 2),
            'current_signal': current_signal,
            'signal_type': signal_type,
            'timestamp': datetime.now(),
            'trend': 'BULLISH' if short_ma > long_ma else 'BEARISH'
        }
        
        return result
    
    def format_signal_message(self, analysis):
        """
        Format analysis result into a readable message
        
        Args:
            analysis (dict): Analysis result
        
        Returns:
            str: Formatted message
        """
        symbol_clean = analysis['symbol'].replace('.NS', '')
        
        if analysis['signal_type']:
            emoji = "🟢" if analysis['signal_type'] == 'BUY' else "🔴"
            message = f"""
{emoji} <b>{analysis['signal_type']} SIGNAL</b> {emoji}

📈 <b>Stock:</b> {symbol_clean}
💰 <b>Price:</b> ₹{analysis['current_price']}
📊 <b>Trend:</b> {analysis['trend']}

📉 <b>MA20:</b> ₹{analysis['short_ma']}
📈 <b>MA50:</b> ₹{analysis['long_ma']}

⏰ <b>Time:</b> {analysis['timestamp'].strftime('%d-%m-%Y %H:%M:%S')}

<i>⚠️ This is not financial advice. Please do your own research.</i>
            """
        else:
            message = f"""
📊 <b>Market Update</b>

📈 <b>Stock:</b> {symbol_clean}
💰 <b>Price:</b> ₹{analysis['current_price']}
📊 <b>Trend:</b> {analysis['trend']}

📉 <b>MA20:</b> ₹{analysis['short_ma']}
📈 <b>MA50:</b> ₹{analysis['long_ma']}

⏰ <b>Time:</b> {analysis['timestamp'].strftime('%d-%m-%Y %H:%M:%S')}
            """
        
        return message.strip()
    
    def run_analysis(self):
        """
        Run analysis on all stocks in watchlist
        """
        logger.info("Starting market analysis...")
        
        signals_found = []
        
        for symbol in self.watchlist:
            try:
                analysis = self.analyze_stock(symbol)
                if analysis:
                    # Store in history
                    self.signals_history.append(analysis)
                    
                    # Check for new signals
                    if analysis['signal_type']:
                        signals_found.append(analysis)
                        
                        # Send Telegram notification
                        message = self.format_signal_message(analysis)
                        self.send_telegram_message(message)
                        
                        logger.info(f"Signal generated for {symbol}: {analysis['signal_type']}")
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        if not signals_found:
            logger.info("No new signals generated")
        else:
            logger.info(f"Generated {len(signals_found)} signals")
        
        return signals_found
    
    def get_market_summary(self):
        """
        Get a summary of current market conditions
        
        Returns:
            str: Market summary message
        """
        logger.info("Generating market summary...")
        
        bullish_count = 0
        bearish_count = 0
        total_analyzed = 0
        
        for symbol in self.watchlist[:5]:  # Analyze top 5 for summary
            try:
                analysis = self.analyze_stock(symbol)
                if analysis:
                    total_analyzed += 1
                    if analysis['trend'] == 'BULLISH':
                        bullish_count += 1
                    else:
                        bearish_count += 1
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in summary for {symbol}: {e}")
        
        if total_analyzed == 0:
            return "Unable to generate market summary at this time."
        
        bullish_percentage = (bullish_count / total_analyzed) * 100
        
        market_sentiment = "BULLISH" if bullish_percentage > 60 else "BEARISH" if bullish_percentage < 40 else "NEUTRAL"
        
        summary = f"""
📊 <b>Market Summary</b>

🎯 <b>Overall Sentiment:</b> {market_sentiment}
📈 <b>Bullish Stocks:</b> {bullish_count}/{total_analyzed} ({bullish_percentage:.1f}%)
📉 <b>Bearish Stocks:</b> {bearish_count}/{total_analyzed} ({100-bullish_percentage:.1f}%)

⏰ <b>Updated:</b> {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

<i>Based on MA20/MA50 crossover analysis</i>
        """
        
        return summary.strip()

def main():
    """
    Main function to run the trading strategy
    """
    # Configuration
    TELEGRAM_BOT_TOKEN = None  # Add your Telegram bot token here
    TELEGRAM_CHAT_ID = None    # Add your Telegram chat ID here
    
    # Initialize strategy
    strategy = SimpleTradingStrategy(
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID
    )
    
    print("🚀 Simple Trading Strategy Started")
    print("=" * 50)
    
    try:
        # Run initial analysis
        signals = strategy.run_analysis()
        
        print(f"\n📊 Analysis Complete!")
        print(f"📈 Signals Generated: {len(signals)}")
        
        # Generate and send market summary
        summary = strategy.get_market_summary()
        print(f"\n{summary}")
        
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            strategy.send_telegram_message(summary)
        
        print("\n✅ Strategy execution completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Strategy stopped by user")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Enhanced Trading Strategy with Advanced Technical Indicators
Includes Bollinger Bands, Multiple EMAs, Volume Analysis, Target Price, and Stop Loss
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

class EnhancedTradingStrategy:
    def __init__(self, telegram_bot_token=None, telegram_chat_id=None):
        """
        Initialize the enhanced trading strategy
        
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
        self.ema_periods = [9, 18, 27]  # EMA periods
        self.bb_period = 20  # Bollinger Bands period
        self.bb_std = 2  # Bollinger Bands standard deviation
        self.volume_threshold = 1.5  # Volume threshold for high volume detection
        
        # Risk management parameters
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.target_profit_pct = 0.10  # 10% target profit
        
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
    
    def get_stock_data(self, symbol, period='6mo'):
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
    
    def calculate_ema(self, data, period):
        """
        Calculate Exponential Moving Average
        
        Args:
            data (pandas.Series): Price data
            period (int): EMA period
        
        Returns:
            pandas.Series: EMA values
        """
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """
        Calculate Bollinger Bands
        
        Args:
            data (pandas.Series): Price data
            period (int): Moving average period
            std_dev (float): Standard deviation multiplier
        
        Returns:
            tuple: (upper_band, middle_band, lower_band)
        """
        middle_band = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def calculate_volume_indicator(self, volume_data, period=20):
        """
        Calculate volume indicators
        
        Args:
            volume_data (pandas.Series): Volume data
            period (int): Period for average volume calculation
        
        Returns:
            dict: Volume indicators
        """
        avg_volume = volume_data.rolling(window=period).mean()
        current_volume = volume_data.iloc[-1]
        avg_volume_current = avg_volume.iloc[-1]
        
        volume_ratio = current_volume / avg_volume_current if avg_volume_current > 0 else 1
        is_high_volume = volume_ratio > self.volume_threshold
        
        return {
            'current_volume': current_volume,
            'avg_volume': avg_volume_current,
            'volume_ratio': volume_ratio,
            'is_high_volume': is_high_volume
        }
    
    def calculate_target_and_stoploss(self, current_price, signal_type):
        """
        Calculate target price and stop loss
        
        Args:
            current_price (float): Current stock price
            signal_type (str): 'BUY' or 'SELL'
        
        Returns:
            dict: Target and stop loss prices
        """
        if signal_type == 'BUY':
            target_price = current_price * (1 + self.target_profit_pct)
            stop_loss = current_price * (1 - self.stop_loss_pct)
        elif signal_type == 'SELL':
            target_price = current_price * (1 - self.target_profit_pct)
            stop_loss = current_price * (1 + self.stop_loss_pct)
        else:
            target_price = None
            stop_loss = None
        
        return {
            'target_price': target_price,
            'stop_loss': stop_loss
        }
    
    def calculate_technical_indicators(self, data):
        """
        Calculate all technical indicators
        
        Args:
            data (pandas.DataFrame): Stock price data
        
        Returns:
            pandas.DataFrame: Data with all indicators added
        """
        # Moving averages
        data[f'MA_{self.short_window}'] = data['Close'].rolling(window=self.short_window).mean()
        data[f'MA_{self.long_window}'] = data['Close'].rolling(window=self.long_window).mean()
        
        # EMAs
        for period in self.ema_periods:
            data[f'EMA_{period}'] = self.calculate_ema(data['Close'], period)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(
            data['Close'], self.bb_period, self.bb_std
        )
        data['BB_Upper'] = bb_upper
        data['BB_Middle'] = bb_middle
        data['BB_Lower'] = bb_lower
        
        # Bollinger Band position
        data['BB_Position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        return data
    
    def generate_enhanced_signals(self, data):
        """
        Generate enhanced buy/sell signals using multiple indicators with confidence calculation
        
        Args:
            data (pandas.DataFrame): Stock data with indicators
        
        Returns:
            pandas.DataFrame: Data with enhanced signals and confidence
        """
        # Initialize signal columns
        data['Signal'] = 0
        data['Position'] = 0
        data['Signal_Strength'] = 0
        data['Confidence'] = 0
        
        # Get latest values
        latest = data.iloc[-1]
        
        # Initialize confidence components
        confidence_factors = []
        
        # 1. Moving Average signals (20% weight)
        ma_signal = 0
        ma_confidence = 0
        if latest[f'MA_{self.short_window}'] > latest[f'MA_{self.long_window}']:
            ma_signal = 1
            ma_strength = (latest[f'MA_{self.short_window}'] - latest[f'MA_{self.long_window}']) / latest[f'MA_{self.long_window}']
            ma_confidence = min(20, abs(ma_strength) * 1000)  # Scale to 0-20
        elif latest[f'MA_{self.short_window}'] < latest[f'MA_{self.long_window}']:
            ma_signal = -1
            ma_strength = (latest[f'MA_{self.long_window}'] - latest[f'MA_{self.short_window}']) / latest[f'MA_{self.long_window}']
            ma_confidence = min(20, abs(ma_strength) * 1000)  # Scale to 0-20
        confidence_factors.append(ma_confidence)
        
        # 2. EMA signals (30% weight)
        ema_signal = 0
        ema_confidence = 0
        ema_9 = latest['EMA_9']
        ema_18 = latest['EMA_18']
        ema_27 = latest['EMA_27']
        
        if ema_9 > ema_18 > ema_27:
            ema_signal = 1  # Bullish EMA alignment
            # Calculate strength based on separation
            sep1 = (ema_9 - ema_18) / ema_18
            sep2 = (ema_18 - ema_27) / ema_27
            ema_confidence = min(30, (sep1 + sep2) * 1500)  # Scale to 0-30
        elif ema_9 < ema_18 < ema_27:
            ema_signal = -1  # Bearish EMA alignment
            sep1 = (ema_18 - ema_9) / ema_9
            sep2 = (ema_27 - ema_18) / ema_18
            ema_confidence = min(30, (sep1 + sep2) * 1500)  # Scale to 0-30
        confidence_factors.append(ema_confidence)
        
        # 3. Bollinger Band signals (25% weight)
        bb_signal = 0
        bb_confidence = 0
        bb_position = latest['BB_Position']
        
        if bb_position < 0.2:  # Near lower band
            bb_signal = 1  # Potential buy
            bb_confidence = (0.2 - bb_position) * 125  # Scale to 0-25
        elif bb_position > 0.8:  # Near upper band
            bb_signal = -1  # Potential sell
            bb_confidence = (bb_position - 0.8) * 125  # Scale to 0-25
        elif 0.4 <= bb_position <= 0.6:  # Middle zone - neutral
            bb_confidence = 5  # Small confidence for neutral
        confidence_factors.append(bb_confidence)
        
        # 4. Volume confirmation (25% weight)
        volume_data = data['Volume']
        current_volume = volume_data.iloc[-1]
        avg_volume = volume_data.rolling(window=20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        volume_confidence = 0
        if volume_ratio > 1.5:  # High volume confirmation
            volume_confidence = min(25, (volume_ratio - 1) * 12.5)  # Scale to 0-25
        elif volume_ratio > 1.2:  # Moderate volume
            volume_confidence = min(15, (volume_ratio - 1) * 25)  # Scale to 0-15
        else:  # Low volume - reduces confidence
            volume_confidence = max(0, volume_ratio * 10)  # Scale to 0-10
        confidence_factors.append(volume_confidence)
        
        # Calculate overall confidence (0-100%) with proper normalization
        # Each factor has a maximum weight, total should not exceed 100%
        ma_weight = 0.20  # 20%
        ema_weight = 0.30  # 30%
        bb_weight = 0.25   # 25%
        volume_weight = 0.25  # 25%
        
        # Normalize each factor to its maximum weight
        normalized_ma = min(confidence_factors[0], 20) * ma_weight / 0.20
        normalized_ema = min(confidence_factors[1], 30) * ema_weight / 0.30
        normalized_bb = min(confidence_factors[2], 25) * bb_weight / 0.25
        normalized_volume = min(confidence_factors[3], 25) * volume_weight / 0.25
        
        # Calculate total confidence (guaranteed to be 0-100%)
        total_confidence = min(100.0, normalized_ma + normalized_ema + normalized_bb + normalized_volume)
        
        # Combine signals
        signal_strength = ma_signal + ema_signal + bb_signal
        
        # Generate final signal with confidence threshold
        if signal_strength >= 2 and total_confidence >= 50:
            final_signal = 1  # Strong buy
        elif signal_strength <= -2 and total_confidence >= 50:
            final_signal = -1  # Strong sell
        elif abs(signal_strength) >= 1 and total_confidence >= 30:
            final_signal = signal_strength  # Moderate signal
        else:
            final_signal = 0  # Hold
        
        data.iloc[-1, data.columns.get_loc('Signal')] = final_signal
        data.iloc[-1, data.columns.get_loc('Signal_Strength')] = signal_strength
        data.iloc[-1, data.columns.get_loc('Confidence')] = total_confidence
        
        # Calculate position changes
        if len(data) > 1:
            prev_signal = data.iloc[-2]['Signal'] if 'Signal' in data.columns else 0
            if final_signal != prev_signal:
                data.iloc[-1, data.columns.get_loc('Position')] = final_signal - prev_signal
        
        return data
    
    def analyze_stock(self, symbol):
        """
        Analyze a single stock with enhanced indicators
        
        Args:
            symbol (str): Stock symbol to analyze
        
        Returns:
            dict: Enhanced analysis results
        """
        logger.info(f"Analyzing {symbol}")
        
        # Get stock data
        data = self.get_stock_data(symbol)
        if data is None or len(data) < max(self.long_window, max(self.ema_periods), self.bb_period):
            logger.warning(f"Insufficient data for {symbol}")
            return None
        
        # Calculate all technical indicators
        data = self.calculate_technical_indicators(data)
        data = self.generate_enhanced_signals(data)
        
        # Get latest values
        latest = data.iloc[-1]
        current_price = latest['Close']
        
        # Volume analysis
        volume_info = self.calculate_volume_indicator(data['Volume'])
        
        # Determine signal type
        signal_type = None
        if latest['Position'] > 0:
            signal_type = 'BUY'
        elif latest['Position'] < 0:
            signal_type = 'SELL'
        
        # Calculate target and stop loss
        target_stoploss = self.calculate_target_and_stoploss(current_price, signal_type)
        
        # Determine trend based on EMA alignment
        ema_9 = latest['EMA_9']
        ema_18 = latest['EMA_18']
        ema_27 = latest['EMA_27']
        
        if ema_9 > ema_18 > ema_27:
            trend = 'STRONG_BULLISH'
        elif ema_9 > ema_18 or ema_18 > ema_27:
            trend = 'BULLISH'
        elif ema_9 < ema_18 < ema_27:
            trend = 'STRONG_BEARISH'
        elif ema_9 < ema_18 or ema_18 < ema_27:
            trend = 'BEARISH'
        else:
            trend = 'NEUTRAL'
        
        # Convert numpy types to Python native types for JSON serialization
        result = {
            'symbol': symbol,
            'current_price': float(round(current_price, 2)),
            'short_ma': float(round(latest[f'MA_{self.short_window}'], 2)),
            'long_ma': float(round(latest[f'MA_{self.long_window}'], 2)),
            'ema_9': float(round(ema_9, 2)),
            'ema_18': float(round(ema_18, 2)),
            'ema_27': float(round(ema_27, 2)),
            'bb_upper': float(round(latest['BB_Upper'], 2)),
            'bb_middle': float(round(latest['BB_Middle'], 2)),
            'bb_lower': float(round(latest['BB_Lower'], 2)),
            'bb_position': float(round(latest['BB_Position'], 3)),
            'current_signal': int(latest['Signal']),
            'signal_strength': float(latest['Signal_Strength']),
            'confidence': float(round(latest['Confidence'], 1)),
            'signal_type': signal_type,
            'target_price': float(round(target_stoploss['target_price'], 2)) if target_stoploss['target_price'] else None,
            'stop_loss': float(round(target_stoploss['stop_loss'], 2)) if target_stoploss['stop_loss'] else None,
            'volume_info': {
                'current_volume': int(volume_info['current_volume']),
                'avg_volume': float(volume_info['avg_volume']),
                'volume_ratio': float(volume_info['volume_ratio']),
                'is_high_volume': bool(volume_info['is_high_volume'])
            },
            'trend': trend,
            'timestamp': datetime.now()
        }
        
        return result
    
    def format_enhanced_signal_message(self, analysis):
        """
        Format enhanced analysis result into a readable message
        
        Args:
            analysis (dict): Enhanced analysis result
        
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
⚡ <b>Signal Strength:</b> {analysis['signal_strength']}/3

<b>📉 Moving Averages:</b>
• MA20: ₹{analysis['short_ma']}
• MA50: ₹{analysis['long_ma']}

<b>📈 EMAs:</b>
• EMA9: ₹{analysis['ema_9']}
• EMA18: ₹{analysis['ema_18']}
• EMA27: ₹{analysis['ema_27']}

<b>📊 Bollinger Bands:</b>
• Upper: ₹{analysis['bb_upper']}
• Middle: ₹{analysis['bb_middle']}
• Lower: ₹{analysis['bb_lower']}
• Position: {analysis['bb_position']}

<b>🎯 Risk Management:</b>
• Target: ₹{analysis['target_price']}
• Stop Loss: ₹{analysis['stop_loss']}

<b>📊 Volume:</b>
• Current: {analysis['volume_info']['current_volume']:,.0f}
• Avg: {analysis['volume_info']['avg_volume']:,.0f}
• Ratio: {analysis['volume_info']['volume_ratio']:.2f}x
• High Volume: {'Yes' if analysis['volume_info']['is_high_volume'] else 'No'}

⏰ <b>Time:</b> {analysis['timestamp'].strftime('%d-%m-%Y %H:%M:%S')}

<i>⚠️ This is not financial advice. Please do your own research.</i>
            """
        else:
            message = f"""
📊 <b>Market Update</b>

📈 <b>Stock:</b> {symbol_clean}
💰 <b>Price:</b> ₹{analysis['current_price']}
📊 <b>Trend:</b> {analysis['trend']}

<b>📈 EMAs:</b>
• EMA9: ₹{analysis['ema_9']}
• EMA18: ₹{analysis['ema_18']}
• EMA27: ₹{analysis['ema_27']}

<b>📊 Bollinger Bands:</b>
• Position: {analysis['bb_position']}

⏰ <b>Time:</b> {analysis['timestamp'].strftime('%d-%m-%Y %H:%M:%S')}
            """
        
        return message.strip()
    
    def run_enhanced_analysis(self):
        """
        Run enhanced analysis on all stocks in watchlist
        """
        logger.info("Starting enhanced market analysis...")
        
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
                        message = self.format_enhanced_signal_message(analysis)
                        self.send_telegram_message(message)
                        
                        logger.info(f"Enhanced signal generated for {symbol}: {analysis['signal_type']} (Strength: {analysis['signal_strength']})")
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        if not signals_found:
            logger.info("No new enhanced signals generated")
        else:
            logger.info(f"Generated {len(signals_found)} enhanced signals")
        
        return signals_found

def main():
    """
    Main function to run the enhanced trading strategy
    """
    # Configuration
    TELEGRAM_BOT_TOKEN = None  # Add your Telegram bot token here
    TELEGRAM_CHAT_ID = None    # Add your Telegram chat ID here
    
    # Initialize enhanced strategy
    strategy = EnhancedTradingStrategy(
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID
    )
    
    print("🚀 Enhanced Trading Strategy Started")
    print("=" * 50)
    
    try:
        # Run enhanced analysis
        signals = strategy.run_enhanced_analysis()
        
        print(f"\n📊 Enhanced Analysis Complete!")
        print(f"📈 Signals Generated: {len(signals)}")
        
        print("\n✅ Enhanced strategy execution completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Enhanced strategy stopped by user")
    except Exception as e:
        logger.error(f"Error in enhanced main execution: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

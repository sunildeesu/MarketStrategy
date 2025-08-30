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
    
    def calculate_rsi(self, data, period=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            data (pandas.Series): Price data
            period (int): RSI period
        
        Returns:
            pandas.Series: RSI values
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data (pandas.Series): Price data
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line EMA period
        
        Returns:
            tuple: (macd_line, signal_line, histogram)
        """
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator
        
        Args:
            high (pandas.Series): High prices
            low (pandas.Series): Low prices
            close (pandas.Series): Close prices
            k_period (int): %K period
            d_period (int): %D period
        
        Returns:
            tuple: (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def calculate_williams_r(self, high, low, close, period=14):
        """
        Calculate Williams %R
        
        Args:
            high (pandas.Series): High prices
            low (pandas.Series): Low prices
            close (pandas.Series): Close prices
            period (int): Period for calculation
        
        Returns:
            pandas.Series: Williams %R values
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def calculate_atr(self, high, low, close, period=14):
        """
        Calculate Average True Range (ATR)
        
        Args:
            high (pandas.Series): High prices
            low (pandas.Series): Low prices
            close (pandas.Series): Close prices
            period (int): ATR period
        
        Returns:
            pandas.Series: ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def calculate_adx(self, high, low, close, period=14):
        """
        Calculate Average Directional Index (ADX)
        
        Args:
            high (pandas.Series): High prices
            low (pandas.Series): Low prices
            close (pandas.Series): Close prices
            period (int): ADX period
        
        Returns:
            tuple: (ADX, +DI, -DI)
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = high.diff()
        dm_minus = low.diff() * -1
        
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        dm_plus[(dm_plus - dm_minus) < 0] = 0
        dm_minus[(dm_minus - dm_plus) < 0] = 0
        
        # Calculate smoothed averages
        tr_smooth = true_range.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # Calculate DX and ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx, di_plus, di_minus
    
    def calculate_obv(self, close, volume):
        """
        Calculate On-Balance Volume (OBV)
        
        Args:
            close (pandas.Series): Close prices
            volume (pandas.Series): Volume data
        
        Returns:
            pandas.Series: OBV values
        """
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def analyze_obv_trend(self, obv, close, period=20):
        """
        Analyze OBV trend and generate recommendation
        
        Args:
            obv (pandas.Series): OBV values
            close (pandas.Series): Close prices
            period (int): Period for trend analysis
        
        Returns:
            dict: OBV analysis results
        """
        # Calculate OBV moving average for trend
        obv_ma = obv.rolling(window=period).mean()
        
        # Get recent values
        current_obv = obv.iloc[-1]
        prev_obv = obv.iloc[-2] if len(obv) > 1 else current_obv
        obv_ma_current = obv_ma.iloc[-1]
        obv_ma_prev = obv_ma.iloc[-2] if len(obv_ma) > 1 else obv_ma_current
        
        # Calculate price trend
        price_ma = close.rolling(window=period).mean()
        price_trend = "UP" if close.iloc[-1] > price_ma.iloc[-1] else "DOWN"
        
        # Calculate OBV trend
        obv_trend = "UP" if current_obv > obv_ma_current else "DOWN"
        obv_momentum = "RISING" if obv_ma_current > obv_ma_prev else "FALLING"
        
        # Generate recommendation based on OBV analysis
        if obv_trend == "UP" and price_trend == "UP" and obv_momentum == "RISING":
            recommendation = "STRONG_BULLISH"
            signal = "BUY"
            strength = 3
        elif obv_trend == "UP" and price_trend == "UP":
            recommendation = "BULLISH"
            signal = "BUY"
            strength = 2
        elif obv_trend == "UP" and price_trend == "DOWN":
            recommendation = "DIVERGENCE_BULLISH"
            signal = "WATCH"
            strength = 1
        elif obv_trend == "DOWN" and price_trend == "DOWN" and obv_momentum == "FALLING":
            recommendation = "STRONG_BEARISH"
            signal = "SELL"
            strength = -3
        elif obv_trend == "DOWN" and price_trend == "DOWN":
            recommendation = "BEARISH"
            signal = "SELL"
            strength = -2
        elif obv_trend == "DOWN" and price_trend == "UP":
            recommendation = "DIVERGENCE_BEARISH"
            signal = "WATCH"
            strength = -1
        else:
            recommendation = "NEUTRAL"
            signal = "HOLD"
            strength = 0
        
        return {
            'current_obv': current_obv,
            'obv_trend': obv_trend,
            'obv_momentum': obv_momentum,
            'price_trend': price_trend,
            'recommendation': recommendation,
            'signal': signal,
            'strength': strength,
            'obv_ma': obv_ma_current
        }
    
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
    
    def calculate_target_and_stoploss(self, current_price, signal_type, bb_upper, bb_middle, bb_lower):
        """
        Calculate target price and stop loss using Bollinger Bands
        
        Args:
            current_price (float): Current stock price
            signal_type (str): 'BUY' or 'SELL'
            bb_upper (float): Bollinger Band upper level
            bb_middle (float): Bollinger Band middle level (SMA)
            bb_lower (float): Bollinger Band lower level
        
        Returns:
            dict: Target and stop loss prices based on Bollinger Bands
        """
        if signal_type == 'BUY':
            # For BUY signals:
            # Target: Upper Bollinger Band (resistance level)
            # Stop Loss: Lower Bollinger Band or slightly below current price
            target_price = bb_upper
            
            # Use the lower of: BB lower band or 3% below current price (safety net)
            safety_stop = current_price * (1 - 0.03)  # 3% safety stop
            stop_loss = min(bb_lower, safety_stop)
            
        elif signal_type == 'SELL':
            # For SELL signals:
            # Target: Lower Bollinger Band (support level)
            # Stop Loss: Upper Bollinger Band or slightly above current price
            target_price = bb_lower
            
            # Use the higher of: BB upper band or 3% above current price (safety net)
            safety_stop = current_price * (1 + 0.03)  # 3% safety stop
            stop_loss = max(bb_upper, safety_stop)
            
        else:
            target_price = None
            stop_loss = None
        
        return {
            'target_price': target_price,
            'stop_loss': stop_loss,
            'bb_target_method': True,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower
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
        
        # RSI
        data['RSI'] = self.calculate_rsi(data['Close'])
        
        # MACD
        macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
        data['MACD'] = macd_line
        data['MACD_Signal'] = signal_line
        data['MACD_Histogram'] = histogram
        
        # Stochastic Oscillator
        stoch_k, stoch_d = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
        data['Stoch_K'] = stoch_k
        data['Stoch_D'] = stoch_d
        
        # Williams %R
        data['Williams_R'] = self.calculate_williams_r(data['High'], data['Low'], data['Close'])
        
        # ATR (Average True Range)
        data['ATR'] = self.calculate_atr(data['High'], data['Low'], data['Close'])
        
        # ADX (Average Directional Index)
        adx, di_plus, di_minus = self.calculate_adx(data['High'], data['Low'], data['Close'])
        data['ADX'] = adx
        data['DI_Plus'] = di_plus
        data['DI_Minus'] = di_minus
        
        # OBV (On-Balance Volume)
        data['OBV'] = self.calculate_obv(data['Close'], data['Volume'])
        
        return data
    
    def generate_advanced_signals(self, data):
        """
        Generate advanced buy/sell signals using comprehensive technical analysis
        
        Args:
            data (pandas.DataFrame): Stock data with all indicators
        
        Returns:
            pandas.DataFrame: Data with advanced signals, confidence, and recommendations
        """
        # Initialize signal columns
        data['Signal'] = 0
        data['Position'] = 0
        data['Signal_Strength'] = 0
        data['Confidence'] = 0.0  # Initialize as float to avoid data type warnings
        data['Recommendation'] = 'HOLD'
        
        # Get latest values
        latest = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else latest
        
        # Initialize signal components and confidence factors
        signals = []
        confidence_components = {}
        
        # 1. TREND ANALYSIS (25% weight)
        trend_signal = 0
        trend_confidence = 15  # Base confidence for having trend data
        
        # EMA Trend Analysis
        ema_9 = latest['EMA_9']
        ema_18 = latest['EMA_18'] 
        ema_27 = latest['EMA_27']
        
        if ema_9 > ema_18 > ema_27:
            trend_signal = 2  # Strong bullish trend
            trend_strength = ((ema_9 - ema_27) / ema_27) * 100
            trend_confidence = min(25, max(20, 15 + abs(trend_strength) * 50))  # Much better scaling
        elif ema_9 > ema_18 or ema_18 > ema_27:
            trend_signal = 1  # Moderate bullish trend
            trend_confidence = 22  # Higher base confidence
        elif ema_9 < ema_18 < ema_27:
            trend_signal = -2  # Strong bearish trend
            trend_strength = ((ema_27 - ema_9) / ema_9) * 100
            trend_confidence = min(25, max(20, 15 + abs(trend_strength) * 50))  # Much better scaling
        elif ema_9 < ema_18 or ema_18 < ema_27:
            trend_signal = -1  # Moderate bearish trend
            trend_confidence = 22  # Higher base confidence
        
        signals.append(trend_signal)
        confidence_components['trend'] = trend_confidence
        
        # 2. MOMENTUM ANALYSIS (20% weight)
        momentum_signal = 0
        momentum_confidence = 10  # Base confidence for having momentum data
        
        # RSI Analysis
        rsi = latest['RSI']
        if pd.notna(rsi):
            if rsi < 30:  # Oversold
                momentum_signal += 1
                momentum_confidence += min(8, (30 - rsi) * 0.4 + 5)  # Better scaling with base
            elif rsi > 70:  # Overbought
                momentum_signal -= 1
                momentum_confidence += min(8, (rsi - 70) * 0.4 + 5)  # Better scaling with base
            else:  # Neutral zone
                momentum_confidence += 6  # Higher base confidence for valid RSI
        
        # MACD Analysis
        macd = latest['MACD']
        macd_signal_line = latest['MACD_Signal']
        macd_histogram = latest['MACD_Histogram']
        
        if pd.notna(macd) and pd.notna(macd_signal_line):
            if macd > macd_signal_line and macd_histogram > 0:
                momentum_signal += 1
                momentum_confidence += min(6, max(4, abs(macd_histogram) * 2000 + 3))  # Much better scaling
            elif macd < macd_signal_line and macd_histogram < 0:
                momentum_signal -= 1
                momentum_confidence += min(6, max(4, abs(macd_histogram) * 2000 + 3))  # Much better scaling
            else:
                momentum_confidence += 4  # Higher base confidence for valid MACD
        
        signals.append(momentum_signal)
        confidence_components['momentum'] = min(20, momentum_confidence)
        
        # 3. OSCILLATOR ANALYSIS (15% weight)
        oscillator_signal = 0
        oscillator_confidence = 8  # Base confidence for having oscillator data
        
        # Stochastic Analysis
        stoch_k = latest['Stoch_K']
        stoch_d = latest['Stoch_D']
        
        if pd.notna(stoch_k) and pd.notna(stoch_d):
            if stoch_k < 20 and stoch_d < 20:  # Oversold
                oscillator_signal += 1
                oscillator_confidence += min(5, (20 - min(stoch_k, stoch_d)) * 0.3 + 2)  # Better scaling
            elif stoch_k > 80 and stoch_d > 80:  # Overbought
                oscillator_signal -= 1
                oscillator_confidence += min(5, (min(stoch_k, stoch_d) - 80) * 0.3 + 2)  # Better scaling
            else:
                oscillator_confidence += 4  # Higher base confidence for valid stochastic
        
        # Williams %R Analysis
        williams_r = latest['Williams_R']
        if pd.notna(williams_r):
            if williams_r < -80:  # Oversold
                oscillator_signal += 1
                oscillator_confidence += min(4, (-80 - williams_r) * 0.2 + 1)  # Better scaling
            elif williams_r > -20:  # Overbought
                oscillator_signal -= 1
                oscillator_confidence += min(4, (williams_r + 20) * 0.2 + 1)  # Better scaling
            else:
                oscillator_confidence += 3  # Higher base confidence for valid Williams %R
        
        signals.append(oscillator_signal)
        confidence_components['oscillator'] = min(15, oscillator_confidence)
        
        # 4. VOLATILITY & SUPPORT/RESISTANCE (15% weight)
        volatility_signal = 0
        volatility_confidence = 8  # Base confidence for having volatility data
        
        # Bollinger Bands Analysis
        bb_position = latest['BB_Position']
        if pd.notna(bb_position):
            if bb_position < 0.1:  # Near lower band - oversold
                volatility_signal += 2
                volatility_confidence += min(7, (0.1 - bb_position) * 70 + 3)  # Much better scaling
            elif bb_position < 0.2:  # Approaching lower band
                volatility_signal += 1
                volatility_confidence += min(5, (0.2 - bb_position) * 50 + 2)  # Much better scaling
            elif bb_position > 0.9:  # Near upper band - overbought
                volatility_signal -= 2
                volatility_confidence += min(7, (bb_position - 0.9) * 70 + 3)  # Much better scaling
            elif bb_position > 0.8:  # Approaching upper band
                volatility_signal -= 1
                volatility_confidence += min(5, (bb_position - 0.8) * 50 + 2)  # Much better scaling
            else:
                volatility_confidence += 6  # Higher base confidence for valid BB position
        
        signals.append(volatility_signal)
        confidence_components['volatility'] = min(15, volatility_confidence)
        
        # 5. TREND STRENGTH ANALYSIS (10% weight)
        strength_signal = 0
        strength_confidence = 5  # Base confidence for having strength data
        
        # ADX Analysis
        adx = latest['ADX']
        di_plus = latest['DI_Plus']
        di_minus = latest['DI_Minus']
        
        if pd.notna(adx) and pd.notna(di_plus) and pd.notna(di_minus):
            if adx > 25:  # Strong trend
                if di_plus > di_minus:
                    strength_signal += 1
                    strength_confidence += min(5, max(3, (adx - 25) * 0.1 + 2))  # Better scaling
                else:
                    strength_signal -= 1
                    strength_confidence += min(5, max(3, (adx - 25) * 0.1 + 2))  # Better scaling
            elif adx > 15:  # Moderate trend
                strength_confidence += 3  # Higher base confidence for moderate trend
            else:
                strength_confidence += 2  # Base confidence for weak trend
        
        signals.append(strength_signal)
        confidence_components['strength'] = min(10, strength_confidence)
        
        # 6. VOLUME CONFIRMATION (15% weight)
        volume_signal = 0
        volume_confidence = 0
        
        volume_data = data['Volume']
        current_volume = volume_data.iloc[-1]
        avg_volume = volume_data.rolling(window=20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 3.0:  # Exceptional volume
            volume_confidence = 15
        elif volume_ratio > 2.0:  # Very high volume
            volume_confidence = 13
        elif volume_ratio > 1.5:  # High volume
            volume_confidence = 11
        elif volume_ratio > 1.2:  # Above average volume
            volume_confidence = 9
        elif volume_ratio > 0.8:  # Normal volume
            volume_confidence = 7
        else:  # Low volume
            volume_confidence = 4
        
        # Volume doesn't generate signals by itself, but confirms other signals
        confidence_components['volume'] = volume_confidence
        
        # CALCULATE OVERALL SIGNAL AND CONFIDENCE
        total_signal_strength = sum(signals)
        
        # Calculate weighted confidence
        total_confidence = (
            confidence_components.get('trend', 0) * 0.25 +
            confidence_components.get('momentum', 0) * 0.20 +
            confidence_components.get('oscillator', 0) * 0.15 +
            confidence_components.get('volatility', 0) * 0.15 +
            confidence_components.get('strength', 0) * 0.10 +
            confidence_components.get('volume', 0) * 0.15
        )
        
        # Normalize confidence to 0-100
        total_confidence = min(100.0, total_confidence)
        
        # GENERATE FINAL SIGNAL AND RECOMMENDATION
        final_signal = 0
        recommendation = 'HOLD'
        
        if total_signal_strength >= 4 and total_confidence >= 70:
            final_signal = 2
            recommendation = 'STRONG_BUY'
        elif total_signal_strength >= 2 and total_confidence >= 50:
            final_signal = 1
            recommendation = 'BUY'
        elif total_signal_strength <= -4 and total_confidence >= 70:
            final_signal = -2
            recommendation = 'STRONG_SELL'
        elif total_signal_strength <= -2 and total_confidence >= 50:
            final_signal = -1
            recommendation = 'SELL'
        elif total_confidence < 30:
            recommendation = 'INSUFFICIENT_DATA'
        else:
            recommendation = 'HOLD'
        
        # Store results
        data.iloc[-1, data.columns.get_loc('Signal')] = final_signal
        data.iloc[-1, data.columns.get_loc('Signal_Strength')] = total_signal_strength
        data.iloc[-1, data.columns.get_loc('Confidence')] = float(total_confidence)  # Fix data type warning
        data.iloc[-1, data.columns.get_loc('Recommendation')] = recommendation
        
        # Calculate position changes
        if len(data) > 1:
            prev_signal = data.iloc[-2]['Signal'] if 'Signal' in data.columns else 0
            if final_signal != prev_signal:
                data.iloc[-1, data.columns.get_loc('Position')] = final_signal - prev_signal
        
        return data
    
    # Keep the old method name for compatibility
    def generate_enhanced_signals(self, data):
        """Wrapper for backward compatibility"""
        return self.generate_advanced_signals(data)
    
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
        
        # OBV analysis
        obv_analysis = self.analyze_obv_trend(data['OBV'], data['Close'])
        
        # Determine trend based on EMA alignment first
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
        
        # Determine signal type based on current signal, not position change
        signal_type = None
        if latest['Signal'] > 0:
            signal_type = 'BUY'
        elif latest['Signal'] < 0:
            signal_type = 'SELL'
        else:
            # For HOLD signals, determine target/stop based on trend
            if trend in ['STRONG_BULLISH', 'BULLISH']:
                signal_type = 'BUY'  # Use BUY logic for bullish trends
            elif trend in ['STRONG_BEARISH', 'BEARISH']:
                signal_type = 'SELL'  # Use SELL logic for bearish trends
            else:
                signal_type = 'BUY'  # Default to BUY logic for neutral trends
        
        # Calculate target and stop loss using Bollinger Bands
        target_stoploss = self.calculate_target_and_stoploss(
            current_price, 
            signal_type,
            latest['BB_Upper'],
            latest['BB_Middle'], 
            latest['BB_Lower']
        )
        
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
            'recommendation': latest['Recommendation'],
            'signal_type': signal_type,
            'target_price': float(round(target_stoploss['target_price'], 2)) if target_stoploss['target_price'] else None,
            'stop_loss': float(round(target_stoploss['stop_loss'], 2)) if target_stoploss['stop_loss'] else None,
            
            # Advanced Technical Indicators
            'technical_indicators': {
                'rsi': float(round(latest['RSI'], 2)) if pd.notna(latest['RSI']) else None,
                'macd': float(round(latest['MACD'], 4)) if pd.notna(latest['MACD']) else None,
                'macd_signal': float(round(latest['MACD_Signal'], 4)) if pd.notna(latest['MACD_Signal']) else None,
                'macd_histogram': float(round(latest['MACD_Histogram'], 4)) if pd.notna(latest['MACD_Histogram']) else None,
                'stoch_k': float(round(latest['Stoch_K'], 2)) if pd.notna(latest['Stoch_K']) else None,
                'stoch_d': float(round(latest['Stoch_D'], 2)) if pd.notna(latest['Stoch_D']) else None,
                'williams_r': float(round(latest['Williams_R'], 2)) if pd.notna(latest['Williams_R']) else None,
                'atr': float(round(latest['ATR'], 2)) if pd.notna(latest['ATR']) else None,
                'adx': float(round(latest['ADX'], 2)) if pd.notna(latest['ADX']) else None,
                'di_plus': float(round(latest['DI_Plus'], 2)) if pd.notna(latest['DI_Plus']) else None,
                'di_minus': float(round(latest['DI_Minus'], 2)) if pd.notna(latest['DI_Minus']) else None,
            },
            
            'volume_info': {
                'current_volume': int(volume_info['current_volume']),
                'avg_volume': float(volume_info['avg_volume']),
                'volume_ratio': float(volume_info['volume_ratio']),
                'is_high_volume': bool(volume_info['is_high_volume'])
            },
            
            # OBV Analysis
            'obv_analysis': {
                'current_obv': float(obv_analysis['current_obv']),
                'obv_trend': obv_analysis['obv_trend'],
                'obv_momentum': obv_analysis['obv_momentum'],
                'price_trend': obv_analysis['price_trend'],
                'recommendation': obv_analysis['recommendation'],
                'signal': obv_analysis['signal'],
                'strength': int(obv_analysis['strength']),
                'obv_ma': float(obv_analysis['obv_ma'])
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
    
    def backtest_strategy(self, symbol, start_date='2023-01-01', end_date='2024-12-31', initial_capital=100000):
        """
        Comprehensive backtesting of the trading strategy
        
        Args:
            symbol (str): Stock symbol to backtest
            start_date (str): Start date for backtesting
            end_date (str): End date for backtesting
            initial_capital (float): Initial capital for backtesting
        
        Returns:
            dict: Comprehensive backtesting results
        """
        logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
        
        try:
            # Get historical data
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if len(data) < 100:  # Need sufficient data
                logger.warning(f"Insufficient data for backtesting {symbol}")
                return None
            
            # Calculate all technical indicators
            data = self.calculate_technical_indicators(data)
            
            # Initialize backtesting variables
            capital = initial_capital
            position = 0  # 0 = no position, 1 = long, -1 = short
            entry_price = 0
            entry_date = None
            target_price = 0
            stop_loss = 0
            
            trades = []
            portfolio_values = []
            signals_generated = []
            
            # Process each day
            for i in range(len(data)):
                if i < max(self.long_window, max(self.ema_periods), 26):  # Skip initial period
                    portfolio_values.append(capital)
                    continue
                
                # Get current data slice for signal generation
                current_data = data.iloc[:i+1].copy()
                current_data = self.generate_advanced_signals(current_data)
                
                current_row = current_data.iloc[-1]
                current_price = current_row['Close']
                current_signal = current_row['Signal']
                recommendation = current_row['Recommendation']
                confidence = current_row['Confidence']
                
                # Record signal
                if current_signal != 0:
                    signals_generated.append({
                        'date': current_row.name,
                        'price': current_price,
                        'signal': current_signal,
                        'recommendation': recommendation,
                        'confidence': confidence
                    })
                
                # Trading logic
                if position == 0:  # No current position
                    if current_signal > 0 and confidence >= 50:  # Buy signal with sufficient confidence
                        # Enter long position
                        position = 1
                        entry_price = current_price
                        entry_date = current_row.name
                        
                        # Calculate target and stop loss using Bollinger Bands
                        target_stop = self.calculate_target_and_stoploss(
                            current_price, 'BUY',
                            current_row['BB_Upper'],
                            current_row['BB_Middle'],
                            current_row['BB_Lower']
                        )
                        target_price = target_stop['target_price']
                        stop_loss = target_stop['stop_loss']
                        
                        logger.debug(f"BUY: {symbol} at ₹{current_price:.2f} on {entry_date.date()}")
                        
                    elif current_signal < 0 and confidence >= 50:  # Sell signal with sufficient confidence
                        # Enter short position (for backtesting purposes)
                        position = -1
                        entry_price = current_price
                        entry_date = current_row.name
                        
                        # Calculate target and stop loss for short position
                        target_stop = self.calculate_target_and_stoploss(
                            current_price, 'SELL',
                            current_row['BB_Upper'],
                            current_row['BB_Middle'],
                            current_row['BB_Lower']
                        )
                        target_price = target_stop['target_price']
                        stop_loss = target_stop['stop_loss']
                        
                        logger.debug(f"SELL: {symbol} at ₹{current_price:.2f} on {entry_date.date()}")
                
                elif position == 1:  # Long position
                    # Check exit conditions
                    exit_trade = False
                    exit_reason = ""
                    
                    if current_price >= target_price:
                        exit_trade = True
                        exit_reason = "Target Hit"
                    elif current_price <= stop_loss:
                        exit_trade = True
                        exit_reason = "Stop Loss"
                    elif current_signal < 0 and confidence >= 60:  # Strong opposite signal
                        exit_trade = True
                        exit_reason = "Signal Reversal"
                    
                    if exit_trade:
                        # Calculate trade result
                        shares = capital / entry_price
                        trade_pnl = shares * (current_price - entry_price)
                        capital += trade_pnl
                        
                        trade_return = (current_price - entry_price) / entry_price * 100
                        days_held = (current_row.name - entry_date).days
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_row.name,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position_type': 'LONG',
                            'return_pct': trade_return,
                            'pnl': trade_pnl,
                            'days_held': days_held,
                            'exit_reason': exit_reason,
                            'target_price': target_price,
                            'stop_loss': stop_loss
                        })
                        
                        position = 0
                        logger.debug(f"EXIT LONG: {symbol} at ₹{current_price:.2f}, Return: {trade_return:.2f}%, Reason: {exit_reason}")
                
                elif position == -1:  # Short position
                    # Check exit conditions for short
                    exit_trade = False
                    exit_reason = ""
                    
                    if current_price <= target_price:
                        exit_trade = True
                        exit_reason = "Target Hit"
                    elif current_price >= stop_loss:
                        exit_trade = True
                        exit_reason = "Stop Loss"
                    elif current_signal > 0 and confidence >= 60:  # Strong opposite signal
                        exit_trade = True
                        exit_reason = "Signal Reversal"
                    
                    if exit_trade:
                        # Calculate trade result for short position
                        shares = capital / entry_price
                        trade_pnl = shares * (entry_price - current_price)  # Profit when price goes down
                        capital += trade_pnl
                        
                        trade_return = (entry_price - current_price) / entry_price * 100
                        days_held = (current_row.name - entry_date).days
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_row.name,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position_type': 'SHORT',
                            'return_pct': trade_return,
                            'pnl': trade_pnl,
                            'days_held': days_held,
                            'exit_reason': exit_reason,
                            'target_price': target_price,
                            'stop_loss': stop_loss
                        })
                        
                        position = 0
                        logger.debug(f"EXIT SHORT: {symbol} at ₹{current_price:.2f}, Return: {trade_return:.2f}%, Reason: {exit_reason}")
                
                # Record portfolio value
                if position == 1:  # Long position
                    shares = capital / entry_price
                    current_portfolio_value = shares * current_price
                elif position == -1:  # Short position
                    shares = capital / entry_price
                    current_portfolio_value = capital + shares * (entry_price - current_price)
                else:  # No position
                    current_portfolio_value = capital
                
                portfolio_values.append(current_portfolio_value)
            
            # Calculate performance metrics
            if not trades:
                logger.warning(f"No trades generated for {symbol}")
                return None
            
            # Calculate buy and hold return for comparison
            buy_hold_return = (data['Close'].iloc[-1] - data['Close'].iloc[max(self.long_window, max(self.ema_periods), 26)]) / data['Close'].iloc[max(self.long_window, max(self.ema_periods), 26)] * 100
            
            # Performance calculations
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['return_pct'] > 0])
            losing_trades = len([t for t in trades if t['return_pct'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            returns = [t['return_pct'] for t in trades]
            avg_return = np.mean(returns) if returns else 0
            avg_winning_return = np.mean([r for r in returns if r > 0]) if [r for r in returns if r > 0] else 0
            avg_losing_return = np.mean([r for r in returns if r < 0]) if [r for r in returns if r < 0] else 0
            
            total_return = (capital - initial_capital) / initial_capital * 100
            
            # Risk metrics
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                max_drawdown = self._calculate_max_drawdown(portfolio_values)
            else:
                sharpe_ratio = 0
                max_drawdown = 0
            
            # Profit factor
            gross_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
            gross_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average holding period
            avg_holding_days = np.mean([t['days_held'] for t in trades]) if trades else 0
            
            backtest_results = {
                'symbol': symbol,
                'period': f"{start_date} to {end_date}",
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return_pct': total_return,
                'buy_hold_return_pct': buy_hold_return,
                'outperformance': total_return - buy_hold_return,
                
                'trade_statistics': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate_pct': win_rate,
                    'avg_return_pct': avg_return,
                    'avg_winning_return_pct': avg_winning_return,
                    'avg_losing_return_pct': avg_losing_return,
                    'avg_holding_days': avg_holding_days
                },
                
                'risk_metrics': {
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown_pct': max_drawdown,
                    'profit_factor': profit_factor,
                    'gross_profit': gross_profit,
                    'gross_loss': gross_loss
                },
                
                'signals_generated': len(signals_generated),
                'trades_executed': total_trades,
                'signal_to_trade_ratio': (total_trades / len(signals_generated) * 100) if signals_generated else 0,
                
                'detailed_trades': trades,
                'portfolio_curve': portfolio_values
            }
            
            logger.info(f"Backtest completed for {symbol}: {total_trades} trades, {win_rate:.1f}% win rate, {total_return:.2f}% return")
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error in backtesting {symbol}: {e}")
            return None
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown from portfolio values"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def run_comprehensive_backtest(self, symbols=None, start_date='2023-01-01', end_date='2024-12-31'):
        """
        Run comprehensive backtesting on multiple symbols
        
        Args:
            symbols (list): List of symbols to backtest (default: watchlist)
            start_date (str): Start date for backtesting
            end_date (str): End date for backtesting
        
        Returns:
            dict: Comprehensive backtesting results for all symbols
        """
        if symbols is None:
            symbols = self.watchlist
        
        logger.info(f"Starting comprehensive backtest for {len(symbols)} symbols")
        
        all_results = {}
        successful_backtests = []
        
        for symbol in symbols:
            try:
                result = self.backtest_strategy(symbol, start_date, end_date)
                if result:
                    all_results[symbol] = result
                    successful_backtests.append(result)
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {e}")
        
        if not successful_backtests:
            logger.error("No successful backtests completed")
            return None
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_statistics(successful_backtests)
        
        comprehensive_results = {
            'individual_results': all_results,
            'aggregate_statistics': aggregate_stats,
            'summary': {
                'total_symbols_tested': len(symbols),
                'successful_backtests': len(successful_backtests),
                'period': f"{start_date} to {end_date}",
                'strategy_name': "Advanced Multi-Indicator Strategy"
            }
        }
        
        return comprehensive_results
    
    def _calculate_aggregate_statistics(self, results):
        """Calculate aggregate statistics across all backtests"""
        if not results:
            return {}
        
        # Aggregate metrics
        total_trades = sum([r['trade_statistics']['total_trades'] for r in results])
        total_winning = sum([r['trade_statistics']['winning_trades'] for r in results])
        total_losing = sum([r['trade_statistics']['losing_trades'] for r in results])
        
        avg_win_rate = np.mean([r['trade_statistics']['win_rate_pct'] for r in results])
        avg_return = np.mean([r['total_return_pct'] for r in results])
        avg_outperformance = np.mean([r['outperformance'] for r in results])
        
        # Risk metrics
        avg_sharpe = np.mean([r['risk_metrics']['sharpe_ratio'] for r in results if r['risk_metrics']['sharpe_ratio'] != 0])
        avg_max_drawdown = np.mean([r['risk_metrics']['max_drawdown_pct'] for r in results])
        avg_profit_factor = np.mean([r['risk_metrics']['profit_factor'] for r in results if r['risk_metrics']['profit_factor'] != float('inf')])
        
        # Best and worst performers
        best_performer = max(results, key=lambda x: x['total_return_pct'])
        worst_performer = min(results, key=lambda x: x['total_return_pct'])
        
        # Consistency metrics
        positive_returns = len([r for r in results if r['total_return_pct'] > 0])
        consistency_rate = (positive_returns / len(results)) * 100
        
        return {
            'total_trades_all_symbols': total_trades,
            'overall_win_rate_pct': (total_winning / total_trades * 100) if total_trades > 0 else 0,
            'average_return_pct': avg_return,
            'average_outperformance_pct': avg_outperformance,
            'consistency_rate_pct': consistency_rate,
            'average_sharpe_ratio': avg_sharpe,
            'average_max_drawdown_pct': avg_max_drawdown,
            'average_profit_factor': avg_profit_factor,
            'best_performer': {
                'symbol': best_performer['symbol'],
                'return_pct': best_performer['total_return_pct']
            },
            'worst_performer': {
                'symbol': worst_performer['symbol'],
                'return_pct': worst_performer['total_return_pct']
            },
            'symbols_with_positive_returns': positive_returns,
            'total_symbols_tested': len(results)
        }
    
    def generate_backtest_report(self, backtest_results):
        """
        Generate a comprehensive backtest report
        
        Args:
            backtest_results (dict): Results from run_comprehensive_backtest
        
        Returns:
            str: Formatted backtest report
        """
        if not backtest_results:
            return "No backtest results available"
        
        agg_stats = backtest_results['aggregate_statistics']
        summary = backtest_results['summary']
        
        report = f"""
🔍 COMPREHENSIVE STRATEGY BACKTEST REPORT
{'='*60}

📊 STRATEGY OVERVIEW
Strategy: {summary['strategy_name']}
Period: {summary['period']}
Symbols Tested: {summary['total_symbols_tested']}
Successful Backtests: {summary['successful_backtests']}

📈 AGGREGATE PERFORMANCE METRICS
{'='*40}
Total Trades (All Symbols): {agg_stats['total_trades_all_symbols']}
Overall Win Rate: {agg_stats['overall_win_rate_pct']:.1f}%
Average Return: {agg_stats['average_return_pct']:.2f}%
Average Outperformance vs Buy & Hold: {agg_stats['average_outperformance_pct']:.2f}%
Consistency Rate: {agg_stats['consistency_rate_pct']:.1f}% (symbols with positive returns)

⚖️ RISK METRICS
{'='*40}
Average Sharpe Ratio: {agg_stats['average_sharpe_ratio']:.2f}
Average Max Drawdown: {agg_stats['average_max_drawdown_pct']:.2f}%
Average Profit Factor: {agg_stats['average_profit_factor']:.2f}

🏆 PERFORMANCE HIGHLIGHTS
{'='*40}
Best Performer: {agg_stats['best_performer']['symbol']} ({agg_stats['best_performer']['return_pct']:.2f}%)
Worst Performer: {agg_stats['worst_performer']['symbol']} ({agg_stats['worst_performer']['return_pct']:.2f}%)
Profitable Symbols: {agg_stats['symbols_with_positive_returns']}/{agg_stats['total_symbols_tested']}

📋 INDIVIDUAL STOCK PERFORMANCE
{'='*40}"""
        
        # Add individual stock performance
        for symbol, result in backtest_results['individual_results'].items():
            symbol_clean = symbol.replace('.NS', '')
            report += f"""
{symbol_clean}:
  Return: {result['total_return_pct']:.2f}% | Trades: {result['trade_statistics']['total_trades']} | Win Rate: {result['trade_statistics']['win_rate_pct']:.1f}%
  Outperformance: {result['outperformance']:.2f}% | Max Drawdown: {result['risk_metrics']['max_drawdown_pct']:.2f}%"""
        
        report += f"""

🎯 STRATEGY RELIABILITY ASSESSMENT
{'='*40}
Based on the backtest results, this strategy shows:

✅ STRENGTHS:
• Win Rate: {agg_stats['overall_win_rate_pct']:.1f}% (Good if >50%)
• Consistency: {agg_stats['consistency_rate_pct']:.1f}% of symbols profitable
• Risk-Adjusted Returns: Sharpe Ratio of {agg_stats['average_sharpe_ratio']:.2f}
• Outperformance: Beats buy & hold by {agg_stats['average_outperformance_pct']:.2f}% on average

⚠️ CONSIDERATIONS:
• Max Drawdown: {agg_stats['average_max_drawdown_pct']:.2f}% (risk tolerance required)
• Market Conditions: Results based on {summary['period']} period
• Transaction Costs: Not included in backtest (would reduce returns)

📊 RELIABILITY SCORE: {self._calculate_reliability_score(agg_stats)}/10

⚠️ DISCLAIMER: Past performance does not guarantee future results.
This backtest is for educational purposes only and should not be considered as financial advice.
        """
        
        return report
    
    def _calculate_reliability_score(self, agg_stats):
        """Calculate a reliability score based on key metrics"""
        score = 0
        
        # Win rate component (0-3 points)
        win_rate = agg_stats['overall_win_rate_pct']
        if win_rate >= 60:
            score += 3
        elif win_rate >= 50:
            score += 2
        elif win_rate >= 40:
            score += 1
        
        # Consistency component (0-2 points)
        consistency = agg_stats['consistency_rate_pct']
        if consistency >= 70:
            score += 2
        elif consistency >= 50:
            score += 1
        
        # Sharpe ratio component (0-2 points)
        sharpe = agg_stats['average_sharpe_ratio']
        if sharpe >= 1.0:
            score += 2
        elif sharpe >= 0.5:
            score += 1
        
        # Outperformance component (0-2 points)
        outperformance = agg_stats['average_outperformance_pct']
        if outperformance >= 5:
            score += 2
        elif outperformance >= 0:
            score += 1
        
        # Max drawdown penalty (0-1 points)
        max_dd = agg_stats['average_max_drawdown_pct']
        if max_dd <= 10:
            score += 1
        
        return min(score, 10)  # Cap at 10
    
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

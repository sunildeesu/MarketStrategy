# Simple Trading Strategy for Indian Stock Market

A basic moving average crossover trading strategy designed for the Indian stock market (NSE/BSE) with Telegram notifications.

## 🚀 Features

- **Moving Average Crossover Strategy**: Uses 20-day and 50-day moving averages
- **Indian Stock Focus**: Pre-configured with top NSE stocks
- **Telegram Notifications**: Real-time buy/sell signals via Telegram
- **Market Summary**: Daily market sentiment analysis
- **Easy to Use**: Simple Python script, no complex setup required

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection for fetching stock data
- Telegram account (optional, for notifications)

## 🛠️ Installation

### 1. Clone or Download the Files

Download these files to your computer:
- `simple_trading_strategy.py`
- `requirements.txt`
- `README.md`

### 2. Install Python Dependencies

Open terminal/command prompt and navigate to the project folder, then run:

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install yfinance pandas numpy requests python-dateutil pytz
```

### 3. Set Up Telegram Bot (Optional)

If you want to receive notifications via Telegram:

#### Step 1: Create a Telegram Bot
1. Open Telegram and search for `@BotFather`
2. Start a chat and send `/newbot`
3. Follow instructions to create your bot
4. Save the **Bot Token** (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

#### Step 2: Get Your Chat ID
1. Search for `@userinfobot` in Telegram
2. Start a chat and send any message
3. The bot will reply with your **Chat ID** (looks like: `123456789`)

#### Step 3: Configure the Script
Edit `simple_trading_strategy.py` and update these lines:
```python
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"  # Replace with your bot token
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"     # Replace with your chat ID
```

## 🎯 How to Run

### Basic Usage (Without Telegram)
```bash
python simple_trading_strategy.py
```

### With Telegram Notifications
1. Configure Telegram settings as described above
2. Run the script:
```bash
python simple_trading_strategy.py
```

## 📊 How the Strategy Works

### Moving Average Crossover Strategy
- **Buy Signal**: When 20-day MA crosses above 50-day MA
- **Sell Signal**: When 20-day MA crosses below 50-day MA
- **Trend Analysis**: Bullish when short MA > long MA, Bearish otherwise

### Default Watchlist (Top 10 NSE Stocks)
1. Reliance Industries (RELIANCE)
2. Tata Consultancy Services (TCS)
3. HDFC Bank (HDFCBANK)
4. Infosys (INFY)
5. Hindustan Unilever (HINDUNILVR)
6. ICICI Bank (ICICIBANK)
7. Kotak Mahindra Bank (KOTAKBANK)
8. Bharti Airtel (BHARTIARTL)
9. ITC Limited (ITC)
10. State Bank of India (SBIN)

## 📱 Sample Output

### Console Output
```
🚀 Simple Trading Strategy Started
==================================================
INFO - Analyzing RELIANCE.NS
INFO - Analyzing TCS.NS
INFO - Signal generated for TCS.NS: BUY

📊 Analysis Complete!
📈 Signals Generated: 1

📊 Market Summary

🎯 Overall Sentiment: BULLISH
📈 Bullish Stocks: 3/5 (60.0%)
📉 Bearish Stocks: 2/5 (40.0%)

⏰ Updated: 30-08-2025 11:45:30

Based on MA20/MA50 crossover analysis

✅ Strategy execution completed successfully!
```

### Telegram Notification
```
🟢 BUY SIGNAL 🟢

📈 Stock: TCS
💰 Price: ₹3,245.50
📊 Trend: BULLISH

📉 MA20: ₹3,198.75
📈 MA50: ₹3,156.20

⏰ Time: 30-08-2025 11:45:30

⚠️ This is not financial advice. Please do your own research.
```

## ⚙️ Customization

### Modify Watchlist
Edit the `watchlist` array in `simple_trading_strategy.py`:
```python
self.watchlist = [
    'RELIANCE.NS',
    'TCS.NS',
    'YOUR_STOCK.NS',  # Add your preferred stocks
]
```

### Change Strategy Parameters
Modify these values in the `__init__` method:
```python
self.short_window = 20  # Short-term MA (default: 20 days)
self.long_window = 50   # Long-term MA (default: 50 days)
```

### Adjust Data Period
Change the data period in `get_stock_data` method:
```python
data = stock.history(period='3mo')  # Options: '1mo', '3mo', '6mo', '1y'
```

## 🔄 Running Continuously

To run the strategy continuously (every 5 minutes during market hours):

### Option 1: Simple Loop (Add to main function)
```python
import schedule
import time

def job():
    strategy.run_analysis()

# Run every 5 minutes during market hours (9:15 AM - 3:30 PM IST)
schedule.every(5).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
```

### Option 2: Using Cron (Linux/Mac)
Add to crontab:
```bash
# Run every 5 minutes during market hours (Monday to Friday)
*/5 9-15 * * 1-5 cd /path/to/strategy && python simple_trading_strategy.py
```

## 📈 Understanding the Signals

### Buy Signal Conditions
- 20-day MA crosses above 50-day MA
- Indicates potential upward momentum
- Consider as entry point for long positions

### Sell Signal Conditions
- 20-day MA crosses below 50-day MA
- Indicates potential downward momentum
- Consider as exit point or short entry

### Market Sentiment
- **Bullish**: >60% of analyzed stocks in uptrend
- **Bearish**: <40% of analyzed stocks in uptrend
- **Neutral**: 40-60% of analyzed stocks in uptrend

## ⚠️ Important Disclaimers

1. **Not Financial Advice**: This is an educational tool, not investment advice
2. **Past Performance**: Historical results don't guarantee future performance
3. **Market Risks**: All investments carry risk of loss
4. **Do Your Research**: Always conduct your own analysis before trading
5. **Test First**: Backtest the strategy before using real money

## 🐛 Troubleshooting

### Common Issues

#### "Module not found" Error
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### "No data found" for a stock
- Check if the stock symbol is correct (must end with .NS for NSE)
- Verify internet connection
- Some stocks might be delisted or suspended

#### Telegram messages not sending
- Verify bot token and chat ID are correct
- Check if bot is blocked or chat is deleted
- Ensure internet connection is stable

#### Rate limiting errors
- The script includes delays between API calls
- If you get rate limited, increase the `time.sleep(1)` value

## 🔧 Advanced Features (Future Enhancements)

### Planned Features
- [ ] Multiple technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Sentiment analysis from news and social media
- [ ] Backtesting with historical data
- [ ] Portfolio management and position sizing
- [ ] Risk management (stop-loss, take-profit)
- [ ] Web dashboard for monitoring
- [ ] Database storage for historical signals
- [ ] Email notifications
- [ ] Multi-timeframe analysis

### Contributing
Feel free to fork this project and submit pull requests for improvements!

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify your Python version (3.8+)
4. Check internet connectivity

## 📄 License

This project is for educational purposes only. Use at your own risk.

---

**Happy Trading! 📈**

*Remember: The best strategy is the one you understand and can stick to consistently.*

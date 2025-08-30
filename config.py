"""
Configuration file for Simple Trading Strategy
Modify these settings according to your preferences
"""

# Telegram Configuration
# Get bot token from @BotFather on Telegram
# Get chat ID from @userinfobot on Telegram
TELEGRAM_BOT_TOKEN = None  # Example: "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
TELEGRAM_CHAT_ID = None    # Example: "123456789"

# Strategy Parameters
SHORT_WINDOW = 20    # Short-term moving average period (days)
LONG_WINDOW = 50     # Long-term moving average period (days)
DATA_PERIOD = '3mo'  # Historical data period ('1mo', '3mo', '6mo', '1y')

# Indian Stock Watchlist (NSE symbols)
# You can modify this list to include your preferred stocks
WATCHLIST = [
    # Large Cap Stocks (Default)
    'RELIANCE.NS',      # Reliance Industries
    'TCS.NS',           # Tata Consultancy Services
    'HDFCBANK.NS',      # HDFC Bank
    'INFY.NS',          # Infosys
    'HINDUNILVR.NS',    # Hindustan Unilever
    'ICICIBANK.NS',     # ICICI Bank
    'KOTAKBANK.NS',     # Kotak Mahindra Bank
    'BHARTIARTL.NS',    # Bharti Airtel
    'ITC.NS',           # ITC Limited
    'SBIN.NS',          # State Bank of India
    
    # Additional Popular Stocks (Uncomment to add)
    # 'WIPRO.NS',         # Wipro
    # 'ASIANPAINT.NS',    # Asian Paints
    # 'MARUTI.NS',        # Maruti Suzuki
    # 'TITAN.NS',         # Titan Company
    # 'NESTLEIND.NS',     # Nestle India
    # 'HCLTECH.NS',       # HCL Technologies
    # 'BAJFINANCE.NS',    # Bajaj Finance
    # 'ULTRACEMCO.NS',    # UltraTech Cement
    # 'POWERGRID.NS',     # Power Grid Corporation
    # 'NTPC.NS',          # NTPC Limited
    # 'LT.NS',            # Larsen & Toubro
    # 'AXISBANK.NS',      # Axis Bank
    # 'TECHM.NS',         # Tech Mahindra
    # 'SUNPHARMA.NS',     # Sun Pharmaceutical
    # 'ONGC.NS',          # Oil & Natural Gas Corporation
]

# Market Timing (IST) - For future scheduling features
MARKET_HOURS = {
    'pre_market_start': '09:00',
    'market_open': '09:15',
    'market_close': '15:30',
    'post_market_end': '16:00',
}

# Notification Settings
SEND_MARKET_SUMMARY = True     # Send daily market summary
SEND_SIGNALS_ONLY = False      # Only send buy/sell signals (no market updates)
INCLUDE_PRICE_ALERTS = True    # Include current price in notifications

# Risk Management (For future implementation)
MAX_POSITIONS = 5              # Maximum number of positions to track
STOP_LOSS_PERCENT = 5.0        # Stop loss percentage
TAKE_PROFIT_PERCENT = 10.0     # Take profit percentage

# Data Update Settings
UPDATE_DELAY = 1               # Delay between stock analysis (seconds)
RETRY_ATTEMPTS = 3             # Number of retry attempts for failed requests

# Performance Optimization Settings
BATCH_SIZE = 10                # Number of stocks to process in each batch
BATCH_DELAY = 2                # Delay between batches (seconds)
ENABLE_PARALLEL_PROCESSING = True  # Enable concurrent processing within batches
MAX_WORKERS = 3                # Maximum number of concurrent workers
CACHE_DURATION = 300           # Cache results for 5 minutes (seconds)

# Stock Set Configuration
ACTIVE_STOCK_SET = 'DEFAULT'   # Options: 'DEFAULT', 'NIFTY_50', 'BANKING', 'IT', 'CUSTOM'
ROTATION_ENABLED = True        # Rotate through different stock sets
ROTATION_INTERVAL = 3          # Rotate every 3 update cycles

# Logging Settings
LOG_LEVEL = 'INFO'             # Logging level: DEBUG, INFO, WARNING, ERROR
ENABLE_FILE_LOGGING = False    # Save logs to file (trading_strategy.log)

# Advanced Settings (For experienced users)
ENABLE_EXTENDED_HOURS = False  # Analyze during pre/post market hours
INCLUDE_VOLUME_ANALYSIS = False # Include volume-based signals
ENABLE_MULTI_TIMEFRAME = False # Analyze multiple timeframes

# Custom Stock Lists (Examples)
NIFTY_50_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
    'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
    'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS', 'NESTLEIND.NS', 'HCLTECH.NS',
    'BAJFINANCE.NS', 'ULTRACEMCO.NS', 'POWERGRID.NS', 'NTPC.NS', 'LT.NS',
    'AXISBANK.NS', 'TECHM.NS', 'SUNPHARMA.NS', 'ONGC.NS', 'COALINDIA.NS',
    'WIPRO.NS', 'TATAMOTORS.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'INDUSINDBK.NS',
    'ADANIENT.NS', 'TATASTEEL.NS', 'BAJAJFINSV.NS', 'HEROMOTOCO.NS', 'CIPLA.NS',
    'BRITANNIA.NS', 'DIVISLAB.NS', 'EICHERMOT.NS', 'DRREDDY.NS', 'APOLLOHOSP.NS',
    'BPCL.NS', 'GRASIM.NS', 'SHRIRAMFIN.NS', 'TRENT.NS', 'ADANIPORTS.NS',
    'BAJAJ-AUTO.NS', 'LTIM.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'M&M.NS'
]

BANKING_STOCKS = [
    'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS',
    'INDUSINDBK.NS', 'BANDHANBNK.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'PNB.NS'
]

IT_STOCKS = [
    'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS',
    'LTIM.NS', 'MPHASIS.NS', 'PERSISTENT.NS', 'COFORGE.NS', 'MINDTREE.NS'
]

# Usage Examples:
# To use Nifty 50 stocks: WATCHLIST = NIFTY_50_STOCKS
# To use only banking stocks: WATCHLIST = BANKING_STOCKS
# To use only IT stocks: WATCHLIST = IT_STOCKS

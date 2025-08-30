"""
Optimized Enhanced Trading Strategy Dashboard with Batch Processing
Performance optimizations for handling larger stock universes
"""

import logging
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, jsonify
import config
from enhanced_trading_strategy import EnhancedTradingStrategy

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = Flask(__name__)

class OptimizedEnhancedDashboard:
    def __init__(self):
        self.strategy = EnhancedTradingStrategy()
        self.market_data = {}
        self.last_updated = None
        self.current_batch = 0
        self.rotation_cycle = 0
        self.data_cache = {}
        self.cache_timestamps = {}
        
        # Stock set management
        self.stock_sets = {
            'DEFAULT': config.WATCHLIST,
            'NIFTY_50': config.NIFTY_50_STOCKS,
            'BANKING': config.BANKING_STOCKS,
            'IT': config.IT_STOCKS
        }
        
        self.current_stock_set = self._get_current_stock_set()
        
        # Start background data update
        self.update_thread = threading.Thread(target=self._background_update, daemon=True)
        self.update_thread.start()
        
        logging.info("🚀 Optimized Enhanced Trading Dashboard Starting...")
        logging.info(f"📊 Dashboard will be available at: http://localhost:5001")
        logging.info(f"🔄 Market data updates with batch processing")
        logging.info(f"📈 Current stock set: {config.ACTIVE_STOCK_SET} ({len(self.current_stock_set)} stocks)")
        logging.info(f"⚡ Batch size: {config.BATCH_SIZE}, Max workers: {config.MAX_WORKERS}")

    def _get_current_stock_set(self):
        """Get the current active stock set based on configuration"""
        if config.ROTATION_ENABLED:
            # Rotate through different stock sets
            sets = ['DEFAULT', 'BANKING', 'IT']
            current_set = sets[self.rotation_cycle % len(sets)]
            return self.stock_sets.get(current_set, config.WATCHLIST)
        else:
            return self.stock_sets.get(config.ACTIVE_STOCK_SET, config.WATCHLIST)

    def _is_cache_valid(self, symbol):
        """Check if cached data is still valid"""
        if symbol not in self.cache_timestamps:
            return False
        
        cache_age = time.time() - self.cache_timestamps[symbol]
        return cache_age < config.CACHE_DURATION

    def _analyze_stock_batch(self, stock_batch):
        """Analyze a batch of stocks with optional parallel processing"""
        batch_results = []
        
        if config.ENABLE_PARALLEL_PROCESSING and len(stock_batch) > 1:
            # Parallel processing within batch
            with ThreadPoolExecutor(max_workers=min(config.MAX_WORKERS, len(stock_batch))) as executor:
                future_to_symbol = {
                    executor.submit(self._analyze_single_stock, symbol): symbol 
                    for symbol in stock_batch
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result(timeout=30)  # 30 second timeout
                        if result:
                            batch_results.append(result)
                    except Exception as e:
                        logging.error(f"Error analyzing {symbol}: {e}")
        else:
            # Sequential processing
            for symbol in stock_batch:
                try:
                    result = self._analyze_single_stock(symbol)
                    if result:
                        batch_results.append(result)
                except Exception as e:
                    logging.error(f"Error analyzing {symbol}: {e}")
        
        return batch_results

    def _analyze_single_stock(self, symbol):
        """Analyze a single stock with caching"""
        # Check cache first
        if self._is_cache_valid(symbol) and symbol in self.data_cache:
            logging.debug(f"Using cached data for {symbol}")
            return self.data_cache[symbol]
        
        # Analyze stock
        logging.info(f"Analyzing {symbol}")
        result = self.strategy.analyze_stock(symbol)
        
        # Cache the result
        if result:
            self.data_cache[symbol] = result
            self.cache_timestamps[symbol] = time.time()
        
        return result

    def _create_batches(self, stock_list):
        """Split stock list into batches"""
        batches = []
        for i in range(0, len(stock_list), config.BATCH_SIZE):
            batch = stock_list[i:i + config.BATCH_SIZE]
            batches.append(batch)
        return batches

    def _background_update(self):
        """Background thread for updating market data with batch processing"""
        while True:
            try:
                start_time = time.time()
                logging.info("Updating optimized enhanced market data...")
                
                # Rotate stock set if enabled
                if config.ROTATION_ENABLED:
                    if self.rotation_cycle % config.ROTATION_INTERVAL == 0:
                        self.current_stock_set = self._get_current_stock_set()
                        logging.info(f"Rotated to stock set: {list(self.stock_sets.keys())[self.rotation_cycle % len(self.stock_sets)]}")
                
                # Create batches
                batches = self._create_batches(self.current_stock_set)
                all_results = []
                
                logging.info(f"Processing {len(self.current_stock_set)} stocks in {len(batches)} batches")
                
                # Process each batch
                for i, batch in enumerate(batches):
                    batch_start = time.time()
                    logging.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} stocks)")
                    
                    batch_results = self._analyze_stock_batch(batch)
                    all_results.extend(batch_results)
                    
                    batch_time = time.time() - batch_start
                    logging.info(f"Batch {i+1} completed in {batch_time:.1f}s ({len(batch_results)} results)")
                    
                    # Delay between batches (except for the last one)
                    if i < len(batches) - 1:
                        time.sleep(config.BATCH_DELAY)
                
                # Sort by confidence
                all_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                
                # Update market data
                self.market_data = {
                    'stock_analysis': all_results,
                    'market_summary': self._calculate_market_summary(all_results),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'processing_stats': {
                        'total_stocks': len(self.current_stock_set),
                        'successful_analysis': len(all_results),
                        'batches_processed': len(batches),
                        'total_time': round(time.time() - start_time, 1),
                        'current_stock_set': config.ACTIVE_STOCK_SET if not config.ROTATION_ENABLED else f"Rotating (cycle {self.rotation_cycle})"
                    }
                }
                
                self.last_updated = datetime.now()
                total_time = time.time() - start_time
                
                logging.info(f"Optimized enhanced market data updated in {total_time:.1f}s")
                logging.info(f"Successfully analyzed {len(all_results)}/{len(self.current_stock_set)} stocks")
                
                # Increment rotation cycle
                self.rotation_cycle += 1
                
                # Wait before next update (5 minutes)
                time.sleep(300)
                
            except Exception as e:
                logging.error(f"Error in background update: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    def _calculate_market_summary(self, stock_analysis):
        """Calculate market summary statistics"""
        if not stock_analysis:
            return {}
        
        trends = [stock.get('trend', 'NEUTRAL') for stock in stock_analysis]
        signals = [stock.get('current_signal', 0) for stock in stock_analysis]
        signal_strengths = [abs(stock.get('signal_strength', 0)) for stock in stock_analysis]
        high_volume_count = sum(1 for stock in stock_analysis 
                               if stock.get('volume_info', {}).get('is_high_volume', False))
        
        # Count trends
        trend_counts = {
            'strong_bullish': trends.count('STRONG_BULLISH'),
            'bullish': trends.count('BULLISH'),
            'neutral': trends.count('NEUTRAL'),
            'bearish': trends.count('BEARISH'),
            'strong_bearish': trends.count('STRONG_BEARISH')
        }
        
        # Determine overall sentiment
        bullish_total = trend_counts['strong_bullish'] + trend_counts['bullish']
        bearish_total = trend_counts['strong_bearish'] + trend_counts['bearish']
        
        if bullish_total > bearish_total:
            overall_sentiment = 'BULLISH'
        elif bearish_total > bullish_total:
            overall_sentiment = 'BEARISH'
        else:
            overall_sentiment = 'NEUTRAL'
        
        # Count signals
        buy_signals = sum(1 for s in signals if s > 0)
        sell_signals = sum(1 for s in signals if s < 0)
        strong_signals = sum(1 for s in signal_strengths if s >= 2)
        
        return {
            'overall_sentiment': overall_sentiment,
            'total_stocks': len(stock_analysis),
            'signals_count': buy_signals + sell_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'strong_signals': strong_signals,
            'high_volume_stocks': high_volume_count,
            **trend_counts
        }

# Initialize dashboard
dashboard = OptimizedEnhancedDashboard()

@app.route('/')
def index():
    """Serve the enhanced dashboard"""
    return render_template('enhanced_dashboard.html')

@app.route('/api/enhanced-market-data')
def get_enhanced_market_data():
    """API endpoint for enhanced market data"""
    return jsonify(dashboard.market_data)

@app.route('/api/performance-stats')
def get_performance_stats():
    """API endpoint for performance statistics"""
    stats = dashboard.market_data.get('processing_stats', {})
    stats.update({
        'cache_size': len(dashboard.data_cache),
        'batch_size': config.BATCH_SIZE,
        'max_workers': config.MAX_WORKERS,
        'rotation_enabled': config.ROTATION_ENABLED,
        'current_cycle': dashboard.rotation_cycle
    })
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)

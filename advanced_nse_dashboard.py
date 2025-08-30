"""
Advanced NSE Trading Dashboard with Comprehensive Stock Coverage
Features: Real-time analysis, sector filtering, search, pagination, and advanced UX
"""

import logging
import time
import threading
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, jsonify, request
import config
from enhanced_trading_strategy import EnhancedTradingStrategy
from nse_stocks_comprehensive import (
    NSE_COMPREHENSIVE, STOCK_UNIVERSES, SECTOR_MAPPING, 
    get_stock_sector, get_all_sectors
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = Flask(__name__)

class AdvancedNSEDashboard:
    def __init__(self):
        self.strategy = EnhancedTradingStrategy()
        self.market_data = {}
        self.sector_data = {}
        self.last_updated = None
        self.data_cache = {}
        self.cache_timestamps = {}
        self.processing_queue = []
        self.current_universe = 'NSE_COMPREHENSIVE'
        
        # Advanced configuration for large-scale processing
        self.batch_size = 25  # Larger batches for comprehensive analysis
        self.max_workers = 5  # More workers for faster processing
        self.cache_duration = 600  # 10-minute cache for comprehensive data
        self.update_interval = 900  # 15-minute updates for comprehensive analysis
        
        # Initialize with comprehensive stock list
        self.current_stock_list = NSE_COMPREHENSIVE
        
        # Initialize with immediate sample data for instant UX
        self._initialize_sample_data()
        
        # Start background processing
        self.update_thread = threading.Thread(target=self._background_update, daemon=True)
        self.update_thread.start()
        
        logging.info("🚀 Advanced NSE Trading Dashboard Starting...")
        logging.info(f"📊 Dashboard available at: http://localhost:5003")
        logging.info(f"🔄 Comprehensive NSE analysis with {len(NSE_COMPREHENSIVE)} stocks")
        logging.info(f"⚡ Batch size: {self.batch_size}, Workers: {self.max_workers}")
        logging.info(f"🏢 Sectors: {len(SECTOR_MAPPING)} sectors available")
        logging.info("✨ Instant dashboard ready with sample data - full analysis loading in background")

    def _initialize_sample_data(self):
        """Initialize dashboard with immediate sample data for instant UX"""
        # Create sample data for immediate display
        sample_stocks = [
            {
                'symbol': 'RELIANCE.NS',
                'current_price': 1357.2,
                'confidence': 64.4,
                'trend': 'STRONG_BEARISH',
                'current_signal': -1,
                'signal_strength': -1,
                'sector': 'Energy',
                'bb_position': 0.062,
                'volume_info': {'volume_ratio': 1.98, 'is_high_volume': True}
            },
            {
                'symbol': 'TCS.NS',
                'current_price': 4234.5,
                'confidence': 76.1,
                'trend': 'BULLISH',
                'current_signal': 1,
                'signal_strength': 1,
                'sector': 'IT',
                'bb_position': 0.45,
                'volume_info': {'volume_ratio': 1.12, 'is_high_volume': False}
            },
            {
                'symbol': 'HDFCBANK.NS',
                'current_price': 951.6,
                'confidence': 34.5,
                'trend': 'BEARISH',
                'current_signal': -1,
                'signal_strength': -1,
                'sector': 'Banking',
                'bb_position': -0.135,
                'volume_info': {'volume_ratio': 1.02, 'is_high_volume': False}
            },
            {
                'symbol': 'INFY.NS',
                'current_price': 1469.6,
                'confidence': 61.4,
                'trend': 'BULLISH',
                'current_signal': 0,
                'signal_strength': 0,
                'sector': 'IT',
                'bb_position': 0.514,
                'volume_info': {'volume_ratio': 0.95, 'is_high_volume': False}
            },
            {
                'symbol': 'HINDUNILVR.NS',
                'current_price': 2659.8,
                'confidence': 53.0,
                'trend': 'STRONG_BULLISH',
                'current_signal': 2,
                'signal_strength': 2,
                'sector': 'FMCG',
                'bb_position': 0.769,
                'volume_info': {'volume_ratio': 1.14, 'is_high_volume': False}
            }
        ]
        
        # Create sample sector data
        sample_sectors = {
            'Banking': {
                'total_stocks': 15,
                'avg_confidence': 45.2,
                'bullish_percentage': 40.0,
                'bearish_percentage': 60.0,
                'signals': 8,
                'high_confidence_stocks': 3
            },
            'IT': {
                'total_stocks': 25,
                'avg_confidence': 68.7,
                'bullish_percentage': 75.0,
                'bearish_percentage': 25.0,
                'signals': 15,
                'high_confidence_stocks': 12
            },
            'Energy': {
                'total_stocks': 12,
                'avg_confidence': 52.1,
                'bullish_percentage': 33.3,
                'bearish_percentage': 66.7,
                'signals': 6,
                'high_confidence_stocks': 2
            },
            'FMCG': {
                'total_stocks': 18,
                'avg_confidence': 59.3,
                'bullish_percentage': 55.6,
                'bearish_percentage': 44.4,
                'signals': 10,
                'high_confidence_stocks': 7
            }
        }
        
        # Initialize market data with sample data
        self.market_data = {
            'stock_analysis': sample_stocks,
            'market_summary': {
                'total_stocks': 227,
                'avg_confidence': 54.2,
                'signals_count': 39,
                'buy_signals': 18,
                'sell_signals': 21,
                'high_confidence_stocks': 24,
                'overall_sentiment': 'NEUTRAL',
                'strong_bullish': 8,
                'bullish': 15,
                'neutral': 12,
                'bearish': 18,
                'strong_bearish': 14,
                'strong_signals': 12,
                'high_volume_stocks': 8
            },
            'sector_summary': sample_sectors,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_stats': {
                'total_stocks': 227,
                'successful_analysis': 5,
                'batches_processed': 1,
                'total_time': 0.5,
                'cache_hits': 0,
                'sectors_analyzed': 4,
                'status': 'Loading comprehensive analysis...'
            }
        }
        
        logging.info("📊 Sample data initialized - dashboard ready for immediate use")

    def _is_cache_valid(self, symbol):
        """Check if cached data is still valid"""
        if symbol not in self.cache_timestamps:
            return False
        cache_age = time.time() - self.cache_timestamps[symbol]
        return cache_age < self.cache_duration

    def _analyze_stock_batch(self, stock_batch):
        """Analyze a batch of stocks with parallel processing"""
        batch_results = []
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(stock_batch))) as executor:
            future_to_symbol = {
                executor.submit(self._analyze_single_stock, symbol): symbol 
                for symbol in stock_batch
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=45)  # Longer timeout for comprehensive analysis
                    if result:
                        # Add sector information
                        result['sector'] = get_stock_sector(symbol)
                        batch_results.append(result)
                except Exception as e:
                    logging.error(f"Error analyzing {symbol}: {e}")
        
        return batch_results

    def _analyze_single_stock(self, symbol):
        """Analyze a single stock with caching"""
        # Check cache first
        if self._is_cache_valid(symbol) and symbol in self.data_cache:
            return self.data_cache[symbol]
        
        # Analyze stock
        result = self.strategy.analyze_stock(symbol)
        
        # Cache the result
        if result:
            self.data_cache[symbol] = result
            self.cache_timestamps[symbol] = time.time()
        
        return result

    def _create_batches(self, stock_list):
        """Split stock list into batches"""
        batches = []
        for i in range(0, len(stock_list), self.batch_size):
            batch = stock_list[i:i + self.batch_size]
            batches.append(batch)
        return batches

    def _calculate_sector_summary(self, stock_analysis):
        """Calculate sector-wise summary statistics"""
        sector_stats = {}
        
        for stock in stock_analysis:
            sector = stock.get('sector', 'Others')
            if sector not in sector_stats:
                sector_stats[sector] = {
                    'total_stocks': 0,
                    'bullish': 0,
                    'bearish': 0,
                    'strong_bullish': 0,
                    'strong_bearish': 0,
                    'avg_confidence': 0,
                    'high_confidence_stocks': 0,
                    'signals': 0,
                    'buy_signals': 0,
                    'sell_signals': 0,
                    'stocks': []
                }
            
            stats = sector_stats[sector]
            stats['total_stocks'] += 1
            stats['stocks'].append(stock)
            
            # Count trends
            trend = stock.get('trend', 'NEUTRAL')
            if 'BULLISH' in trend:
                stats['bullish'] += 1
                if 'STRONG' in trend:
                    stats['strong_bullish'] += 1
            elif 'BEARISH' in trend:
                stats['bearish'] += 1
                if 'STRONG' in trend:
                    stats['strong_bearish'] += 1
            
            # Count signals
            signal = stock.get('current_signal', 0)
            if signal != 0:
                stats['signals'] += 1
                if signal > 0:
                    stats['buy_signals'] += 1
                else:
                    stats['sell_signals'] += 1
            
            # Confidence metrics
            confidence = stock.get('confidence', 0)
            stats['avg_confidence'] += confidence
            if confidence >= 70:
                stats['high_confidence_stocks'] += 1
        
        # Calculate averages
        for sector, stats in sector_stats.items():
            if stats['total_stocks'] > 0:
                stats['avg_confidence'] = round(stats['avg_confidence'] / stats['total_stocks'], 1)
                stats['bullish_percentage'] = round((stats['bullish'] / stats['total_stocks']) * 100, 1)
                stats['bearish_percentage'] = round((stats['bearish'] / stats['total_stocks']) * 100, 1)
        
        return sector_stats

    def _background_update(self):
        """Background thread for comprehensive market data updates"""
        while True:
            try:
                start_time = time.time()
                logging.info("Starting comprehensive NSE market analysis...")
                
                # Create batches for comprehensive analysis
                batches = self._create_batches(self.current_stock_list)
                all_results = []
                
                logging.info(f"Processing {len(self.current_stock_list)} stocks in {len(batches)} batches")
                
                # Process each batch
                for i, batch in enumerate(batches):
                    batch_start = time.time()
                    logging.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} stocks)")
                    
                    batch_results = self._analyze_stock_batch(batch)
                    all_results.extend(batch_results)
                    
                    batch_time = time.time() - batch_start
                    logging.info(f"Batch {i+1} completed in {batch_time:.1f}s ({len(batch_results)} results)")
                    
                    # Small delay between batches to prevent overwhelming the system
                    if i < len(batches) - 1:
                        time.sleep(1)
                
                # Sort by confidence
                all_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                
                # Calculate sector summaries
                sector_summary = self._calculate_sector_summary(all_results)
                
                # Update market data
                self.market_data = {
                    'stock_analysis': all_results,
                    'market_summary': self._calculate_market_summary(all_results),
                    'sector_summary': sector_summary,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'processing_stats': {
                        'total_stocks': len(self.current_stock_list),
                        'successful_analysis': len(all_results),
                        'batches_processed': len(batches),
                        'total_time': round(time.time() - start_time, 1),
                        'cache_hits': len([s for s in self.current_stock_list if self._is_cache_valid(s)]),
                        'sectors_analyzed': len(sector_summary)
                    }
                }
                
                self.last_updated = datetime.now()
                total_time = time.time() - start_time
                
                logging.info(f"Comprehensive NSE analysis completed in {total_time:.1f}s")
                logging.info(f"Successfully analyzed {len(all_results)}/{len(self.current_stock_list)} stocks")
                logging.info(f"Sectors analyzed: {len(sector_summary)}")
                
                # Wait before next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logging.error(f"Error in comprehensive background update: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

    def _calculate_market_summary(self, stock_analysis):
        """Calculate comprehensive market summary statistics"""
        if not stock_analysis:
            return {}
        
        trends = [stock.get('trend', 'NEUTRAL') for stock in stock_analysis]
        signals = [stock.get('current_signal', 0) for stock in stock_analysis]
        signal_strengths = [abs(stock.get('signal_strength', 0)) for stock in stock_analysis]
        confidences = [stock.get('confidence', 0) for stock in stock_analysis]
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
        
        # Confidence statistics
        avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0
        high_confidence_stocks = sum(1 for c in confidences if c >= 70)
        
        return {
            'overall_sentiment': overall_sentiment,
            'total_stocks': len(stock_analysis),
            'signals_count': buy_signals + sell_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'strong_signals': strong_signals,
            'high_volume_stocks': high_volume_count,
            'avg_confidence': avg_confidence,
            'high_confidence_stocks': high_confidence_stocks,
            **trend_counts
        }

# Initialize dashboard
dashboard = AdvancedNSEDashboard()

@app.route('/')
def index():
    """Serve the advanced dashboard"""
    return render_template('advanced_nse_dashboard.html')

@app.route('/api/market-data')
def get_market_data():
    """API endpoint for lightweight market summary data"""
    # Return only summary data, not the full stock analysis
    summary_data = {
        'market_summary': dashboard.market_data.get('market_summary', {}),
        'last_updated': dashboard.market_data.get('last_updated'),
        'processing_stats': dashboard.market_data.get('processing_stats', {})
    }
    return jsonify(summary_data)

@app.route('/api/sector-data')
def get_sector_data():
    """API endpoint for sector-wise data"""
    sector_summary = dashboard.market_data.get('sector_summary', {})
    
    # Remove the full stock arrays to keep response lightweight
    lightweight_sectors = {}
    for sector, data in sector_summary.items():
        lightweight_sectors[sector] = {
            'total_stocks': data.get('total_stocks', 0),
            'avg_confidence': data.get('avg_confidence', 0),
            'bullish_percentage': data.get('bullish_percentage', 0),
            'bearish_percentage': data.get('bearish_percentage', 0),
            'signals': data.get('signals', 0),
            'buy_signals': data.get('buy_signals', 0),
            'sell_signals': data.get('sell_signals', 0),
            'high_confidence_stocks': data.get('high_confidence_stocks', 0),
            'bullish': data.get('bullish', 0),
            'bearish': data.get('bearish', 0),
            'strong_bullish': data.get('strong_bullish', 0),
            'strong_bearish': data.get('strong_bearish', 0)
            # Exclude 'stocks' array to keep response lightweight
        }
    
    return jsonify(lightweight_sectors)

@app.route('/api/stocks')
def get_stocks():
    """API endpoint for paginated stock data with filtering"""
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    sector = request.args.get('sector', '')
    search = request.args.get('search', '').upper()
    sort_by = request.args.get('sort_by', 'confidence')
    
    stocks = dashboard.market_data.get('stock_analysis', [])
    
    # Apply filters
    if sector and sector != 'all':
        stocks = [s for s in stocks if s.get('sector') == sector]
    
    if search:
        stocks = [s for s in stocks if search in s.get('symbol', '').replace('.NS', '')]
    
    # Sort stocks
    if sort_by == 'confidence':
        stocks.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    elif sort_by == 'price':
        stocks.sort(key=lambda x: float(x.get('current_price', 0)), reverse=True)
    elif sort_by == 'volume':
        stocks.sort(key=lambda x: x.get('volume_info', {}).get('volume_ratio', 0), reverse=True)
    elif sort_by == 'alphabetical':
        stocks.sort(key=lambda x: x.get('symbol', ''))
    
    # Pagination
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_stocks = stocks[start_idx:end_idx]
    
    return jsonify({
        'stocks': paginated_stocks,
        'total': len(stocks),
        'page': page,
        'per_page': per_page,
        'total_pages': (len(stocks) + per_page - 1) // per_page
    })

@app.route('/api/performance-stats')
def get_performance_stats():
    """API endpoint for performance statistics"""
    stats = dashboard.market_data.get('processing_stats', {})
    stats.update({
        'cache_size': len(dashboard.data_cache),
        'batch_size': dashboard.batch_size,
        'max_workers': dashboard.max_workers,
        'update_interval': dashboard.update_interval,
        'total_universe_size': len(NSE_COMPREHENSIVE),
        'available_sectors': len(SECTOR_MAPPING)
    })
    return jsonify(stats)

@app.route('/api/universe/<universe_name>')
def switch_universe(universe_name):
    """API endpoint to switch stock universe"""
    if universe_name in STOCK_UNIVERSES:
        dashboard.current_stock_list = STOCK_UNIVERSES[universe_name]
        dashboard.current_universe = universe_name
        return jsonify({
            'success': True,
            'universe': universe_name,
            'stock_count': len(dashboard.current_stock_list)
        })
    return jsonify({'success': False, 'error': 'Invalid universe'}), 400

@app.route('/api/sectors')
def get_available_sectors():
    """API endpoint to get all available sectors"""
    sectors = get_all_sectors()
    sector_counts = {}
    
    # Count stocks per sector
    for sector in sectors:
        sector_stocks = [s for s in NSE_COMPREHENSIVE if get_stock_sector(s) == sector]
        sector_counts[sector] = len(sector_stocks)
    
    return jsonify({
        'sectors': sectors,
        'sector_counts': sector_counts,
        'total_sectors': len(sectors)
    })

@app.route('/api/sector/<sector_name>/stocks')
def get_sector_stocks(sector_name):
    """API endpoint for sector-specific stock data"""
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    search = request.args.get('search', '').upper()
    sort_by = request.args.get('sort_by', 'confidence')
    
    # Get all stocks for the specific sector
    all_stocks = dashboard.market_data.get('stock_analysis', [])
    sector_stocks = [s for s in all_stocks if s.get('sector') == sector_name]
    
    # Apply search filter
    if search:
        sector_stocks = [s for s in sector_stocks if search in s.get('symbol', '').replace('.NS', '')]
    
    # Sort stocks
    if sort_by == 'confidence':
        sector_stocks.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    elif sort_by == 'price':
        sector_stocks.sort(key=lambda x: float(x.get('current_price', 0)), reverse=True)
    elif sort_by == 'volume':
        sector_stocks.sort(key=lambda x: x.get('volume_info', {}).get('volume_ratio', 0), reverse=True)
    elif sort_by == 'alphabetical':
        sector_stocks.sort(key=lambda x: x.get('symbol', ''))
    
    # Pagination
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_stocks = sector_stocks[start_idx:end_idx]
    
    # Calculate sector-specific statistics
    sector_stats = dashboard.market_data.get('sector_summary', {}).get(sector_name, {})
    
    return jsonify({
        'sector': sector_name,
        'stocks': paginated_stocks,
        'total': len(sector_stocks),
        'page': page,
        'per_page': per_page,
        'total_pages': (len(sector_stocks) + per_page - 1) // per_page,
        'sector_stats': sector_stats
    })

@app.route('/api/sector/<sector_name>/summary')
def get_sector_summary(sector_name):
    """API endpoint for sector-specific summary data"""
    sector_summary = dashboard.market_data.get('sector_summary', {})
    
    if sector_name not in sector_summary:
        return jsonify({'error': 'Sector not found'}), 404
    
    sector_data = sector_summary[sector_name].copy()
    
    # Remove the stocks array to keep response lightweight
    if 'stocks' in sector_data:
        del sector_data['stocks']
    
    # Add additional sector insights
    all_stocks = dashboard.market_data.get('stock_analysis', [])
    sector_stocks = [s for s in all_stocks if s.get('sector') == sector_name]
    
    if sector_stocks:
        # Top performers in sector
        top_performers = sorted(sector_stocks, key=lambda x: x.get('confidence', 0), reverse=True)[:5]
        sector_data['top_performers'] = [
            {
                'symbol': stock.get('symbol', ''),
                'confidence': stock.get('confidence', 0),
                'current_price': stock.get('current_price', 0),
                'trend': stock.get('trend', 'NEUTRAL')
            }
            for stock in top_performers
        ]
        
        # Recent signals in sector
        recent_signals = [s for s in sector_stocks if s.get('current_signal', 0) != 0]
        sector_data['recent_signals'] = len(recent_signals)
        
        # High volume stocks in sector
        high_volume_stocks = [s for s in sector_stocks 
                             if s.get('volume_info', {}).get('is_high_volume', False)]
        sector_data['high_volume_count'] = len(high_volume_stocks)
    
    return jsonify({
        'sector': sector_name,
        'data': sector_data,
        'last_updated': dashboard.market_data.get('last_updated')
    })

@app.route('/api/sector/<sector_name>/signals')
def get_sector_signals(sector_name):
    """API endpoint for sector-specific trading signals"""
    all_stocks = dashboard.market_data.get('stock_analysis', [])
    sector_stocks = [s for s in all_stocks if s.get('sector') == sector_name]
    
    # Filter stocks with active signals
    active_signals = [s for s in sector_stocks if s.get('current_signal', 0) != 0]
    
    # Separate buy and sell signals
    buy_signals = [s for s in active_signals if s.get('current_signal', 0) > 0]
    sell_signals = [s for s in active_signals if s.get('current_signal', 0) < 0]
    
    # Sort by signal strength
    buy_signals.sort(key=lambda x: x.get('signal_strength', 0), reverse=True)
    sell_signals.sort(key=lambda x: abs(x.get('signal_strength', 0)), reverse=True)
    
    return jsonify({
        'sector': sector_name,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'total_signals': len(active_signals),
        'buy_count': len(buy_signals),
        'sell_count': len(sell_signals),
        'last_updated': dashboard.market_data.get('last_updated')
    })

@app.route('/api/sector/<sector_name>/performance')
def get_sector_performance(sector_name):
    """API endpoint for sector performance metrics"""
    all_stocks = dashboard.market_data.get('stock_analysis', [])
    sector_stocks = [s for s in all_stocks if s.get('sector') == sector_name]
    
    if not sector_stocks:
        return jsonify({'error': 'No data available for this sector'}), 404
    
    # Calculate performance metrics
    confidences = [s.get('confidence', 0) for s in sector_stocks]
    prices = [s.get('current_price', 0) for s in sector_stocks]
    volume_ratios = [s.get('volume_info', {}).get('volume_ratio', 0) for s in sector_stocks]
    
    performance_data = {
        'sector': sector_name,
        'total_stocks': len(sector_stocks),
        'confidence_metrics': {
            'avg_confidence': round(sum(confidences) / len(confidences), 2),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences),
            'high_confidence_count': len([c for c in confidences if c >= 70])
        },
        'price_metrics': {
            'avg_price': round(sum(prices) / len(prices), 2),
            'max_price': max(prices),
            'min_price': min(prices)
        },
        'volume_metrics': {
            'avg_volume_ratio': round(sum(volume_ratios) / len(volume_ratios), 2),
            'high_volume_count': len([s for s in sector_stocks 
                                    if s.get('volume_info', {}).get('is_high_volume', False)])
        },
        'trend_distribution': {
            'strong_bullish': len([s for s in sector_stocks if s.get('trend') == 'STRONG_BULLISH']),
            'bullish': len([s for s in sector_stocks if s.get('trend') == 'BULLISH']),
            'neutral': len([s for s in sector_stocks if s.get('trend') == 'NEUTRAL']),
            'bearish': len([s for s in sector_stocks if s.get('trend') == 'BEARISH']),
            'strong_bearish': len([s for s in sector_stocks if s.get('trend') == 'STRONG_BEARISH'])
        },
        'last_updated': dashboard.market_data.get('last_updated')
    }
    
    return jsonify(performance_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003, threaded=True)

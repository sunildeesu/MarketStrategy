#!/usr/bin/env python3
"""
Enhanced Trading Strategy Dashboard
A web-based dashboard with advanced technical indicators, target prices, stop loss, and volume analysis
"""

from flask import Flask, render_template, jsonify
import json
from datetime import datetime
import threading
import time
from enhanced_trading_strategy import EnhancedTradingStrategy

app = Flask(__name__)

# Global variables to store enhanced data
enhanced_market_data = {
    'last_updated': None,
    'signals': [],
    'market_summary': {},
    'stock_analysis': []
}

def update_enhanced_market_data():
    """Background task to update enhanced market data periodically"""
    global enhanced_market_data
    
    strategy = EnhancedTradingStrategy()
    
    while True:
        try:
            print("Updating enhanced market data...")
            
            # Run enhanced analysis
            signals = strategy.run_enhanced_analysis()
            
            # Get individual stock analysis
            stock_analysis = []
            for symbol in strategy.watchlist:
                analysis = strategy.analyze_stock(symbol)
                if analysis:
                    stock_analysis.append(analysis)
                time.sleep(1)  # Rate limiting
            
            # Sort by confidence (highest first)
            stock_analysis.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Calculate enhanced market summary
            total_stocks = len(stock_analysis)
            strong_bullish = sum(1 for stock in stock_analysis if stock['trend'] == 'STRONG_BULLISH')
            bullish = sum(1 for stock in stock_analysis if stock['trend'] == 'BULLISH')
            strong_bearish = sum(1 for stock in stock_analysis if stock['trend'] == 'STRONG_BEARISH')
            bearish = sum(1 for stock in stock_analysis if stock['trend'] == 'BEARISH')
            neutral = sum(1 for stock in stock_analysis if stock['trend'] == 'NEUTRAL')
            
            # High volume stocks
            high_volume_stocks = sum(1 for stock in stock_analysis if stock['volume_info']['is_high_volume'])
            
            # Signal strength distribution
            strong_signals = sum(1 for stock in stock_analysis if abs(stock['signal_strength']) >= 2)
            moderate_signals = sum(1 for stock in stock_analysis if abs(stock['signal_strength']) == 1)
            
            # Overall sentiment based on enhanced criteria
            bullish_total = strong_bullish + bullish
            bearish_total = strong_bearish + bearish
            
            if bullish_total > bearish_total:
                overall_sentiment = 'BULLISH'
            elif bearish_total > bullish_total:
                overall_sentiment = 'BEARISH'
            else:
                overall_sentiment = 'NEUTRAL'
            
            enhanced_summary = {
                'total_stocks': total_stocks,
                'strong_bullish': strong_bullish,
                'bullish': bullish,
                'strong_bearish': strong_bearish,
                'bearish': bearish,
                'neutral': neutral,
                'bullish_total': bullish_total,
                'bearish_total': bearish_total,
                'bullish_percentage': (bullish_total / total_stocks * 100) if total_stocks > 0 else 0,
                'bearish_percentage': (bearish_total / total_stocks * 100) if total_stocks > 0 else 0,
                'overall_sentiment': overall_sentiment,
                'signals_count': len(signals),
                'high_volume_stocks': high_volume_stocks,
                'strong_signals': strong_signals,
                'moderate_signals': moderate_signals,
                'buy_signals': len([s for s in signals if s['signal_type'] == 'BUY']),
                'sell_signals': len([s for s in signals if s['signal_type'] == 'SELL'])
            }
            
            # Update global data
            enhanced_market_data.update({
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'signals': signals,
                'market_summary': enhanced_summary,
                'stock_analysis': stock_analysis
            })
            
            print(f"Enhanced market data updated at {enhanced_market_data['last_updated']}")
            
        except Exception as e:
            print(f"Error updating enhanced market data: {e}")
        
        # Wait 5 minutes before next update
        time.sleep(300)

@app.route('/')
def enhanced_dashboard():
    """Enhanced dashboard page"""
    return render_template('enhanced_dashboard.html')

@app.route('/api/enhanced-market-data')
def get_enhanced_market_data():
    """API endpoint to get enhanced market data"""
    return jsonify(enhanced_market_data)

@app.route('/api/enhanced-signals')
def get_enhanced_signals():
    """API endpoint to get enhanced signals"""
    return jsonify({
        'signals': enhanced_market_data['signals'],
        'last_updated': enhanced_market_data['last_updated']
    })

@app.route('/api/enhanced-stocks')
def get_enhanced_stocks():
    """API endpoint to get enhanced stock analysis"""
    return jsonify({
        'stocks': enhanced_market_data['stock_analysis'],
        'last_updated': enhanced_market_data['last_updated']
    })

@app.route('/api/enhanced-summary')
def get_enhanced_summary():
    """API endpoint to get enhanced market summary"""
    return jsonify({
        'summary': enhanced_market_data['market_summary'],
        'last_updated': enhanced_market_data['last_updated']
    })

@app.route('/api/stock-details/<symbol>')
def get_stock_details(symbol):
    """API endpoint to get detailed analysis for a specific stock"""
    stock_data = None
    for stock in enhanced_market_data['stock_analysis']:
        if stock['symbol'] == symbol:
            stock_data = stock
            break
    
    if stock_data:
        return jsonify({
            'stock': stock_data,
            'last_updated': enhanced_market_data['last_updated']
        })
    else:
        return jsonify({'error': 'Stock not found'}), 404

if __name__ == '__main__':
    # Start background data update thread
    data_thread = threading.Thread(target=update_enhanced_market_data, daemon=True)
    data_thread.start()
    
    # Initial data load
    time.sleep(2)  # Give thread time to start
    
    print("🚀 Enhanced Trading Dashboard Starting...")
    print("📊 Enhanced Dashboard will be available at: http://localhost:5001")
    print("🔄 Enhanced market data updates every 5 minutes")
    print("✨ Features: Bollinger Bands, EMAs (9,18,27), Volume Analysis, Target/Stop Loss")
    
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)

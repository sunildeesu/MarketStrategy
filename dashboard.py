#!/usr/bin/env python3
"""
Trading Strategy Dashboard
A web-based dashboard to display trading strategy results and market analysis
"""

from flask import Flask, render_template, jsonify
import json
from datetime import datetime
import threading
import time
from simple_trading_strategy import SimpleTradingStrategy

app = Flask(__name__)

# Global variables to store data
market_data = {
    'last_updated': None,
    'signals': [],
    'market_summary': {},
    'stock_analysis': []
}

def update_market_data():
    """Background task to update market data periodically"""
    global market_data
    
    strategy = SimpleTradingStrategy()
    
    while True:
        try:
            print("Updating market data...")
            
            # Run analysis
            signals = strategy.run_analysis()
            
            # Get individual stock analysis
            stock_analysis = []
            for symbol in strategy.watchlist:
                analysis = strategy.analyze_stock(symbol)
                if analysis:
                    stock_analysis.append(analysis)
                time.sleep(1)  # Rate limiting
            
            # Calculate market summary
            bullish_count = sum(1 for stock in stock_analysis if stock['trend'] == 'BULLISH')
            bearish_count = len(stock_analysis) - bullish_count
            total_stocks = len(stock_analysis)
            
            market_summary = {
                'total_stocks': total_stocks,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'bullish_percentage': (bullish_count / total_stocks * 100) if total_stocks > 0 else 0,
                'bearish_percentage': (bearish_count / total_stocks * 100) if total_stocks > 0 else 0,
                'overall_sentiment': 'BULLISH' if bullish_count > bearish_count else 'BEARISH' if bearish_count > bullish_count else 'NEUTRAL',
                'signals_count': len(signals)
            }
            
            # Update global data
            market_data.update({
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'signals': signals,
                'market_summary': market_summary,
                'stock_analysis': stock_analysis
            })
            
            print(f"Market data updated at {market_data['last_updated']}")
            
        except Exception as e:
            print(f"Error updating market data: {e}")
        
        # Wait 5 minutes before next update
        time.sleep(300)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/market-data')
def get_market_data():
    """API endpoint to get current market data"""
    return jsonify(market_data)

@app.route('/api/signals')
def get_signals():
    """API endpoint to get current signals"""
    return jsonify({
        'signals': market_data['signals'],
        'last_updated': market_data['last_updated']
    })

@app.route('/api/stocks')
def get_stocks():
    """API endpoint to get stock analysis"""
    return jsonify({
        'stocks': market_data['stock_analysis'],
        'last_updated': market_data['last_updated']
    })

@app.route('/api/summary')
def get_summary():
    """API endpoint to get market summary"""
    return jsonify({
        'summary': market_data['market_summary'],
        'last_updated': market_data['last_updated']
    })

if __name__ == '__main__':
    # Start background data update thread
    data_thread = threading.Thread(target=update_market_data, daemon=True)
    data_thread.start()
    
    # Initial data load
    time.sleep(2)  # Give thread time to start
    
    print("🚀 Trading Dashboard Starting...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🔄 Market data updates every 5 minutes")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

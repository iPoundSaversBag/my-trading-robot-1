#!/usr/bin/env python3
"""
Enhanced Live Bot State Updater
Continuously updates the live bot state file with comprehensive details
even when the main trading bot is not running.
"""

import json
import asyncio
import logging
import os
import sys
import time
import traceback
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import ccxt
import psutil

# Configuration
UPDATE_INTERVAL = 30  # Update every 30 seconds
STATE_FILE = "../data/live_bot_state.json"
LOG_FILE = "logs/state_updater.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedStateUpdater:
    def __init__(self):
        self.exchange = None
        self.update_count = 0
        self.last_update = None
        
    def _init_exchange(self):
        """Initialize the exchange connection"""
        try:
            self.exchange = ccxt.binance({
                'sandbox': False,
                'enableRateLimit': True,
                'timeout': 10000,
            })
            logger.info("Binance exchange initialized")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            self.exchange = None

    async def get_market_data(self):
        """Fetch current market data and indicators"""
        try:
            if not self.exchange:
                self._init_exchange()
                
            # Add timeout and retry logic
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    # Get current ticker data with timeout
                    ticker = self.exchange.fetch_ticker('BTC/USDT')
                    
                    # Get some OHLCV data for 24h metrics
                    ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=24)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    high_24h = df['high'].max()
                    low_24h = df['low'].min()
                    open_24h = df['open'].iloc[0]
                    current_price = ticker['last']
                    change_24h = ((current_price - open_24h) / open_24h) * 100
                    
                    return {
                        'symbol': 'BTCUSDT',
                        'current_price': float(current_price),
                        'volume_24h': float(ticker['quoteVolume']),
                        'high_24h': float(high_24h),
                        'low_24h': float(low_24h),
                        'change_24h': float(change_24h),
                        'bid': float(ticker['bid']),
                        'ask': float(ticker['ask']),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Market data fetch attempt {retry_count} failed: {e}")
                    if retry_count < max_retries:
                        await asyncio.sleep(2)  # Wait before retry
                    else:
                        raise e
                        
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {
                'symbol': 'BTCUSDT',
                'current_price': None,
                'volume_24h': None,
                'high_24h': None,
                'low_24h': None,
                'change_24h': None,
                'timestamp': None,
                'error': str(e)
            }

    def get_basic_position_state(self):
        """Get basic position state without nesting"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                    # Only extract basic position info to prevent nesting
                    return {
                        'in_position': data.get('in_position', False),
                        'position_details': data.get('position_details', None)
                    }
            else:
                return {
                    'in_position': False,
                    'position_details': None
                }
        except Exception as e:
            logger.error(f"Error reading position state: {e}")
            return {
                'in_position': False,
                'position_details': None
            }

    def get_system_metrics(self):
        """Get current system resource metrics"""
        try:
            return {
                'cpu_percent': round(psutil.cpu_percent(interval=1), 1),
                'memory_percent': round(psutil.virtual_memory().percent, 1),
                'disk_percent': round(psutil.disk_usage('/').percent, 1),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                'cpu_percent': None,
                'memory_percent': None,
                'disk_percent': None,
                'timestamp': None,
                'error': str(e)
            }

    def get_file_status(self):
        """Get status of important trading files"""
        try:
            files_to_check = [
                '../data/trading_journal.json',
                '../live_trading/live_bot_state.json',
                '../logs/live_bot.log',
                '../logs/strategy.log'
            ]
            
            file_status = {}
            for file_path in files_to_check:
                filename = os.path.basename(file_path)
                try:
                    if os.path.exists(file_path):
                        stat = os.stat(file_path)
                        size_mb = round(stat.st_size / (1024 * 1024), 2)
                        last_modified = datetime.fromtimestamp(stat.st_mtime)
                        age_hours = round((datetime.now() - last_modified).total_seconds() / 3600, 1)
                        
                        file_status[filename] = {
                            'exists': True,
                            'size_mb': size_mb,
                            'last_modified': last_modified.isoformat(),
                            'age_hours': age_hours
                        }
                    else:
                        file_status[filename] = {
                            'exists': False,
                            'size_mb': 0,
                            'last_modified': None,
                            'age_hours': None
                        }
                except Exception as e:
                    file_status[filename] = {
                        'exists': False,
                        'error': str(e)
                    }
            
            return file_status
        except Exception as e:
            logger.error(f"Error getting file status: {e}")
            return {'error': str(e)}

    def get_trading_summary(self):
        """Get trading performance summary"""
        try:
            if os.path.exists('../data/trading_journal.json'):
                with open('../data/trading_journal.json', 'r') as f:
                    journal = json.load(f)
                    
                total_sessions = len(journal.get('sessions', []))
                total_trades = 0
                total_pnl = 0
                last_session = None
                
                sessions = journal.get('sessions', [])
                if sessions:
                    # Handle both list and dict session formats
                    if isinstance(sessions, list):
                        for session in sessions:
                            if isinstance(session, dict):
                                trades = session.get('trades', [])
                                total_trades += len(trades)
                                for trade in trades:
                                    if isinstance(trade, dict):
                                        total_pnl += trade.get('pnl', 0)
                        last_session = sessions[-1] if sessions else None
                    elif isinstance(sessions, dict):
                        # Handle dict format
                        for session_key, session in sessions.items():
                            if isinstance(session, dict):
                                trades = session.get('trades', [])
                                total_trades += len(trades)
                                for trade in trades:
                                    if isinstance(trade, dict):
                                        total_pnl += trade.get('pnl', 0)
                        # Get the last session by timestamp or key
                        session_items = list(sessions.items())
                        last_session = session_items[-1][1] if session_items else None
                
                return {
                    'total_sessions': total_sessions,
                    'total_trades': total_trades,
                    'total_pnl': round(total_pnl, 4),
                    'last_session': last_session,
                    'last_update': datetime.now(timezone.utc).isoformat()
                }
            else:
                return {
                    'total_sessions': 0,
                    'total_trades': 0,
                    'total_pnl': 0,
                    'last_session': None,
                    'last_update': None
                }
        except Exception as e:
            logger.error(f"Error getting trading summary: {e}")
            return {
                'total_sessions': 0,
                'total_trades': 0,
                'total_pnl': 0,
                'last_session': None,
                'last_update': None,
                'error': str(e)
            }

    def calculate_position_metrics(self, position_state, market_data):
        """Calculate position metrics if in a position"""
        try:
            if not position_state.get('in_position') or not position_state.get('position_details'):
                return None
                
            pos_details = position_state['position_details']
            current_price = market_data.get('current_price')
            
            if not current_price:
                return None
                
            entry_price = pos_details.get('entry_price', 0)
            quantity = pos_details.get('quantity', 0)
            side = pos_details.get('side', 'long')
            
            if side.lower() == 'long':
                unrealized_pnl = (current_price - entry_price) * quantity
            else:
                unrealized_pnl = (entry_price - current_price) * quantity
                
            return {
                'entry_price': entry_price,
                'current_price': current_price,
                'quantity': quantity,
                'side': side,
                'unrealized_pnl': round(unrealized_pnl, 2),
                'unrealized_pnl_percent': round((unrealized_pnl / (entry_price * quantity)) * 100, 2) if entry_price and quantity else 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error calculating position metrics: {e}")
            return None

    async def update_state_files(self):
        """Update both state files with current data"""
        try:
            # Get all data components
            position_state = self.get_basic_position_state()
            market_data = await self.get_market_data()
            trading_summary = self.get_trading_summary()
            system_metrics = self.get_system_metrics()
            file_status = self.get_file_status()
            position_metrics = self.calculate_position_metrics(position_state, market_data)
            
            # Create consolidated state structure
            consolidated_state = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'updater_version': '1.0.0',
                'position_state': position_state,
                'market_data': market_data,
                'position_metrics': position_metrics,
                'trading_summary': trading_summary,
                'system_status': {
                    'system_metrics': system_metrics,
                    'file_status': file_status,
                    'last_update': datetime.now(timezone.utc).isoformat()
                }
            }
            
            # Write consolidated state to main file
            with open(STATE_FILE, 'w') as f:
                json.dump(consolidated_state, f, indent=2)
            
            self.update_count += 1
            self.last_update = datetime.now(timezone.utc).isoformat()
            logger.info(f"Consolidated state updated successfully (update #{self.update_count})")
            
        except Exception as e:
            logger.error(f"Error updating consolidated state: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    async def run(self):
        """Main update loop"""
        logger.info(f"Enhanced State Updater starting... (interval: {UPDATE_INTERVAL}s)")
        
        while True:
            try:
                logger.info("Creating enhanced state...")
                await self.update_state_files()
                await asyncio.sleep(UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(UPDATE_INTERVAL)

async def main():
    updater = EnhancedStateUpdater()
    try:
        await updater.run()
    except KeyboardInterrupt:
        logger.info("State updater stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())

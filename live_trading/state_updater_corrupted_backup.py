#!/usr/bin/env python3
"""
Enhanced Live Bot State Updater
Continuously updates the live bot state file with comprehensive     def get_position_state(self):
        """Get current position state from existing state file"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                    # Extract only the position_state part, not the entire nested structure
                    if isinstance(data, dict):
                        # If it has position_state field, extract that
                        if 'position_state' in data and isinstance(data['position_state'], dict):
                            pos_state = data['position_state']
                            # Make sure we only get the actual position info, not nested data
                            if 'in_position' in pos_state or 'position_details' in pos_state:
                                return {
                                    'in_position': pos_state.get('in_position', False),
                                    'position_details': pos_state.get('position_details', None)
                                }
                        # If data itself looks like position state
                        if 'in_position' in data or 'position_details' in data:
                            return {
                                'in_position': data.get('in_position', False),
                                'position_details': data.get('position_details', None)
                            }
                    
            # Default state if file doesn't exist or doesn't have valid position data
            return {
                'in_position': False,
                'position_details': None
            }
        except Exception as e:
            logger.error(f"Error reading position state: {e}")
            return {
                'in_position': False,
                'position_details': None
            }n the main trading bot is not running.
"""

import json
import asyncio
import logging
import os
import sys
import time
import traceback
import pandas as pd
import ccxt
import psutil
from datetime import datetime, timezone
from pathlib import Path

# Configuration
UPDATE_INTERVAL = 30  # Update every 30 seconds
STATE_FILE = "../live_trading/live_bot_state.json"
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

def get_system_metrics():
    """Get basic system metrics"""
    try:
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {'error': str(e)}

class EnhancedStateUpdater:
    """Enhanced state updater that provides comprehensive bot state information"""
    
    def __init__(self):
        self.exchange = None
        self.last_update = None
        self.update_count = 0
        self._init_exchange()
        
    def _init_exchange(self):
        """Initialize CCXT exchange for market data"""
        try:
            self.exchange = ccxt.binance({
                'rateLimit': 1200,
                'enableRateLimit': True,
                'sandbox': False,
            })
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
    
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
    
    def get_position_state(self):
        """Get current position state from existing state file"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
            else:
                return {
                    'in_position': False,
                    'position_details': None
                }
        except Exception as e:
            logger.error(f"Error reading position state: {e}")
            return {
                'in_position': False,
                'position_details': None,
                'error': str(e)
            }
    
    def get_system_status(self):
        """Get comprehensive system status"""
        try:
            metrics = get_system_metrics()
            
            # Check file timestamps
            file_status = {}
            important_files = [
                '../data/trading_journal.json',
                'data/live_bot_state.json',
                'logs/live_bot.log',
                'data/crypto_data_5m.parquet',
                'live_trading/health_history.json'
            ]
            
            for file_path in important_files:
                if os.path.exists(file_path):
                    stat = os.stat(file_path)
                    age_hours = (time.time() - stat.st_mtime) / 3600
                    file_status[os.path.basename(file_path)] = {
                        'exists': True,
                        'size_mb': round(stat.st_size / (1024 * 1024), 2),
                        'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'age_hours': round(age_hours, 1)
                    }
                else:
                    file_status[os.path.basename(file_path)] = {
                        'exists': False,
                        'size_mb': 0,
                        'last_modified': None,
                        'age_hours': None
                    }
            
            return {
                'system_metrics': metrics,
                'file_status': file_status,
                'updater_status': {
                    'last_update': self.last_update,
                    'update_count': self.update_count,
                    'uptime_minutes': round((time.time() - getattr(self, 'start_time', time.time())) / 60, 1)
                }
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'system_metrics': None,
                'file_status': {},
                'updater_status': {
                    'error': str(e)
                }
            }
    
    def get_trading_summary(self):
        """Get trading performance summary"""
        try:
            if os.path.exists('../data/trading_journal.json'):
                with open('../data/trading_journal.json', 'r') as f:
                    journal = json.load(f)
                
                metadata = journal.get('metadata', {})
                sessions = journal.get('sessions', {})
                
                # Handle both list and dict formats for sessions
                if isinstance(sessions, list):
                    # List format
                    total_trades = sum(len(session.get('trades', [])) for session in sessions)
                    total_pnl = sum(session.get('total_pnl', 0) for session in sessions)
                    session_count = len(sessions)
                elif isinstance(sessions, dict):
                    # Dict format
                    total_trades = sum(len(session.get('trades', [])) for session in sessions.values())
                    total_pnl = sum(session.get('total_pnl', 0) for session in sessions.values())
                    session_count = len(sessions)
                else:
                    # Unknown format
                    total_trades = 0
                    total_pnl = 0
                    session_count = 0
                
                return {
                    'total_sessions': session_count,
                    'total_trades': total_trades,
                    'total_pnl': round(total_pnl, 4),
                    'last_session': metadata.get('last_session'),
                    'last_update': metadata.get('last_update')
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
            return {'error': str(e)}
    
    async def create_enhanced_state(self):
        """Create comprehensive enhanced state"""
        logger.info("Creating enhanced state...")
        
        # Get all components
        position_state = self.get_position_state()
        market_data = await self.get_market_data()
        system_status = self.get_system_status()
        trading_summary = self.get_trading_summary()
        
        # Calculate position metrics if in position
        position_metrics = None
        if position_state.get('in_position') and position_state.get('position_details'):
            details = position_state['position_details']
            current_price = market_data.get('current_price')
            
            if current_price and details.get('entry_price'):
                entry_price = details['entry_price']
                size = details.get('size', 0)
                
                unrealized_pnl = (current_price - entry_price) * size
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                
                position_metrics = {
                    'unrealized_pnl': round(unrealized_pnl, 6),
                    'unrealized_pnl_pct': round(unrealized_pnl_pct, 2),
                    'current_value': round(current_price * size, 2),
                    'entry_value': round(entry_price * size, 2),
                    'stop_loss_distance': round(abs(current_price - details.get('stop_loss', current_price)) / current_price * 100, 2) if details.get('stop_loss') else None,
                    'take_profit_distance': round(abs(details.get('partial_tp_price', current_price) - current_price) / current_price * 100, 2) if details.get('partial_tp_price') else None
                }
        
        enhanced_state = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'updater_version': '1.0.0',
            'position_state': position_state,
            'market_data': market_data,
            'position_metrics': position_metrics,
            'trading_summary': trading_summary,
            'system_status': system_status
        }
        
        return enhanced_state
    
    async def update_state_files(self):
        """Update the consolidated live bot state file with enhanced data"""
        try:
            enhanced_state = await self.create_enhanced_state()
            
            # Save consolidated enhanced state to the main live_bot_state.json file
            with open(STATE_FILE, 'w') as f:
                json.dump(enhanced_state, f, indent=2, default=str)
            
            self.last_update = datetime.now(timezone.utc).isoformat()
            self.update_count += 1
            
            logger.info(f"Consolidated state updated successfully (update #{self.update_count})")
            
        except Exception as e:
            logger.error(f"Error updating consolidated state: {e}")
            logger.error(traceback.format_exc())
    
    async def run(self):
        """Main update loop"""
        self.start_time = time.time()
        logger.info(f"Enhanced State Updater starting... (interval: {UPDATE_INTERVAL}s)")
        
        while True:
            try:
                await self.update_state_files()
                await asyncio.sleep(UPDATE_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("State updater stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(UPDATE_INTERVAL)

async def main():
    """Main entry point"""
    updater = EnhancedStateUpdater()
    await updater.run()

if __name__ == "__main__":
    asyncio.run(main())

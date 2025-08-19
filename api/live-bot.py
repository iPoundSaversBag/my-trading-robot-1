"""
Vercel Live Trading Bot - Uses backtest configuration for live trading
Integrates with your existing optimization parameters
"""

import json
import os
import requests
import hashlib
import hmac
import time
from urllib.parse import urlencode
from http.server import BaseHTTPRequestHandler

def load_trading_config():
    """Load optimized trading parameters from backtest results"""
    try:
        # Try to load optimized parameters from backtest
        config_path = os.path.join(os.path.dirname(__file__), 'live_trading_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded optimized parameters: RSI={config.get('RSI_PERIOD')}, MA_FAST={config.get('MA_FAST')}, MA_SLOW={config.get('MA_SLOW')}")
            return config
        else:
            print("⚠️ No optimized config found, using defaults")
            return None
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

# Load optimized parameters or use defaults
OPTIMIZED_CONFIG = load_trading_config()

# Default trading parameters (fallback if optimized config not available)
DEFAULT_CONFIG = {
    "INITIAL_CAPITAL": 10000,
    "POSITION_SIZE": 0.02,
    "COMMISSION_RATE": 0.001,
    "SLIPPAGE_RATE": 0.0001,
    "MAX_PORTFOLIO_RISK": 0.15,
    "MAX_CONSECUTIVE_LOSSES": 5,
    "RISK_REDUCTION_FACTOR": 0.8,
    "MIN_PROFIT_FACTOR_FOR_RISK_INCREASE": 1.2,
    "min_confidence_for_trade": 0.04,
    "USE_ML_REGIME_DETECTION": True,
    "BLOCK_LOW_CONFIDENCE_SIGNALS": True,
    "VOLUME_CONFIRMATION": True,
    "SYMBOL": "BTCUSDT",
    "TIMEFRAME": "5m",
    "RSI_OVERSOLD": 30,
    "RSI_OVERBOUGHT": 70,
    "RSI_PERIOD": 14,
    "MA_FAST": 12,
    "MA_SLOW": 26,
    "MA_SIGNAL": 9
}

# Use optimized config if available, otherwise use defaults
TRADING_CONFIG = OPTIMIZED_CONFIG if OPTIMIZED_CONFIG else DEFAULT_CONFIG

class VercelLiveBot:
    def __init__(self):
        self.api_key = os.environ.get('BINANCE_API_KEY', '')
        self.api_secret = os.environ.get('BINANCE_API_SECRET', '')
        self.base_url = 'https://api.binance.com'
        self.config = TRADING_CONFIG
        self.trade_log = []
    
    def _create_signature(self, params):
        """Create HMAC SHA256 signature for Binance API"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_request(self, endpoint, params=None, method='GET'):
        """Make authenticated request to Binance API"""
        if not params:
            params = {}
        
        params['timestamp'] = int(time.time() * 1000)
        signature = self._create_signature(params)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': self.api_key}
        
        try:
            if method == 'POST':
                response = requests.post(f"{self.base_url}{endpoint}", params=params, headers=headers, timeout=10)
            else:
                response = requests.get(f"{self.base_url}{endpoint}", params=params, headers=headers, timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_market_data(self, symbol='BTCUSDT', interval='5m', limit=100):
        """Get market data using your backtest timeframe"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v3/klines",
                params={'symbol': symbol, 'interval': interval, 'limit': limit},
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_indicators(self, prices, volumes=None):
        """Calculate trading indicators using your backtest parameters"""
        if len(prices) < max(self.config['RSI_PERIOD'], self.config['MA_SLOW'], self.config.get('ADX_PERIOD', 14)):
            return None
        
        # RSI calculation
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, delta) for delta in deltas]
        losses = [abs(min(0, delta)) for delta in deltas]
        
        period = self.config['RSI_PERIOD']
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        rsi = 50 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
        
        # Moving averages
        ma_fast = sum(prices[-self.config['MA_FAST']:]) / self.config['MA_FAST']
        ma_slow = sum(prices[-self.config['MA_SLOW']:]) / self.config['MA_SLOW']
        
        # ADX calculation for trend strength
        adx_period = self.config.get('ADX_PERIOD', 14)
        adx = self._calculate_adx(prices, adx_period)
        
        # ATR for volatility measurement
        atr_period = self.config.get('ATR_PERIOD', 14)
        atr = self._calculate_atr(prices, atr_period)
        
        # Volume analysis if available
        volume_ratio = 1.0
        if volumes and len(volumes) >= 20:
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        return {
            'rsi': rsi,
            'ma_fast': ma_fast,
            'ma_slow': ma_slow,
            'current_price': prices[-1],
            'adx': adx,
            'atr': atr,
            'volume_ratio': volume_ratio,
            'price_change_pct': ((prices[-1] - prices[-2]) / prices[-2] * 100) if len(prices) > 1 else 0
        }
    
    def _calculate_adx(self, prices, period=14):
        """Calculate ADX for trend strength measurement"""
        if len(prices) < period + 1:
            return 25  # Default neutral value
        
        # Simplified ADX calculation
        price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        avg_change = sum(price_changes[-period:]) / period
        recent_change = sum(price_changes[-5:]) / 5
        
        # Simple ADX approximation: trend strength based on price momentum
        adx = min(100, max(0, (recent_change / avg_change) * 25)) if avg_change > 0 else 25
        return adx
    
    def _calculate_atr(self, prices, period=14):
        """Calculate ATR for volatility measurement"""
        if len(prices) < period + 1:
            return 0.01  # Default low volatility
        
        # Simplified ATR calculation
        price_ranges = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        atr = sum(price_ranges[-period:]) / period
        return atr
    
    def detect_market_regime(self, indicators):
        """Detect current market regime based on indicators"""
        adx = indicators['adx']
        atr = indicators['atr']
        volume_ratio = indicators['volume_ratio']
        price_change_pct = indicators['price_change_pct']
        ma_fast = indicators['ma_fast']
        ma_slow = indicators['ma_slow']
        current_price = indicators['current_price']
        
        # Volatility regime
        volatility_threshold = self.config.get('volatility_threshold', 0.03)
        price_volatility = abs(price_change_pct) / 100
        
        if price_volatility > volatility_threshold:
            volatility_regime = 'high_volatility'
        else:
            volatility_regime = 'low_volatility'
        
        # Trend regime based on ADX and MA
        adx_threshold = 25
        if adx > adx_threshold:
            if ma_fast > ma_slow and current_price > ma_slow:
                trend_regime = 'trending_bull'
            elif ma_fast < ma_slow and current_price < ma_slow:
                trend_regime = 'trending_bear'
            else:
                trend_regime = 'trending_bull' if price_change_pct > 0 else 'trending_bear'
        else:
            trend_regime = 'ranging'
        
        # Breakout detection
        volume_threshold = self.config.get('volume_threshold_multiplier', 2.0)
        if volume_ratio > volume_threshold and abs(price_change_pct) > 1.0:
            if price_change_pct > 0:
                breakout_regime = 'breakout_bullish'
            else:
                breakout_regime = 'breakout_bearish'
        else:
            breakout_regime = None
        
        # Primary regime priority: breakout > trending > ranging
        if breakout_regime:
            primary_regime = breakout_regime
        else:
            primary_regime = trend_regime
        
        return {
            'primary': primary_regime,
            'trend': trend_regime,
            'volatility': volatility_regime,
            'breakout': breakout_regime,
            'adx': adx,
            'volume_ratio': volume_ratio,
            'volatility_level': price_volatility
        }
    
    def generate_signal(self, indicators, market_regime):
        """Generate trading signals using regime-aware backtest logic"""
        if not indicators:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
        
        rsi = indicators['rsi']
        ma_fast = indicators['ma_fast']
        ma_slow = indicators['ma_slow']
        current_price = indicators['current_price']
        adx = indicators['adx']
        
        confidence = 0
        reasons = []
        signal = 'HOLD'
        
        # Get regime-specific filter settings
        regime = market_regime['primary']
        
        # Apply regime-specific filters based on optimized parameters
        filters_active = {
            'ichimoku_cloud': self.config.get(f'USE_ICHIMOKU_CLOUD_FILTER_{regime}', True),
            'rsi': self.config.get(f'USE_RSI_FILTER_{regime}', True),
            'adx': self.config.get(f'USE_ADX_FILTER_{regime}', True),
            'bbands': self.config.get(f'USE_BBANDS_FILTER_{regime}', False),
            'macd': self.config.get(f'USE_MACD_FILTER_{regime}', False),
            'volume_breakout': self.config.get(f'USE_VOLUME_BREAKOUT_FILTER_{regime}', True)
        }
        
        # RSI signals (if RSI filter is active for this regime)
        if filters_active['rsi']:
            if rsi < self.config['RSI_OVERSOLD']:
                confidence += 0.3
                reasons.append(f"RSI oversold ({rsi:.1f})")
                signal = 'BUY'
            elif rsi > self.config['RSI_OVERBOUGHT']:
                confidence += 0.3
                reasons.append(f"RSI overbought ({rsi:.1f})")
                signal = 'SELL'
        
        # Moving average signals (Ichimoku substitute)
        if filters_active['ichimoku_cloud']:
            if ma_fast > ma_slow and current_price > ma_fast:
                confidence += 0.25
                reasons.append(f"Bullish MA cross")
                if signal != 'SELL':
                    signal = 'BUY'
            elif ma_fast < ma_slow and current_price < ma_fast:
                confidence += 0.25
                reasons.append(f"Bearish MA cross")
                if signal != 'BUY':
                    signal = 'SELL'
        
        # ADX trend strength filter
        if filters_active['adx']:
            if adx > 25:
                confidence += 0.2
                reasons.append(f"Strong trend (ADX: {adx:.1f})")
            else:
                confidence -= 0.1
                reasons.append(f"Weak trend (ADX: {adx:.1f})")
        
        # Volume breakout confirmation
        if filters_active['volume_breakout'] and market_regime['breakout']:
            confidence += 0.25
            reasons.append(f"Volume breakout ({market_regime['volume_ratio']:.1f}x)")
        
        # Regime-specific confidence adjustments
        regime_adjustments = {
            'trending_bull': 0.1,
            'trending_bear': 0.1,
            'ranging': -0.1,
            'breakout_bullish': 0.2,
            'breakout_bearish': 0.2,
            'high_volatility': -0.05,
            'low_volatility': 0.05
        }
        
        confidence += regime_adjustments.get(regime, 0)
        
        # Apply minimum confidence threshold from backtest optimization
        min_confidence = self.config.get('min_confidence_for_trade', 0.04)
        
        if confidence < min_confidence:
            signal = 'HOLD'
            reasons.append(f"Below confidence threshold ({confidence:.3f} < {min_confidence:.3f})")
        
        return {
            'signal': signal,
            'confidence': round(confidence, 4),
            'reason': ' | '.join(reasons),
            'regime': regime,
            'regime_details': market_regime,
            'filters_used': [k for k, v in filters_active.items() if v]
        }
        
        # RSI signals (from your backtest parameters)
        if rsi < self.config['RSI_OVERSOLD']:
            signal = 'BUY'
            confidence += 0.4
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > self.config['RSI_OVERBOUGHT']:
            signal = 'SELL' 
            confidence += 0.4
            reasons.append(f"RSI overbought ({rsi:.1f})")
        else:
            signal = 'HOLD'
        
        # Moving average signals
        if ma_fast > ma_slow and current_price > ma_fast:
            if signal != 'SELL':
                signal = 'BUY'
            confidence += 0.3
            reasons.append("MA bullish trend")
        elif ma_fast < ma_slow and current_price < ma_fast:
            if signal != 'BUY':
                signal = 'SELL'
            confidence += 0.3
            reasons.append("MA bearish trend")
        
        # Check minimum confidence threshold from backtest
        if confidence < self.config['min_confidence_for_trade']:
            signal = 'HOLD'
            reasons.append(f"Low confidence ({confidence:.2f})")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': '; '.join(reasons),
            'indicators': indicators
        }
    
    def calculate_position_size(self, account_balance, current_price):
        """Calculate position size using your backtest risk management"""
        usdt_balance = account_balance.get('USDT', 0)
        
        # Use your backtest position sizing
        max_position_value = usdt_balance * self.config['POSITION_SIZE']
        quantity = max_position_value / current_price
        
        # Apply minimum and maximum constraints
        min_qty = 0.00001  # Binance minimum for BTC
        max_qty = usdt_balance * self.config['MAX_PORTFOLIO_RISK'] / current_price
        
        return max(min_qty, min(quantity, max_qty))
    
    def execute_live_trading_cycle(self):
        """Execute one live trading cycle using regime-aware backtest strategy"""
        try:
            # Get market data with volume
            klines = self.get_market_data(self.config['SYMBOL'], self.config['TIMEFRAME'])
            if "error" in klines:
                return {"error": "Failed to get market data", "details": klines["error"]}
            
            # Extract closing prices and volumes
            closes = [float(kline[4]) for kline in klines]
            volumes = [float(kline[5]) for kline in klines]
            
            # Calculate indicators including regime detection data
            indicators = self.calculate_indicators(closes, volumes)
            if not indicators:
                return {"error": "Insufficient market data for analysis"}
            
            # Detect current market regime
            market_regime = self.detect_market_regime(indicators)
            
            # Generate regime-aware trading signal
            signal_data = self.generate_signal(indicators, market_regime)
            
            # Get account information
            account = self._make_request('/api/v3/account')
            if "error" in account:
                return {"error": "Failed to get account info", "details": account["error"]}
            
            # Extract balances
            balances = {}
            for balance in account.get('balances', []):
                free = float(balance['free'])
                if free > 0:
                    balances[balance['asset']] = free
            
            # Execute trade if signal is strong enough
            trade_result = None
            if signal_data['signal'] in ['BUY', 'SELL'] and signal_data['confidence'] >= self.config['min_confidence_for_trade']:
                
                quantity = self.calculate_position_size(balances, indicators['current_price'])
                
                # Place order (commented for safety - uncomment when ready for live trading)
                # trade_params = {
                #     'symbol': self.config['SYMBOL'],
                #     'side': signal_data['signal'],
                #     'type': 'MARKET',
                #     'quantity': f"{quantity:.5f}"
                # }
                # trade_result = self._make_request('/api/v3/order', trade_params, 'POST')
                
                # For now, simulate the trade
                trade_result = {
                    'simulated': True,
                    'symbol': self.config['SYMBOL'],
                    'side': signal_data['signal'],
                    'quantity': f"{quantity:.5f}",
                    'price': indicators['current_price'],
                    'value': quantity * indicators['current_price']
                }
            
            return {
                'status': 'success',
                'timestamp': int(time.time()),
                'signal': signal_data,
                'market_regime': market_regime,
                'account_balance': balances,
                'trade_executed': trade_result,
                'config_source': 'optimized' if OPTIMIZED_CONFIG else 'default',
                'parameters_used': {
                    'RSI_PERIOD': self.config['RSI_PERIOD'],
                    'MA_FAST': self.config['MA_FAST'], 
                    'MA_SLOW': self.config['MA_SLOW'],
                    'RSI_OVERBOUGHT': self.config['RSI_OVERBOUGHT'],
                    'RSI_OVERSOLD': self.config['RSI_OVERSOLD'],
                    'source_window': self.config.get('source_window', 'N/A'),
                    'optimization_timestamp': self.config.get('optimization_timestamp', 'N/A')
                },
                'regime_integration': 'active',
                'backtest_integration': 'active',
                'bidirectional_sync': 'enabled'
            }
            
        except Exception as e:
            return {"error": f"Trading cycle failed: {str(e)}"}

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Security check for automated requests
            auth_header = self.headers.get('Authorization')
            cron_secret = os.environ.get('BOT_SECRET')
            if cron_secret and auth_header != f'Bearer {cron_secret}':
                self.send_response(401)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                error_response = {"error": "Unauthorized"}
                self.wfile.write(json.dumps(error_response).encode())
                return
            
            bot = VercelLiveBot()
            result = bot.execute_live_trading_cycle()
            
            status_code = 200 if result.get('status') == 'success' else 500
            
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()
            
            self.wfile.write(json.dumps(result, indent=2).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "status": "error",
                "message": f"Live bot failed: {str(e)}"
            }
            self.wfile.write(json.dumps(error_response).encode())

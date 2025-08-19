#!/usr/bin/env python3
"""
Live Bot Integration Test - Test all functionality safely
"""

import json
import os
import sys
from datetime import datetime

def test_configuration():
    """Test configuration loading"""
    print("Testing Configuration...")
    
    try:
        with open("api/live_trading_config.json", 'r') as f:
            config = json.load(f)
        
        required_params = [
            'RSI_PERIOD', 'RSI_OVERBOUGHT', 'RSI_OVERSOLD',
            'STOP_LOSS_MULTIPLIER', 'TAKE_PROFIT_MULTIPLIER',
            'POSITION_SIZE', 'min_confidence_for_trade'
        ]
        
        missing = [p for p in required_params if p not in config]
        
        if missing:
            print(f"Missing parameters: {missing}")
            return False
        else:
            print("All required parameters present")
            return True
            
    except Exception as e:
        print(f"Config test failed: {e}")
        return False

def test_signal_generation():
    """Test signal generation logic"""
    print("\nTesting Signal Generation...")
    
    # Mock market data
    test_data = {
        'price': 50000.0,
        'rsi': 25.0,  # Oversold
        'ma_fast': 49800.0,
        'ma_slow': 50200.0,
        'atr': 1500.0,
        'volume': 1000000.0
    }
    
    try:
        with open("api/live_trading_config.json", 'r') as f:
            config = json.load(f)
        
        # Test RSI signal
        rsi_oversold = test_data['rsi'] < config['RSI_OVERSOLD']
        rsi_overbought = test_data['rsi'] > config['RSI_OVERBOUGHT']
        
        # Test MA signal
        ma_bullish = test_data['ma_fast'] > test_data['ma_slow']
        
        print(f"   RSI: {test_data['rsi']:.1f}")
        print(f"   RSI Oversold ({config['RSI_OVERSOLD']}): {rsi_oversold}")
        print(f"   MA Bullish: {ma_bullish}")
        
        # Generate signal
        if rsi_oversold and ma_bullish:
            signal = 'BUY'
            confidence = 0.75
        elif rsi_overbought and not ma_bullish:
            signal = 'SELL'
            confidence = 0.75
        else:
            signal = 'HOLD'
            confidence = 0.25
        
        print(f"   Generated Signal: {signal} (confidence: {confidence:.2f})")
        
        return signal != 'HOLD' and confidence >= config['min_confidence_for_trade']
        
    except Exception as e:
        print(f"Signal test failed: {e}")
        return False

def test_risk_management():
    """Test risk management calculations"""
    print("\nTesting Risk Management...")
    
    try:
        with open("api/live_trading_config.json", 'r') as f:
            config = json.load(f)
        
        # Test parameters
        price = 50000.0
        atr = 1500.0
        capital = config.get('INITIAL_CAPITAL', 10000)
        position_size = config.get('POSITION_SIZE', 0.02)
        
        # Calculate trade size
        trade_amount = capital * position_size
        quantity = trade_amount / price
        
        # Calculate stops
        stop_loss_mult = config.get('STOP_LOSS_MULTIPLIER', 2.0)
        take_profit_mult = config.get('TAKE_PROFIT_MULTIPLIER', 3.0)
        
        stop_loss = price - (atr * stop_loss_mult)
        take_profit = price + (atr * take_profit_mult)
        
        # Calculate risk
        risk_per_trade = abs(price - stop_loss) * quantity
        risk_percentage = (risk_per_trade / capital) * 100
        
        print(f"   Capital: ${capital:,.2f}")
        print(f"   Position Size: {position_size*100:.1f}%")
        print(f"   Trade Amount: ${trade_amount:,.2f}")
        print(f"   Quantity: {quantity:.6f}")
        print(f"   Stop Loss: ${stop_loss:,.2f}")
        print(f"   Take Profit: ${take_profit:,.2f}")
        print(f"   Risk per Trade: ${risk_per_trade:,.2f} ({risk_percentage:.2f}%)")
        
        # Validate risk is reasonable
        max_risk = config.get('MAX_PORTFOLIO_RISK', 0.15) * 100
        risk_ok = risk_percentage <= max_risk
        
        print(f"   Risk Check: {risk_ok} (max: {max_risk:.1f}%)")
        
        return risk_ok
        
    except Exception as e:
        print(f"Risk management test failed: {e}")
        return False

if __name__ == "__main__":
    print("LIVE BOT INTEGRATION TEST")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Signal Generation", test_signal_generation), 
        ("Risk Management", test_risk_management)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"PASSED: {test_name}")
            else:
                print(f"FAILED: {test_name}")
        except Exception as e:
            print(f"ERROR: {test_name} - {e}")
    
    print(f"\nTEST RESULTS: {passed}/{total} passed")
    
    if passed == total:
        print("ALL TESTS PASSED!")
        print("Live bot configuration is ready")
    else:
        print("Some tests failed - please review configuration")

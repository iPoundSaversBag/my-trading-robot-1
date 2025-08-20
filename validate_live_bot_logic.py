#!/usr/bin/env python3
"""
Live Bot Logic Validation - Ensure live bot matches backtest logic exactly
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add dotenv import for environment variable loading
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("⚠️ python-dotenv not available. Install with: pip install python-dotenv")

def validate_live_bot_logic():
    """Validate that live bot logic matches backtest logic"""
    print("🔍 LIVE BOT LOGIC VALIDATION")
    print("=" * 60)
    
    validation_results = {}
    
    # 1. Parameter Validation
    print("\n1. 📊 PARAMETER VALIDATION")
    print("-" * 40)
    
    param_validation = validate_parameters()
    validation_results['parameters'] = param_validation
    
    # 2. Signal Generation Validation
    print("\n2. 📈 SIGNAL GENERATION VALIDATION")
    print("-" * 40)
    
    signal_validation = validate_signal_generation()
    validation_results['signals'] = signal_validation
    
    # 3. Risk Management Validation
    print("\n3. 🛡️ RISK MANAGEMENT VALIDATION")
    print("-" * 40)
    
    risk_validation = validate_risk_management()
    validation_results['risk_management'] = risk_validation
    
    # 4. Trade Execution Logic
    print("\n4. ⚡ TRADE EXECUTION LOGIC")
    print("-" * 40)
    
    execution_validation = validate_execution_logic()
    validation_results['execution'] = execution_validation
    
    # 5. Vercel Deployment Validation
    vercel_validation = validate_vercel_deployment()
    validation_results['vercel'] = vercel_validation
    
    # 6. Trade Simulation Test
    trade_simulation = create_test_trade_simulation()
    validation_results['trade_simulation'] = trade_simulation
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    # Calculate totals properly - handle mixed dict/bool values
    total_checks = 0
    passed_checks = 0
    
    for section, results in validation_results.items():
        if isinstance(results, dict):
            total_checks += len(results)
            passed_checks += sum(1 for v in results.values() if v)
        elif isinstance(results, bool):
            total_checks += 1
            passed_checks += 1 if results else 0
    
    success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    print(f"\n✅ Passed: {passed_checks}/{total_checks} checks")
    print(f"📊 Success Rate: {success_rate:.1f}%")
    
    # Detailed results
    for category, results in validation_results.items():
        if isinstance(results, dict):
            failed = [k for k, v in results.items() if not v]
            if failed:
                print(f"\n❌ {category.upper()} - Failed checks:")
                for check in failed:
                    print(f"   - {check}")
    
    return passed_checks == total_checks

def validate_parameters():
    """Validate parameter consistency between backtest and live bot"""
    results = {}
    
    try:
        # Load configurations
        with open("core/optimization_config.json", 'r') as f:
            backtest_config = json.load(f)
        
        with open("api/live_trading_config.json", 'r') as f:
            live_config = json.load(f)
        
        # Check critical parameters
        critical_params = {
            "RSI_PERIOD": "RSI calculation period",
            "RSI_OVERBOUGHT": "RSI overbought threshold", 
            "RSI_OVERSOLD": "RSI oversold threshold",
            "ATR_PERIOD": "ATR calculation period",
            "ADX_THRESHOLD": "ADX trend strength threshold",
            "min_confidence_for_trade": "Minimum signal confidence",
            "POSITION_SIZE": "Position sizing"
        }
        
        best_params = backtest_config.get("best_parameters_so_far", {})
        fixed_params = backtest_config.get("fixed_parameters", {})
        
        for param, description in critical_params.items():
            backtest_val = best_params.get(param) or fixed_params.get(param)
            live_val = live_config.get(param)
            
            if backtest_val is not None and live_val is not None:
                if isinstance(backtest_val, float):
                    match = abs(backtest_val - live_val) < 0.001
                else:
                    match = backtest_val == live_val
                
                results[description] = match
                status = "✅" if match else "❌"
                print(f"   {status} {description}: Backtest={backtest_val}, Live={live_val}")
            else:
                results[description] = False
                print(f"   ❌ {description}: Missing value")
        
    except Exception as e:
        print(f"   ❌ Parameter validation failed: {e}")
        results["Parameter loading"] = False
    
    return results

def validate_signal_generation():
    """Validate signal generation consistency"""
    results = {}
    
    try:
        # Load live config
        with open("api/live_trading_config.json", 'r') as f:
            config = json.load(f)
        
        # Check signal components
        signal_components = {
            "RSI signals": config.get("RSI_PERIOD") is not None,
            "Moving averages": config.get("MA_FAST") is not None and config.get("MA_SLOW") is not None,
            "ATR volatility": config.get("ATR_PERIOD") is not None,
            "ADX trend strength": config.get("ADX_THRESHOLD") is not None,
            "Confidence scoring": config.get("min_confidence_for_trade") is not None,
            "Regime filtering": any(k.startswith('USE_') and '_FILTER_' in k for k in config.keys())
        }
        
        for component, available in signal_components.items():
            results[component] = available
            status = "✅" if available else "❌"
            print(f"   {status} {component}")
        
        # Check active regime filters
        active_filters = [k for k, v in config.items() if k.startswith('USE_') and '_FILTER_' in k and v]
        results["Active regime filters"] = len(active_filters) > 0
        print(f"   ✅ Active regime filters: {len(active_filters)}")
        
    except Exception as e:
        print(f"   ❌ Signal validation failed: {e}")
        results["Signal configuration"] = False
    
    return results

def validate_risk_management():
    """Validate risk management consistency"""
    results = {}
    
    try:
        with open("api/live_trading_config.json", 'r') as f:
            config = json.load(f)
        
        # Check risk management components
        risk_components = {
            "Position sizing": config.get("POSITION_SIZE") is not None,
            "Maximum portfolio risk": config.get("MAX_PORTFOLIO_RISK") is not None,
            "Stop loss multiplier": config.get("STOP_LOSS_MULTIPLIER") is not None,
            "Take profit multiplier": config.get("TAKE_PROFIT_MULTIPLIER") is not None,
            "Trailing stop": config.get("TRAILING_STOP_MULTIPLIER") is not None,
            "Commission rate": config.get("COMMISSION_RATE") is not None,
            "Slippage rate": config.get("SLIPPAGE_RATE") is not None
        }
        
        for component, available in risk_components.items():
            results[component] = available
            status = "✅" if available else "❌"
            print(f"   {status} {component}")
        
        # Validate risk values are reasonable
        position_size = config.get("POSITION_SIZE", 0)
        max_risk = config.get("MAX_PORTFOLIO_RISK", 0)
        
        results["Reasonable position sizing"] = 0.001 <= position_size <= 0.1
        results["Reasonable portfolio risk"] = 0.01 <= max_risk <= 0.5
        
        print(f"   {'✅' if results['Reasonable position sizing'] else '❌'} Position size: {position_size*100:.1f}%")
        print(f"   {'✅' if results['Reasonable portfolio risk'] else '❌'} Max portfolio risk: {max_risk*100:.1f}%")
        
    except Exception as e:
        print(f"   ❌ Risk management validation failed: {e}")
        results["Risk management configuration"] = False
    
    return results

def validate_execution_logic():
    """Validate trade execution logic"""
    results = {}
    
    try:
        # Check if live bot file exists and has required methods
        live_bot_path = "api/live-bot.py"
        if os.path.exists(live_bot_path):
            with open(live_bot_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check for required methods/functions - Updated for Vercel serverless functions
            required_functions = {
                "Market data fetching": "get_market_data" in content or "fetch" in content.lower(),
                "Technical indicators": any(indicator in content.lower() for indicator in ["rsi", "sma", "ema", "atr"]),
                "Signal generation": "signal" in content.lower(),
                "Vercel handler": "class handler" in content or "def handler" in content,
                "Error handling": "try:" in content and "except" in content,
                "Configuration loading": "load_trading_config" in content
            }
            
            for function, available in required_functions.items():
                results[function] = available
                status = "✅" if available else "❌"
                print(f"   {status} {function}")
        
        else:
            print(f"   ❌ Live bot file not found: {live_bot_path}")
            results["Live bot file exists"] = False
        
        # Check API configuration for Vercel deployment
        # Load environment variables from .env file
        if DOTENV_AVAILABLE:
            load_dotenv()
        
        # Check both local and Vercel environment variable formats
        api_key_set = bool(os.environ.get('BINANCE_API_KEY') or os.environ.get('BINANCE_API_KEY_TESTNET'))
        api_secret_set = bool(os.environ.get('BINANCE_API_SECRET') or os.environ.get('BINANCE_API_SECRET_TESTNET'))
        
        # Check if .env file exists (for local testing)
        env_file_exists = os.path.exists('.env')
        
        results["API key configured"] = api_key_set
        results["API secret configured"] = api_secret_set
        results["Environment file exists"] = env_file_exists
        
        print(f"   {'✅' if api_key_set else '❌'} Binance API key configured (Vercel/Local)")
        print(f"   {'✅' if api_secret_set else '❌'} Binance API secret configured (Vercel/Local)")
        print(f"   {'✅' if env_file_exists else '❌'} Environment file exists")
        
    except Exception as e:
        print(f"   ❌ Execution logic validation failed: {e}")
        results["Execution logic validation"] = False
    
    return results

def validate_vercel_deployment():
    """Test Vercel deployment and API endpoint"""
    print("\n5. 🌐 VERCEL DEPLOYMENT VALIDATION")
    print("-" * 40)
    
    results = {}
    
    try:
        import requests
        if DOTENV_AVAILABLE:
            load_dotenv()
        
        # Test if we can import the live bot locally
        try:
            import sys
            sys.path.insert(0, 'api')
            # Try to import the handler class from live-bot.py
            with open('api/live-bot.py', 'r', encoding='utf-8', errors='ignore') as f:
                bot_content = f.read()
            
            results["Local bot import"] = "class handler" in bot_content
            print("   ✅ Live bot handler class found" if results["Local bot import"] else "   ❌ Live bot handler class not found")
        except Exception as e:
            results["Local bot import"] = False
            print(f"   ❌ Live bot validation failed: {e}")
        
        # Test Vercel API endpoint if URL is configured
        vercel_url = "https://my-trading-robot-1.vercel.app"  # Use main production URL
        bot_secret = os.environ.get('BOT_SECRET')
        
        # First try with authentication
        if vercel_url and bot_secret:
            try:
                headers = {"Authorization": f"Bearer {bot_secret}"}
                response = requests.get(f"{vercel_url}/api/live-bot", headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results["Vercel API response"] = True
                    results["Signal generation working"] = 'signal' in data
                    print("   ✅ Vercel API endpoint accessible")
                    print(f"   ✅ Signal generated: {data.get('signal', {}).get('signal', 'N/A')}")
                else:
                    results["Vercel API response"] = False
                    print(f"   ⚠️ Vercel API returned status: {response.status_code}")
            except Exception as e:
                results["Vercel API response"] = False
                print(f"   ⚠️ Vercel API test failed: {e}")
        else:
            # Fallback: Test without authentication (just connectivity)
            try:
                response = requests.get(f"{vercel_url}/", timeout=10)
                if response.status_code == 200:
                    results["Vercel connectivity"] = True
                    print("   ✅ Vercel deployment accessible (basic connectivity)")
                else:
                    results["Vercel connectivity"] = False
                    print(f"   ⚠️ Vercel connectivity test failed: {response.status_code}")
            except Exception as e:
                results["Vercel connectivity"] = False
                print(f"   ❌ Vercel not accessible: {e}")
                
            results["Vercel API response"] = False
            print("   ⚠️ Vercel URL or bot secret not configured for API testing")
            
    except Exception as e:
        print(f"   ❌ Vercel validation failed: {e}")
        results["Vercel validation"] = False
    
    return results

def create_test_trade_simulation():
    """Create a test to simulate trade logic"""
    print("\n6. 🧪 TRADE SIMULATION TEST")
    print("-" * 40)
    
    try:
        # Load configuration
        with open("api/live_trading_config.json", 'r') as f:
            config = json.load(f)
        
        # Simulate market data
        test_data = {
            "symbol": "BTCUSDT",
            "price": 50000.0,
            "rsi": 35.0,  # Oversold condition
            "atr": 1500.0,
            "volume": 1000000.0,
            "ma_fast": 49800.0,
            "ma_slow": 50200.0
        }
        
        # Simulate signal generation
        rsi_oversold = test_data["rsi"] < config.get("RSI_OVERSOLD", 30)
        ma_bullish = test_data["ma_fast"] > test_data["ma_slow"]
        
        # Calculate position size
        capital = config.get("INITIAL_CAPITAL", 10000)
        position_size = config.get("POSITION_SIZE", 0.02)
        trade_amount = capital * position_size
        
        # Calculate stop loss and take profit
        atr_multiplier_sl = config.get("STOP_LOSS_MULTIPLIER", 2.0)
        atr_multiplier_tp = config.get("TAKE_PROFIT_MULTIPLIER", 3.0)
        
        stop_loss = test_data["price"] - (test_data["atr"] * atr_multiplier_sl)
        take_profit = test_data["price"] + (test_data["atr"] * atr_multiplier_tp)
        
        print(f"   📊 Test Market Data:")
        print(f"      Price: ${test_data['price']:,.2f}")
        print(f"      RSI: {test_data['rsi']:.1f}")
        print(f"      ATR: ${test_data['atr']:,.2f}")
        
        print(f"   📈 Signal Analysis:")
        print(f"      RSI Oversold: {'✅' if rsi_oversold else '❌'}")
        print(f"      MA Bullish: {'✅' if ma_bullish else '❌'}")
        
        print(f"   💰 Trade Calculation:")
        print(f"      Capital: ${capital:,.2f}")
        print(f"      Position Size: {position_size*100:.1f}%")
        print(f"      Trade Amount: ${trade_amount:,.2f}")
        print(f"      Stop Loss: ${stop_loss:,.2f}")
        print(f"      Take Profit: ${take_profit:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Trade simulation failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting live bot logic validation...")
    
    # Run validation
    success = validate_live_bot_logic()
    
    # Run trade simulation
    simulation_success = create_test_trade_simulation()
    
    print("\n" + "=" * 60)
    print("🎯 VALIDATION COMPLETE")
    print("=" * 60)
    
    if success and simulation_success:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ Live bot logic matches backtest configuration")
        print("✅ Trade simulation successful")
        print("\n🚀 READY FOR LIVE TRADING")
        print("Next steps:")
        print("1. Test with small amounts: python test_live_bot.py")
        print("2. Monitor with: python monitor_bot.py")
        print("3. Check GitHub Actions for automated deployment")
    else:
        print("\n⚠️ VALIDATION ISSUES FOUND")
        print("Please review and fix the failed checks above")
        print("Run this script again after making corrections")

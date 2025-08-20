#!/usr/bin/env python3
"""
AUTOMATED TRADING PIPELINE
Connects backtest → live bot → results monitoring

This creates a fully automated system where:
1. Backtest/watcher generates optimal parameters
2. Automatically syncs them to live bot (Vercel)
3. Live bot uses optimized parameters for trading
4. Results flow back to local system for monitoring
"""

import json
import os
import requests
import time
import shutil
from datetime import datetime
from pathlib import Path

class AutomatedTradingPipeline:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.config_file = self.base_dir / "live_trading_config.json"
        self.results_dir = self.base_dir / "live_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Vercel endpoints
        self.vercel_base = "https://my-trading-robot-1.vercel.app"
        self.live_bot_endpoint = f"{self.vercel_base}/api/live-bot"
        self.auth_headers = {"Authorization": "Bearer 93699b3917045092715b8e16c01f2e1d"}
        
        # Parameter sync settings
        self.sync_interval = 300  # 5 minutes
        self.results_fetch_interval = 60  # 1 minute
        
    def load_latest_backtest_params(self):
        """Load the most recent optimized parameters from backtest"""
        try:
            # Check for latest backtest results
            runs_dir = self.base_dir / "runs"
            if not runs_dir.exists():
                print("⚠️ No backtest runs directory found")
                return None
            
            # Find most recent run
            run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
            if not run_dirs:
                print("⚠️ No backtest run directories found")
                return None
            
            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
            
            # Look for optimized parameters file
            param_files = [
                latest_run / "optimized_parameters.json",
                latest_run / "final_params.json",
                latest_run / "best_params.json"
            ]
            
            for param_file in param_files:
                if param_file.exists():
                    with open(param_file, 'r') as f:
                        params = json.load(f)
                    print(f"✅ Loaded optimized parameters from: {param_file}")
                    return params
            
            print("⚠️ No optimized parameters file found in latest run")
            return None
            
        except Exception as e:
            print(f"❌ Error loading backtest parameters: {e}")
            return None
    
    def create_live_trading_config(self, backtest_params):
        """Convert backtest parameters to live trading config"""
        if not backtest_params:
            return None
        
        # Base trading configuration
        live_config = {
            # Core trading parameters
            "SYMBOL": "BTCUSDT",
            "TIMEFRAME": "5m",
            "INITIAL_CAPITAL": 10000,
            "POSITION_SIZE": 0.02,
            "COMMISSION_RATE": 0.001,
            "SLIPPAGE_RATE": 0.0001,
            "MAX_PORTFOLIO_RISK": 0.15,
            "min_confidence_for_trade": 0.04,
            "USE_ML_REGIME_DETECTION": True,
            "BLOCK_LOW_CONFIDENCE_SIGNALS": True,
            "VOLUME_CONFIRMATION": True,
            
            # Timestamp for tracking
            "updated_at": datetime.now().isoformat(),
            "source": "automated_backtest_sync"
        }
        
        # Copy optimized parameters
        param_mapping = {
            "RSI_PERIOD": "RSI_PERIOD",
            "RSI_OVERBOUGHT": "RSI_OVERBOUGHT", 
            "RSI_OVERSOLD": "RSI_OVERSOLD",
            "MA_FAST": "MA_FAST",
            "MA_SLOW": "MA_SLOW",
            "MA_SIGNAL": "MA_SIGNAL",
            "TENKAN_SEN_PERIOD": "TENKAN_SEN_PERIOD",
            "KIJUN_SEN_PERIOD": "KIJUN_SEN_PERIOD",
            "SENKOU_SPAN_B_PERIOD": "SENKOU_SPAN_B_PERIOD",
            "ADX_THRESHOLD": "ADX_THRESHOLD",
            "ATR_PERIOD": "ATR_PERIOD",
            "volatility_threshold": "volatility_threshold",
            "volatility_window": "volatility_window",
            "trend_window": "trend_window"
        }
        
        for backtest_key, live_key in param_mapping.items():
            if backtest_key in backtest_params:
                live_config[live_key] = backtest_params[backtest_key]
        
        return live_config
    
    def sync_parameters_to_vercel(self, live_config):
        """Sync parameters to Vercel for live bot use"""
        try:
            # Save config locally first
            with open(self.config_file, 'w') as f:
                json.dump(live_config, f, indent=2)
            print(f"✅ Saved live config locally: {self.config_file}")
            
            # Sync to Vercel via API
            sync_endpoint = f"{self.vercel_base}/api/parameter-sync"
            sync_payload = {
                "action": "update_parameters",
                "config": live_config,
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(
                sync_endpoint, 
                json=sync_payload,
                headers=self.auth_headers,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Parameters synced to Vercel successfully")
                print(f"   Updated at: {result.get('timestamp')}")
                return True
            else:
                print(f"⚠️ Vercel sync failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False
            
        except Exception as e:
            print(f"❌ Error syncing parameters to Vercel: {e}")
            return False
    
    def fetch_live_results(self):
        """Fetch current results from live bot"""
        try:
            response = requests.get(self.live_bot_endpoint, headers=self.auth_headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract key metrics
                results = {
                    "timestamp": datetime.now().isoformat(),
                    "signal": data.get("signal", {}),
                    "account_balance": data.get("account_balance", {}),
                    "trade_executed": data.get("trade_executed", {}),
                    "performance_metrics": data.get("performance_metrics", {}),
                    "system_status": data.get("system_status", {})
                }
                
                # Save results
                results_file = self.results_dir / f"live_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                return results
            else:
                print(f"⚠️ Live bot API returned status: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching live results: {e}")
            return None
    
    def create_daily_summary(self):
        """Create daily summary of live trading performance"""
        try:
            today = datetime.now().strftime('%Y%m%d')
            today_files = list(self.results_dir.glob(f"live_results_{today}_*.json"))
            
            if not today_files:
                return None
            
            # Aggregate daily data
            daily_summary = {
                "date": today,
                "total_queries": len(today_files),
                "signals": {"BUY": 0, "SELL": 0, "HOLD": 0},
                "balance_start": None,
                "balance_end": None,
                "trades_executed": 0,
                "avg_confidence": 0,
                "created_at": datetime.now().isoformat()
            }
            
            confidences = []
            
            for result_file in sorted(today_files):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # Track signals
                signal = data.get("signal", {}).get("signal", "HOLD")
                if signal in daily_summary["signals"]:
                    daily_summary["signals"][signal] += 1
                
                # Track confidence
                confidence = data.get("signal", {}).get("confidence", 0)
                if confidence > 0:
                    confidences.append(confidence)
                
                # Track balance changes
                balance = data.get("account_balance", {})
                if balance and daily_summary["balance_start"] is None:
                    daily_summary["balance_start"] = balance
                if balance:
                    daily_summary["balance_end"] = balance
                
                # Track trades
                trade = data.get("trade_executed", {})
                if trade and not trade.get("simulated", True):
                    daily_summary["trades_executed"] += 1
            
            if confidences:
                daily_summary["avg_confidence"] = sum(confidences) / len(confidences)
            
            # Save daily summary
            summary_file = self.results_dir / f"daily_summary_{today}.json"
            with open(summary_file, 'w') as f:
                json.dump(daily_summary, f, indent=2)
            
            return daily_summary
            
        except Exception as e:
            print(f"❌ Error creating daily summary: {e}")
            return None
    
    def run_parameter_sync_cycle(self):
        """Run one cycle of parameter synchronization"""
        print(f"\n🔄 PARAMETER SYNC CYCLE - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        # 1. Load latest backtest parameters
        backtest_params = self.load_latest_backtest_params()
        if not backtest_params:
            print("⚠️ No backtest parameters available, skipping sync")
            return False
        
        # 2. Create live trading config
        live_config = self.create_live_trading_config(backtest_params)
        if not live_config:
            print("❌ Failed to create live trading config")
            return False
        
        # 3. Sync to Vercel
        sync_success = self.sync_parameters_to_vercel(live_config)
        if sync_success:
            print("✅ Parameter sync cycle completed successfully")
            
            # Show key parameters
            print(f"\n📊 Synchronized Parameters:")
            key_params = ["RSI_PERIOD", "MA_FAST", "MA_SLOW", "min_confidence_for_trade"]
            for param in key_params:
                if param in live_config:
                    print(f"   {param}: {live_config[param]}")
        
        return sync_success
    
    def run_results_monitoring_cycle(self):
        """Run one cycle of results monitoring"""
        print(f"\n📊 RESULTS MONITORING CYCLE - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        # Fetch current results
        results = self.fetch_live_results()
        if results:
            signal = results.get("signal", {})
            balance = results.get("account_balance", {})
            
            print(f"📡 Current Signal: {signal.get('signal', 'N/A')} ({signal.get('confidence', 0):.3f} confidence)")
            print(f"💰 USDT Balance: ${balance.get('USDT', 0):,.2f}")
            print(f"🪙 BTC Balance: {balance.get('BTC', 0):.6f}")
            
            # Create daily summary
            daily_summary = self.create_daily_summary()
            if daily_summary:
                print(f"📋 Daily Summary: {daily_summary['total_queries']} queries, {daily_summary['trades_executed']} trades")
        
        return results is not None
    
    def run_automated_pipeline(self, cycles=None):
        """Run the complete automated pipeline"""
        print("🚀 STARTING AUTOMATED TRADING PIPELINE")
        print("=" * 70)
        print(f"🔄 Parameter Sync Interval: {self.sync_interval} seconds")
        print(f"📊 Results Monitoring Interval: {self.results_fetch_interval} seconds")
        print(f"🌐 Live Bot Endpoint: {self.live_bot_endpoint}")
        print("=" * 70)
        
        cycle_count = 0
        last_param_sync = 0
        last_results_fetch = 0
        
        try:
            while cycles is None or cycle_count < cycles:
                current_time = time.time()
                
                # Parameter sync cycle
                if current_time - last_param_sync >= self.sync_interval:
                    self.run_parameter_sync_cycle()
                    last_param_sync = current_time
                
                # Results monitoring cycle
                if current_time - last_results_fetch >= self.results_fetch_interval:
                    self.run_results_monitoring_cycle()
                    last_results_fetch = current_time
                
                # Brief pause
                time.sleep(10)
                cycle_count += 1
                
        except KeyboardInterrupt:
            print(f"\n🛑 Pipeline stopped by user")
        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
        
        print(f"\n✅ Automated pipeline completed {cycle_count} cycles")

def main():
    """Main entry point"""
    pipeline = AutomatedTradingPipeline()
    
    # Check if this is a one-time sync or continuous monitoring
    import sys
    if "--sync-once" in sys.argv:
        print("🔄 Running one-time parameter sync...")
        pipeline.run_parameter_sync_cycle()
    elif "--monitor-once" in sys.argv:
        print("📊 Running one-time results monitoring...")
        pipeline.run_results_monitoring_cycle()
    elif "--test" in sys.argv:
        print("🧪 Running test cycles...")
        pipeline.run_automated_pipeline(cycles=3)
    else:
        print("🔄 Running continuous automated pipeline...")
        pipeline.run_automated_pipeline()

if __name__ == "__main__":
    main()

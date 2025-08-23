#!/usr/bin/env python3
"""
Comprehensive analysis of ML model faults and their potential impact on backtesting performance.
This script identifies specific weaknesses in our ML models that could hinder backtesting.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from core.production_regime_detector import ProductionRegimeDetector as MLMarketRegimeDetector
from core.strategy import MarketRegime
import warnings
warnings.filterwarnings('ignore')

class MLFaultAnalyzer:
    def __init__(self):
        self.model_dir = Path("ml_models")
        self.enhanced_model = None
        self.ensemble_model = None
        self.scaler = None
        self.test_data = None
        self.current_detector = None
        
    def load_models(self):
        """Load all ML models for comparison"""
        try:
            # Load enhanced model
            with open(self.model_dir / "enhanced_regime_classifier.pkl", 'rb') as f:
                self.enhanced_model = pickle.load(f)
            print("✅ Enhanced model loaded")
            
            # Load ensemble model  
            with open(self.model_dir / "ensemble_regime_classifier.pkl", 'rb') as f:
                self.ensemble_model = pickle.load(f)
            print("✅ Ensemble model loaded")
            
            # Load scaler
            with open(self.model_dir / "feature_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Feature scaler loaded")
            
            # Load current detector for comparison
            self.current_detector = MLMarketRegimeDetector()
            print("✅ Current detector loaded")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
        return True
    
    def generate_test_data(self):
        """Generate comprehensive test data covering all market scenarios"""
        np.random.seed(42)
        n_samples = 2000
        
        # Create realistic market data scenarios
        scenarios = []
        
        # 1. Strong trending bull market
        bull_trend = self._create_scenario('bull_trend', 200, trend=0.02, volatility=0.015)
        scenarios.append(('TRENDING_BULL', bull_trend))
        
        # 2. Strong trending bear market  
        bear_trend = self._create_scenario('bear_trend', 200, trend=-0.018, volatility=0.02)
        scenarios.append(('TRENDING_BEAR', bear_trend))
        
        # 3. Sideways ranging market
        ranging = self._create_scenario('ranging', 300, trend=0.0, volatility=0.01)
        scenarios.append(('RANGING', ranging))
        
        # 4. High volatility periods
        high_vol = self._create_scenario('high_vol', 200, trend=0.005, volatility=0.04)
        scenarios.append(('HIGH_VOLATILITY', high_vol))
        
        # 5. Low volatility periods
        low_vol = self._create_scenario('low_vol', 400, trend=0.002, volatility=0.008)
        scenarios.append(('LOW_VOLATILITY', low_vol))
        
        # 6. Bullish breakout scenarios (rare but critical)
        bull_breakout = self._create_breakout_scenario('bull_breakout', 150, direction='up')
        scenarios.append(('BREAKOUT_BULLISH', bull_breakout))
        
        # 7. Bearish breakout scenarios
        bear_breakout = self._create_breakout_scenario('bear_breakout', 150, direction='down')
        scenarios.append(('BREAKOUT_BEARISH', bear_breakout))
        
        # 8. Accumulation periods
        accumulation = self._create_scenario('accumulation', 200, trend=0.001, volatility=0.012)
        scenarios.append(('ACCUMULATION', accumulation))
        
        # 9. Distribution periods  
        distribution = self._create_scenario('distribution', 200, trend=-0.001, volatility=0.015)
        scenarios.append(('DISTRIBUTION', distribution))
        
        # Combine all scenarios
        all_data = []
        all_labels = []
        
        for regime_name, data in scenarios:
            all_data.append(data)
            all_labels.extend([regime_name] * len(data))
        
        self.test_data = pd.concat(all_data, ignore_index=True)
        self.test_labels = all_labels
        
        print(f"✅ Generated {len(self.test_data)} test samples across {len(scenarios)} market regimes")
        return True
    
    def _create_scenario(self, name, length, trend, volatility):
        """Create market data for a specific scenario"""
        # Generate price data with specified characteristics
        returns = np.random.normal(trend, volatility, length)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create volume data correlated with price moves
        volume_base = 1000000
        volume_multiplier = 1 + 0.5 * np.abs(returns) / volatility
        volume = volume_base * volume_multiplier * (1 + 0.2 * np.random.randn(length))
        
        # Create OHLC data
        data = pd.DataFrame({
            'open': prices * (1 + 0.001 * np.random.randn(length)),
            'high': prices * (1 + np.maximum(0, 0.002 * np.random.randn(length))),
            'low': prices * (1 - np.maximum(0, 0.002 * np.random.randn(length))),
            'close': prices,
            'volume': volume
        })
        
        # Ensure high >= close >= low
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        
        return data
    
    def _create_breakout_scenario(self, name, length, direction):
        """Create breakout scenario with sudden price movement"""
        # Pre-breakout consolidation
        consolidation_length = length // 3
        consolidation = self._create_scenario(f'{name}_consolidation', consolidation_length, 0.0, 0.008)
        
        # Breakout phase
        breakout_length = length // 3
        breakout_trend = 0.03 if direction == 'up' else -0.03
        breakout = self._create_scenario(f'{name}_breakout', breakout_length, breakout_trend, 0.025)
        
        # Post-breakout continuation
        continuation_length = length - consolidation_length - breakout_length
        continuation_trend = 0.015 if direction == 'up' else -0.015
        continuation = self._create_scenario(f'{name}_continuation', continuation_length, continuation_trend, 0.02)
        
        # Adjust volume for breakout (higher during breakout)
        breakout['volume'] *= 2.5
        
        return pd.concat([consolidation, breakout, continuation], ignore_index=True)
    
    def analyze_model_faults(self):
        """Comprehensive analysis of model faults and weaknesses"""
        print("\n" + "="*60)
        print("🔍 COMPREHENSIVE ML FAULT ANALYSIS")
        print("="*60)
        
        faults = {}
        
        # 1. Test enhanced model performance
        enhanced_faults = self._test_model_performance(self.enhanced_model, "Enhanced Model")
        faults['enhanced'] = enhanced_faults
        
        # 2. Test ensemble model performance  
        ensemble_faults = self._test_model_performance(self.ensemble_model, "Ensemble Model")
        faults['ensemble'] = ensemble_faults
        
        # 3. Class imbalance analysis
        class_balance_faults = self._analyze_class_imbalance()
        faults['class_balance'] = class_balance_faults
        
        # 4. Feature importance stability
        feature_stability_faults = self._analyze_feature_stability()
        faults['feature_stability'] = feature_stability_faults
        
        # 5. Prediction confidence analysis
        confidence_faults = self._analyze_prediction_confidence()
        faults['confidence'] = confidence_faults
        
        # 6. Performance in critical trading scenarios
        trading_faults = self._analyze_trading_critical_scenarios()
        faults['trading_critical'] = trading_faults
        
        return faults
    
    def _test_model_performance(self, model, model_name):
        """Test model performance and identify specific faults"""
        print(f"\n📊 Testing {model_name}")
        print("-" * 40)
        
        try:
            # Extract features using the enhanced feature engineering
            features = self._extract_enhanced_features(self.test_data)
            if features is None:
                return {"error": "Feature extraction failed"}
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make predictions
            predictions = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            accuracy = accuracy_score(self.test_labels, predictions)
            macro_f1 = f1_score(self.test_labels, predictions, average='macro')
            precision = precision_score(self.test_labels, predictions, average='macro', zero_division=0)
            recall = recall_score(self.test_labels, predictions, average='macro', zero_division=0)
            
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Macro F1: {macro_f1:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            
            # Identify specific faults
            faults = []
            
            # Low overall accuracy fault
            if accuracy < 0.5:
                faults.append({
                    "fault": "LOW_ACCURACY",
                    "severity": "HIGH",
                    "value": accuracy,
                    "impact": "Poor regime classification will lead to wrong trading parameters"
                })
            
            # Low macro F1 fault (indicates poor performance on minority classes)
            if macro_f1 < 0.4:
                faults.append({
                    "fault": "POOR_MINORITY_CLASS_PERFORMANCE", 
                    "severity": "HIGH",
                    "value": macro_f1,
                    "impact": "Critical regimes (breakouts) will be missed, leading to poor risk management"
                })
            
            # Class-specific performance analysis
            report = classification_report(self.test_labels, predictions, output_dict=True, zero_division=0)
            
            critical_regimes = ['BREAKOUT_BULLISH', 'BREAKOUT_BEARISH', 'HIGH_VOLATILITY']
            for regime in critical_regimes:
                if regime in report:
                    if report[regime]['precision'] < 0.1:  # Less than 10% precision
                        faults.append({
                            "fault": f"CRITICAL_REGIME_DETECTION_FAILURE",
                            "regime": regime,
                            "severity": "CRITICAL",
                            "precision": report[regime]['precision'],
                            "recall": report[regime]['recall'],
                            "impact": f"Complete failure to detect {regime} will cause massive losses during these periods"
                        })
            
            # Prediction confidence analysis
            max_probabilities = np.max(probabilities, axis=1)
            low_confidence_ratio = np.mean(max_probabilities < 0.6)
            
            if low_confidence_ratio > 0.4:  # More than 40% low confidence predictions
                faults.append({
                    "fault": "HIGH_UNCERTAINTY_PREDICTIONS",
                    "severity": "MEDIUM", 
                    "value": low_confidence_ratio,
                    "impact": "Unreliable regime detection will cause parameter instability during trading"
                })
            
            return {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "precision": precision,
                "recall": recall,
                "faults": faults,
                "detailed_report": report
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_enhanced_features(self, data):
        """Extract features using the same method as enhanced model training"""
        try:
            # This should match the feature extraction in simplified_ml_improvements.py
            features_list = []
            
            for i in range(len(data)):
                if i < 20:  # Need minimum data for feature calculation
                    continue
                    
                window_data = data.iloc[max(0, i-19):i+1].copy()
                if len(window_data) < 20:
                    continue
                
                features = self._calculate_enhanced_features(window_data)
                if features is not None:
                    features_list.append(features)
            
            if not features_list:
                return None
                
            return np.array(features_list)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _calculate_enhanced_features(self, data):
        """Calculate the 32 enhanced features for a single data window"""
        try:
            import ta
            
            if len(data) < 20:
                return None
            
            features = {}
            
            # Basic price features
            features['returns'] = (data['close'].iloc[-1] / data['close'].iloc[-2] - 1) if len(data) >= 2 else 0
            features['log_returns'] = np.log(data['close'].iloc[-1] / data['close'].iloc[-2]) if len(data) >= 2 else 0
            features['price_sma_ratio'] = data['close'].iloc[-1] / data['close'].rolling(10).mean().iloc[-1] if len(data) >= 10 else 1
            features['high_low_ratio'] = (data['high'].iloc[-1] - data['low'].iloc[-1]) / data['close'].iloc[-1] if data['close'].iloc[-1] != 0 else 0
            
            # Technical indicators
            features['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi().iloc[-1] / 100 if len(data) >= 14 else 0.5
            features['adx'] = ta.trend.ADXIndicator(data['high'], data['low'], data['close']).adx().iloc[-1] / 100 if len(data) >= 14 else 0
            
            bb = ta.volatility.BollingerBands(data['close'])
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            features['bb_width'] = (bb_upper - bb_lower) / data['close'].iloc[-1] if data['close'].iloc[-1] != 0 else 0
            
            features['atr'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close']).average_true_range().iloc[-1] / data['close'].iloc[-1] if data['close'].iloc[-1] != 0 else 0
            
            # Volume features
            features['volume_sma_ratio'] = data['volume'].iloc[-1] / data['volume'].rolling(10).mean().iloc[-1] if len(data) >= 10 else 1
            features['price_volume_trend'] = ta.volume.VolumePriceTrendIndicator(data['close'], data['volume']).volume_price_trend().iloc[-1] / 1000000
            
            # More technical indicators
            macd = ta.trend.MACD(data['close'])
            features['macd'] = macd.macd().iloc[-1] / data['close'].iloc[-1] if data['close'].iloc[-1] != 0 else 0
            features['cci'] = ta.trend.CCIIndicator(data['high'], data['low'], data['close']).cci().iloc[-1] / 100
            features['stoch'] = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close']).stoch().iloc[-1] / 100
            
            # Advanced features (simplified versions)
            features['vp_divergence'] = self._simple_vp_divergence(data)
            features['vwap_deviation'] = self._simple_vwap_deviation(data)
            features['ad_price_divergence'] = self._simple_ad_divergence(data)
            features['obv_momentum'] = self._simple_obv_momentum(data)
            
            # Momentum alignment features
            sma_5 = data['close'].rolling(5).mean().iloc[-1]
            sma_10 = data['close'].rolling(10).mean().iloc[-1]
            sma_20 = data['close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else sma_10
            
            features['momentum_alignment'] = 1 if sma_5 > sma_10 > sma_20 else (-1 if sma_5 < sma_10 < sma_20 else 0)
            features['momentum_strength'] = abs(sma_5 - sma_20) / sma_20 if sma_20 != 0 else 0
            
            # Rate of change
            roc_periods = min(10, len(data) - 1)
            features['roc_acceleration'] = (data['close'].iloc[-1] / data['close'].iloc[-roc_periods] - 1) if roc_periods > 0 else 0
            
            # Volatility features
            volatility = data['close'].pct_change().std()
            features['vol_regime'] = min(volatility * 100, 5) / 5  # Normalize to 0-1
            features['vol_persistence'] = volatility / data['close'].pct_change().rolling(5).std().mean() if len(data) >= 5 else 1
            features['vol_clustering'] = self._simple_vol_clustering(data)
            features['intraday_vs_daily_vol'] = self._simple_intraday_vol(data)
            
            # Breakout features
            features['breakout_strength_up'] = self._simple_breakout_strength(data, 'up')
            features['breakout_strength_down'] = self._simple_breakout_strength(data, 'down')
            features['breakout_volume_confirm'] = self._simple_volume_confirm(data)
            features['range_position'] = self._simple_range_position(data)
            
            # Order flow features
            features['buy_pressure'] = self._simple_buy_pressure(data)
            features['sell_pressure'] = self._simple_sell_pressure(data)
            features['vw_buy_pressure'] = features['buy_pressure'] * features['volume_sma_ratio']
            features['vw_sell_pressure'] = features['sell_pressure'] * features['volume_sma_ratio']
            
            # Convert to array
            feature_array = [features[key] for key in sorted(features.keys())]
            return np.array(feature_array)
            
        except Exception as e:
            return None
    
    def _simple_vp_divergence(self, data):
        """Simple volume-price divergence calculation"""
        try:
            price_change = data['close'].iloc[-1] - data['close'].iloc[-5] if len(data) >= 5 else 0
            volume_change = data['volume'].iloc[-1] - data['volume'].iloc[-5] if len(data) >= 5 else 0
            
            if price_change > 0 and volume_change < 0:
                return -1  # Bearish divergence
            elif price_change < 0 and volume_change > 0:
                return 1   # Bullish divergence
            else:
                return 0   # No divergence
        except:
            return 0
    
    def _simple_vwap_deviation(self, data):
        """Simple VWAP deviation calculation"""
        try:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
            return (data['close'].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] if vwap.iloc[-1] != 0 else 0
        except:
            return 0
    
    def _simple_ad_divergence(self, data):
        """Simple A/D line divergence"""
        try:
            import ta
            ad_line = ta.volume.AccDistIndexIndicator(data['high'], data['low'], data['close'], data['volume']).acc_dist_index()
            ad_change = ad_line.iloc[-1] - ad_line.iloc[-5] if len(ad_line) >= 5 else 0
            price_change = data['close'].iloc[-1] - data['close'].iloc[-5] if len(data) >= 5 else 0
            
            if (price_change > 0 and ad_change < 0) or (price_change < 0 and ad_change > 0):
                return 1  # Divergence detected
            return 0
        except:
            return 0
    
    def _simple_obv_momentum(self, data):
        """Simple OBV momentum calculation"""
        try:
            import ta
            obv = ta.volume.OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
            obv_change = (obv.iloc[-1] - obv.iloc[-5]) / obv.iloc[-5] if len(obv) >= 5 and obv.iloc[-5] != 0 else 0
            return max(-1, min(1, obv_change))  # Clip to [-1, 1]
        except:
            return 0
    
    def _simple_vol_clustering(self, data):
        """Simple volatility clustering measure"""
        try:
            returns = data['close'].pct_change().dropna()
            vol = returns.rolling(5).std()
            return vol.std() / vol.mean() if len(vol) > 0 and vol.mean() != 0 else 0
        except:
            return 0
    
    def _simple_intraday_vol(self, data):
        """Simple intraday vs daily volatility ratio"""
        try:
            intraday_vol = (data['high'] - data['low']) / data['close']
            daily_vol = data['close'].pct_change().abs()
            return intraday_vol.mean() / daily_vol.mean() if daily_vol.mean() != 0 else 1
        except:
            return 1
    
    def _simple_breakout_strength(self, data, direction):
        """Simple breakout strength calculation"""
        try:
            period = min(20, len(data))
            if direction == 'up':
                resistance = data['high'].rolling(period).max().iloc[-1]
                return max(0, (data['close'].iloc[-1] - resistance) / resistance) if resistance != 0 else 0
            else:
                support = data['low'].rolling(period).min().iloc[-1]
                return max(0, (support - data['close'].iloc[-1]) / support) if support != 0 else 0
        except:
            return 0
    
    def _simple_volume_confirm(self, data):
        """Simple volume confirmation for breakouts"""
        try:
            vol_avg = data['volume'].rolling(10).mean().iloc[-1]
            current_vol = data['volume'].iloc[-1]
            return current_vol / vol_avg if vol_avg != 0 else 1
        except:
            return 1
    
    def _simple_range_position(self, data):
        """Simple range position calculation"""
        try:
            period = min(20, len(data))
            range_high = data['high'].rolling(period).max().iloc[-1]
            range_low = data['low'].rolling(period).min().iloc[-1]
            if range_high != range_low:
                return (data['close'].iloc[-1] - range_low) / (range_high - range_low)
            return 0.5
        except:
            return 0.5
    
    def _simple_buy_pressure(self, data):
        """Simple buy pressure calculation"""
        try:
            close_pos = (data['close'] - data['low']) / (data['high'] - data['low'])
            return close_pos.rolling(5).mean().iloc[-1] if len(close_pos) >= 5 else 0.5
        except:
            return 0.5
    
    def _simple_sell_pressure(self, data):
        """Simple sell pressure calculation"""  
        try:
            return 1 - self._simple_buy_pressure(data)
        except:
            return 0.5
    
    def _analyze_class_imbalance(self):
        """Analyze class imbalance issues"""
        print(f"\n⚖️ Class Imbalance Analysis")
        print("-" * 40)
        
        from collections import Counter
        label_counts = Counter(self.test_labels)
        total_samples = len(self.test_labels)
        
        imbalance_faults = []
        
        for regime, count in label_counts.items():
            ratio = count / total_samples
            print(f"{regime}: {count} samples ({ratio:.1%})")
            
            if ratio < 0.05:  # Less than 5% representation
                imbalance_faults.append({
                    "fault": "SEVERE_CLASS_UNDERREPRESENTATION",
                    "regime": regime,
                    "severity": "HIGH",
                    "ratio": ratio,
                    "impact": f"Only {ratio:.1%} of data - model will rarely predict this regime correctly"
                })
            elif ratio < 0.10:  # Less than 10% representation
                imbalance_faults.append({
                    "fault": "MODERATE_CLASS_UNDERREPRESENTATION", 
                    "regime": regime,
                    "severity": "MEDIUM",
                    "ratio": ratio,
                    "impact": f"Only {ratio:.1%} of data - model may miss important instances"
                })
        
        return imbalance_faults
    
    def _analyze_feature_stability(self):
        """Analyze feature importance stability"""
        print(f"\n🔧 Feature Stability Analysis")
        print("-" * 40)
        
        stability_faults = []
        
        try:
            # Get feature importance if available
            if hasattr(self.enhanced_model, 'feature_importances_'):
                importances = self.enhanced_model.feature_importances_
                
                # Check for extremely dominant features
                max_importance = np.max(importances)
                if max_importance > 0.3:  # Single feature dominates
                    stability_faults.append({
                        "fault": "FEATURE_DOMINANCE",
                        "severity": "MEDIUM",
                        "max_importance": max_importance,
                        "impact": "Over-reliance on single feature makes model brittle"
                    })
                
                # Check for too many irrelevant features
                irrelevant_features = np.sum(importances < 0.01)
                if irrelevant_features > len(importances) * 0.5:
                    stability_faults.append({
                        "fault": "TOO_MANY_IRRELEVANT_FEATURES",
                        "severity": "LOW",
                        "irrelevant_count": irrelevant_features,
                        "total_features": len(importances),
                        "impact": "Model complexity without benefit - potential overfitting"
                    })
                
                print(f"Max feature importance: {max_importance:.3f}")
                print(f"Irrelevant features (< 1%): {irrelevant_features}/{len(importances)}")
        
        except Exception as e:
            stability_faults.append({
                "fault": "FEATURE_IMPORTANCE_UNAVAILABLE",
                "severity": "LOW", 
                "error": str(e),
                "impact": "Cannot assess feature stability"
            })
        
        return stability_faults
    
    def _analyze_prediction_confidence(self):
        """Analyze prediction confidence patterns"""
        print(f"\n🎯 Prediction Confidence Analysis")
        print("-" * 40)
        
        confidence_faults = []
        
        try:
            # Extract features and make predictions
            features = self._extract_enhanced_features(self.test_data)
            if features is None:
                return [{"fault": "FEATURE_EXTRACTION_FAILED", "severity": "HIGH"}]
            
            features_scaled = self.scaler.transform(features)
            probabilities = self.enhanced_model.predict_proba(features_scaled)
            
            # Analyze confidence patterns
            max_probs = np.max(probabilities, axis=1)
            avg_confidence = np.mean(max_probs)
            low_confidence_ratio = np.mean(max_probs < 0.5)  # Worse than random
            very_low_confidence_ratio = np.mean(max_probs < 0.4)  # Very uncertain
            
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Low confidence predictions: {low_confidence_ratio:.1%}")
            print(f"Very low confidence predictions: {very_low_confidence_ratio:.1%}")
            
            if avg_confidence < 0.6:
                confidence_faults.append({
                    "fault": "LOW_AVERAGE_CONFIDENCE",
                    "severity": "HIGH",
                    "avg_confidence": avg_confidence,
                    "impact": "Model is generally uncertain - unreliable for trading decisions"
                })
            
            if low_confidence_ratio > 0.3:
                confidence_faults.append({
                    "fault": "HIGH_UNCERTAINTY_RATIO",
                    "severity": "HIGH", 
                    "low_confidence_ratio": low_confidence_ratio,
                    "impact": "Too many uncertain predictions - will cause parameter instability"
                })
        
        except Exception as e:
            confidence_faults.append({
                "fault": "CONFIDENCE_ANALYSIS_FAILED",
                "severity": "MEDIUM",
                "error": str(e)
            })
        
        return confidence_faults
    
    def _analyze_trading_critical_scenarios(self):
        """Analyze performance in trading-critical scenarios"""
        print(f"\n💰 Trading-Critical Scenario Analysis")
        print("-" * 40)
        
        trading_faults = []
        
        # Define critical scenarios for trading
        critical_scenarios = {
            'BREAKOUT_BULLISH': "Missing bull breakouts causes lost profits",
            'BREAKOUT_BEARISH': "Missing bear breakouts causes major losses", 
            'HIGH_VOLATILITY': "Wrong regime during volatility spikes = disaster",
            'TRENDING_BULL': "Missing bull trends reduces profitability",
            'TRENDING_BEAR': "Missing bear trends increases losses"
        }
        
        try:
            # Test model on critical scenarios
            features = self._extract_enhanced_features(self.test_data)
            if features is None:
                return [{"fault": "FEATURE_EXTRACTION_FAILED", "severity": "HIGH"}]
            
            features_scaled = self.scaler.transform(features)
            predictions = self.enhanced_model.predict(features_scaled)
            
            # Calculate precision/recall for each critical scenario
            from sklearn.metrics import precision_recall_fscore_support
            
            unique_labels = list(set(self.test_labels))
            precision, recall, f1, support = precision_recall_fscore_support(
                self.test_labels, predictions, labels=unique_labels, zero_division=0
            )
            
            for i, label in enumerate(unique_labels):
                if label in critical_scenarios:
                    print(f"{label}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}")
                    
                    # Critical faults for trading scenarios
                    if precision[i] < 0.2:  # Less than 20% precision
                        trading_faults.append({
                            "fault": "CRITICAL_SCENARIO_PRECISION_FAILURE",
                            "regime": label,
                            "severity": "CRITICAL",
                            "precision": precision[i],
                            "impact": critical_scenarios[label]
                        })
                    
                    if recall[i] < 0.3:  # Less than 30% recall
                        trading_faults.append({
                            "fault": "CRITICAL_SCENARIO_RECALL_FAILURE",
                            "regime": label,
                            "severity": "HIGH", 
                            "recall": recall[i],
                            "impact": f"Missing {100-recall[i]*100:.0f}% of {label} scenarios"
                        })
        
        except Exception as e:
            trading_faults.append({
                "fault": "TRADING_SCENARIO_ANALYSIS_FAILED",
                "severity": "HIGH",
                "error": str(e)
            })
        
        return trading_faults
    
    def generate_fault_report(self, faults):
        """Generate comprehensive fault report"""
        print("\n" + "="*80)
        print("🚨 ML MODEL FAULT IMPACT ASSESSMENT")
        print("="*80)
        
        critical_faults = []
        high_faults = []
        medium_faults = []
        low_faults = []
        
        # Categorize all faults by severity
        for category, fault_list in faults.items():
            if isinstance(fault_list, dict) and 'faults' in fault_list:
                fault_list = fault_list['faults']
            elif not isinstance(fault_list, list):
                continue
                
            for fault in fault_list:
                severity = fault.get('severity', 'UNKNOWN')
                fault['category'] = category
                
                if severity == 'CRITICAL':
                    critical_faults.append(fault)
                elif severity == 'HIGH':
                    high_faults.append(fault)
                elif severity == 'MEDIUM':
                    medium_faults.append(fault)
                else:
                    low_faults.append(fault)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(critical_faults, high_faults, medium_faults)
        
        # Print summary
        print(f"\n📊 FAULT SUMMARY:")
        print(f"🔴 CRITICAL: {len(critical_faults)} faults")
        print(f"🟠 HIGH: {len(high_faults)} faults") 
        print(f"🟡 MEDIUM: {len(medium_faults)} faults")
        print(f"🟢 LOW: {len(low_faults)} faults")
        
        # Print critical faults in detail
        if critical_faults:
            print(f"\n🔴 CRITICAL FAULTS (WILL HINDER PERFORMANCE):")
            for i, fault in enumerate(critical_faults, 1):
                print(f"\n{i}. {fault['fault']}")
                print(f"   Impact: {fault['impact']}")
                if 'regime' in fault:
                    print(f"   Regime: {fault['regime']}")
                if 'precision' in fault:
                    print(f"   Precision: {fault['precision']:.3f}")
                if 'recall' in fault:
                    print(f"   Recall: {fault['recall']:.3f}")
        
        # Print high-priority faults
        if high_faults:
            print(f"\n🟠 HIGH PRIORITY FAULTS:")
            for i, fault in enumerate(high_faults, 1):
                print(f"\n{i}. {fault['fault']}")
                print(f"   Impact: {fault['impact']}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Overall verdict
        print(f"\n🎯 OVERALL VERDICT:")
        if len(critical_faults) >= 3:
            print("❌ DO NOT DEPLOY - Too many critical faults will severely hinder backtesting performance")
        elif len(critical_faults) >= 1 and len(high_faults) >= 3:
            print("⚠️ DEPLOY WITH CAUTION - Address critical faults first, monitor closely")
        elif len(high_faults) >= 5:
            print("⚠️ DEPLOY WITH IMPROVEMENTS - Address high-priority faults for better performance")
        else:
            print("✅ SAFE TO DEPLOY - Faults are manageable and won't significantly hinder performance")
        
        return {
            'critical': critical_faults,
            'high': high_faults, 
            'medium': medium_faults,
            'low': low_faults,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, critical_faults, high_faults, medium_faults):
        """Generate specific recommendations based on identified faults"""
        recommendations = []
        
        # Check for specific fault patterns
        has_breakout_failures = any('BREAKOUT' in str(fault) for fault in critical_faults + high_faults)
        has_confidence_issues = any('CONFIDENCE' in str(fault) for fault in critical_faults + high_faults)
        has_class_imbalance = any('UNDERREPRESENTATION' in str(fault) for fault in critical_faults + high_faults)
        
        if has_breakout_failures:
            recommendations.append("Collect more breakout training data or use specialized breakout detection models")
        
        if has_confidence_issues:
            recommendations.append("Implement confidence thresholds - fallback to rule-based detection for low-confidence predictions")
        
        if has_class_imbalance:
            recommendations.append("Use ensemble voting with current ML system for underrepresented regimes")
        
        if len(critical_faults) > 0:
            recommendations.append("Deploy enhanced ML gradually with A/B testing against current system")
        
        recommendations.append("Monitor regime detection accuracy in live trading and implement automatic fallbacks")
        
        return recommendations

def main():
    print("🔍 ML Model Fault Analysis")
    print("=" * 50)
    
    analyzer = MLFaultAnalyzer()
    
    # Load models
    if not analyzer.load_models():
        print("❌ Failed to load models")
        return
    
    # Generate test data
    if not analyzer.generate_test_data():
        print("❌ Failed to generate test data")
        return
    
    # Analyze faults
    faults = analyzer.analyze_model_faults()
    
    # Generate final report
    report = analyzer.generate_fault_report(faults)
    
    # Save detailed report
    with open('ml_fault_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: ml_fault_analysis_report.json")

if __name__ == "__main__":
    main()

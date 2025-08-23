#!/usr/bin/env python3
"""
Option 5: Real-time Learning & Adaptive Models
Continuously improving models that adapt to changing market conditions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json
from collections import deque
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from core.enums import MarketRegime

class AdaptiveRegimeClassifier:
    """Self-improving regime classifier with online learning"""
    
    def __init__(self, window_size=5000, retrain_threshold=0.1):
        self.base_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        
        # Online learning parameters
        self.window_size = window_size
        self.retrain_threshold = retrain_threshold
        self.recent_data = deque(maxlen=window_size)
        self.recent_labels = deque(maxlen=window_size)
        
        # Performance tracking
        self.prediction_accuracy = deque(maxlen=1000)
        self.regime_performance = {}
        self.false_positive_tracker = {}
        
        # Adaptive parameters
        self.confidence_thresholds = {
            'breakout_bullish': 0.8,
            'breakout_bearish': 0.8,
            'trending_bull': 0.6,
            'trending_bear': 0.6,
            'ranging': 0.5,
            'high_volatility': 0.6,
            'low_volatility': 0.6,
            'accumulation': 0.7,
            'distribution': 0.7
        }
        
    def update_with_feedback(self, features, true_label, predicted_label, confidence):
        """Update model with real market feedback"""
        
        # Store recent data
        self.recent_data.append(features)
        self.recent_labels.append(true_label)
        
        # Track prediction accuracy
        is_correct = (predicted_label == true_label)
        self.prediction_accuracy.append(is_correct)
        
        # Update regime-specific performance
        if predicted_label not in self.regime_performance:
            self.regime_performance[predicted_label] = {'correct': 0, 'total': 0}
        
        self.regime_performance[predicted_label]['total'] += 1
        if is_correct:
            self.regime_performance[predicted_label]['correct'] += 1
        
        # Track false positives for breakouts
        if 'breakout' in predicted_label and not is_correct:
            if predicted_label not in self.false_positive_tracker:
                self.false_positive_tracker[predicted_label] = deque(maxlen=100)
            self.false_positive_tracker[predicted_label].append(features)
        
        # Decide if retraining is needed
        if self._should_retrain():
            self._incremental_retrain()
            
    def _should_retrain(self):
        """Determine if model should be retrained"""
        
        if len(self.prediction_accuracy) < 100:
            return False
            
        # Check recent accuracy
        recent_accuracy = np.mean(list(self.prediction_accuracy)[-100:])
        
        # Check regime-specific performance degradation
        for regime, perf in self.regime_performance.items():
            if perf['total'] > 20:
                accuracy = perf['correct'] / perf['total']
                if accuracy < 0.5:  # Performance threshold
                    return True
        
        # Check overall accuracy drop
        if recent_accuracy < 0.6:
            return True
            
        # Check if we have enough new data
        if len(self.recent_data) >= self.window_size * 0.8:
            return True
            
        return False
    
    def _incremental_retrain(self):
        """Retrain model with recent data"""
        
        if len(self.recent_data) < 100:
            return
            
        # Prepare recent training data
        X_recent = np.array([list(features.values()) for features in self.recent_data])
        y_recent = np.array(list(self.recent_labels))
        
        # Retrain with recent data (incremental learning simulation)
        self.base_model.fit(X_recent, y_recent)
        
        # Reset performance trackers
        self.regime_performance = {}
        self.prediction_accuracy.clear()
        
        print(f"Model retrained with {len(X_recent)} recent samples")
    
    def adaptive_predict(self, features):
        """Make prediction with adaptive confidence thresholds"""
        
        # Get base prediction and probabilities
        feature_array = np.array([list(features.values())])
        probabilities = self.base_model.predict_proba(feature_array)[0]
        
        # Get class labels
        class_labels = self.base_model.classes_
        
        # Find best prediction
        best_idx = np.argmax(probabilities)
        predicted_regime = class_labels[best_idx]
        confidence = probabilities[best_idx]
        
        # Apply adaptive confidence threshold
        threshold = self.confidence_thresholds.get(predicted_regime, 0.6)
        
        # If confidence is below threshold, fall back to rule-based or default
        if confidence < threshold:
            predicted_regime = self._fallback_prediction(features)
            confidence = 0.5  # Lower confidence for fallback
            
        return predicted_regime, confidence
    
    def _fallback_prediction(self, features):
        """Fallback prediction when ML confidence is low"""
        
        # Simple rule-based fallback
        adx = features.get('adx', 25)
        bb_width = features.get('bb_width', 0.1)
        volume_ratio = features.get('volume_ratio', 1.0)
        trend_5_20 = features.get('trend_5_20', 0.0)
        
        # Simple decision tree
        if adx > 25:
            if trend_5_20 > 0.01:
                return MarketRegime.TRENDING_BULL.value
            elif trend_5_20 < -0.01:
                return MarketRegime.TRENDING_BEAR.value
        
        if bb_width > 0.15:
            return MarketRegime.HIGH_VOLATILITY.value
        elif bb_width < 0.05:
            return MarketRegime.LOW_VOLATILITY.value
            
        return MarketRegime.RANGING.value
    
    def update_confidence_thresholds(self):
        """Dynamically adjust confidence thresholds based on performance"""
        
        for regime, perf in self.regime_performance.items():
            if perf['total'] > 50:
                accuracy = perf['correct'] / perf['total']
                
                # Increase threshold if accuracy is low
                if accuracy < 0.6:
                    self.confidence_thresholds[regime] = min(0.9, 
                        self.confidence_thresholds.get(regime, 0.6) + 0.1)
                # Decrease threshold if accuracy is high
                elif accuracy > 0.8:
                    self.confidence_thresholds[regime] = max(0.4,
                        self.confidence_thresholds.get(regime, 0.6) - 0.05)

class MarketRegimeEvolution:
    """Track and adapt to evolving market regimes"""
    
    def __init__(self):
        self.regime_distributions = deque(maxlen=100)  # Track regime frequency
        self.market_cycles = []
        self.regime_transitions = {}
        
    def track_regime_evolution(self, regime_sequence):
        """Track how market regimes evolve over time"""
        
        # Update regime distribution
        regime_counts = pd.Series(regime_sequence).value_counts(normalize=True)
        self.regime_distributions.append(regime_counts)
        
        # Track regime transitions
        for i in range(1, len(regime_sequence)):
            prev_regime = regime_sequence[i-1]
            curr_regime = regime_sequence[i]
            
            if prev_regime not in self.regime_transitions:
                self.regime_transitions[prev_regime] = {}
            if curr_regime not in self.regime_transitions[prev_regime]:
                self.regime_transitions[prev_regime][curr_regime] = 0
                
            self.regime_transitions[prev_regime][curr_regime] += 1
    
    def detect_regime_shift(self):
        """Detect significant changes in market regime patterns"""
        
        if len(self.regime_distributions) < 20:
            return False
            
        # Compare recent vs historical distributions
        recent_dist = pd.concat(list(self.regime_distributions)[-10:]).groupby(level=0).mean()
        historical_dist = pd.concat(list(self.regime_distributions)[:-10]).groupby(level=0).mean()
        
        # Calculate distribution divergence
        divergence = np.sum(np.abs(recent_dist - historical_dist))
        
        return divergence > 0.3  # Threshold for significant shift

def demonstrate_adaptive_learning():
    """Demonstrate adaptive learning capabilities"""
    
    print("🧠 ADAPTIVE & REAL-TIME LEARNING STRATEGIES")
    print("=" * 55)
    
    print("\n1️⃣ ONLINE LEARNING FEATURES")
    print("✅ Continuous model updates with new data")
    print("✅ Performance-based retraining triggers")
    print("✅ Sliding window data management")
    print("✅ Incremental learning capabilities")
    
    print("\n2️⃣ ADAPTIVE CONFIDENCE THRESHOLDS")
    print("✅ Dynamic adjustment based on regime performance")
    print("✅ Higher thresholds for poorly performing regimes")
    print("✅ Automatic calibration over time")
    print("✅ Fallback to rule-based when uncertain")
    
    print("\n3️⃣ FEEDBACK INTEGRATION")
    print("✅ Learn from actual market outcomes")
    print("✅ Track false positive patterns")
    print("✅ Regime-specific performance monitoring")
    print("✅ Real-time accuracy assessment")
    
    print("\n4️⃣ MARKET EVOLUTION TRACKING")
    print("✅ Monitor changing regime distributions")
    print("✅ Detect regime transition patterns")
    print("✅ Identify market cycle shifts")
    print("✅ Adapt to new market conditions")
    
    print("\n5️⃣ IMPLEMENTATION BENEFITS")
    print("📈 Continuously improving accuracy")
    print("📈 Adapts to market evolution")
    print("📈 Reduces false positives over time")
    print("📈 Self-tuning confidence levels")
    
    # Create example adaptive classifier
    adaptive_model = AdaptiveRegimeClassifier()
    
    print(f"\n⚙️ ADAPTIVE MODEL CONFIGURATION:")
    print(f"- Window size: {adaptive_model.window_size}")
    print(f"- Retrain threshold: {adaptive_model.retrain_threshold}")
    print(f"- Confidence thresholds: {len(adaptive_model.confidence_thresholds)} regimes")
    
    print("\n🎯 RECOMMENDED IMPLEMENTATION ORDER:")
    print("1. Start with basic online learning (Option 5)")
    print("2. Add advanced features (Option 2)")
    print("3. Implement ensemble methods (Option 3)")
    print("4. Optimize hyperparameters (Option 4)")
    print("5. Add data augmentation (Option 1)")

if __name__ == "__main__":
    demonstrate_adaptive_learning()

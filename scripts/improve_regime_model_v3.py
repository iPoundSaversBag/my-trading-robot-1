#!/usr/bin/env python3
"""
Option 3: Advanced Model Architecture & Ensemble Methods
Uses multiple specialized models and ensemble techniques
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from core.enums import MarketRegime

class SpecializedRegimeClassifiers:
    """Create specialized classifiers for different regime types"""
    
    def __init__(self):
        self.models = {}
        self.confidence_thresholds = {}
        
    def create_specialized_models(self):
        """Create different models optimized for specific regime types"""
        
        # 1. Trend Specialist Model
        self.models['trend_specialist'] = {
            'model': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'target_regimes': [
                MarketRegime.TRENDING_BULL.value,
                MarketRegime.TRENDING_BEAR.value
            ],
            'features_focus': [
                'adx', 'plus_di', 'minus_di', 'trend_5_20', 'trend_20_50',
                'macd', 'macd_signal', 'sma_5', 'sma_20', 'sma_50'
            ]
        }
        
        # 2. Volatility Specialist Model
        self.models['volatility_specialist'] = {
            'model': RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                random_state=42
            ),
            'target_regimes': [
                MarketRegime.HIGH_VOLATILITY.value,
                MarketRegime.LOW_VOLATILITY.value
            ],
            'features_focus': [
                'bb_width', 'volatility_5d', 'volatility_20d', 'volatility_ratio',
                'atr', 'atr_ratio', 'price_change', 'price_change_5d'
            ]
        }
        
        # 3. Breakout Specialist Model (High Precision Focus)
        self.models['breakout_specialist'] = {
            'model': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'target_regimes': [
                MarketRegime.BREAKOUT_BULLISH.value,
                MarketRegime.BREAKOUT_BEARISH.value
            ],
            'features_focus': [
                'volume_ratio', 'bb_position', 'price_position', 
                'volatility_ratio', 'atr_ratio', 'rsi'
            ]
        }
        
        # 4. Accumulation/Distribution Specialist
        self.models['accumdist_specialist'] = {
            'model': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            ),
            'target_regimes': [
                MarketRegime.ACCUMULATION.value,
                MarketRegime.DISTRIBUTION.value
            ],
            'features_focus': [
                'obv', 'obv_sma', 'volume_ratio', 'rsi', 'bb_position',
                'price_position', 'adx'
            ]
        }
        
        return self.models

class HierarchicalRegimeClassifier:
    """Hierarchical classification approach"""
    
    def __init__(self):
        self.level1_classifier = None  # High-level regime types
        self.level2_classifiers = {}   # Specific regime classifiers
        
    def create_hierarchical_structure(self):
        """Create a hierarchical classification structure"""
        
        # Level 1: Broad categories
        self.regime_hierarchy = {
            'trending': [
                MarketRegime.TRENDING_BULL.value,
                MarketRegime.TRENDING_BEAR.value
            ],
            'volatile': [
                MarketRegime.HIGH_VOLATILITY.value,
                MarketRegime.LOW_VOLATILITY.value
            ],
            'breakout': [
                MarketRegime.BREAKOUT_BULLISH.value,
                MarketRegime.BREAKOUT_BEARISH.value
            ],
            'accumdist': [
                MarketRegime.ACCUMULATION.value,
                MarketRegime.DISTRIBUTION.value
            ],
            'ranging': [
                MarketRegime.RANGING.value
            ]
        }
        
        # Level 1 Classifier: Determine broad category
        self.level1_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Level 2 Classifiers: Specific regime within category
        for category in ['trending', 'volatile', 'breakout', 'accumdist']:
            self.level2_classifiers[category] = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )

class CascadingClassifier:
    """Cascading classifier with confidence-based decisions"""
    
    def __init__(self):
        self.cascade_models = []
        self.confidence_thresholds = [0.9, 0.8, 0.7, 0.6]
        
    def create_cascade(self):
        """Create cascade of models with increasing complexity"""
        
        # Stage 1: Simple, high-confidence classifier
        stage1 = {
            'model': RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42
            ),
            'confidence_threshold': 0.9,
            'features': ['adx', 'bb_width', 'volatility_ratio', 'trend_5_20']
        }
        
        # Stage 2: Medium complexity
        stage2 = {
            'model': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42
            ),
            'confidence_threshold': 0.8,
            'features': None  # Use all features
        }
        
        # Stage 3: High complexity
        stage3 = {
            'model': MLPClassifier(
                hidden_layer_sizes=(150, 100, 50),
                random_state=42
            ),
            'confidence_threshold': 0.7,
            'features': None  # Use all features
        }
        
        self.cascade_models = [stage1, stage2, stage3]

class EnsembleRegimeClassifier:
    """Advanced ensemble combining multiple approaches"""
    
    def __init__(self):
        self.base_models = []
        self.meta_model = None
        self.specialized_models = SpecializedRegimeClassifiers()
        
    def create_ensemble(self):
        """Create sophisticated ensemble model"""
        
        # Base models with different strengths
        base_models = [
            ('rf_balanced', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                class_weight='balanced',
                random_state=42
            )),
            ('gb_precision', GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=10,
                random_state=42
            )),
            ('rf_depth', RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                random_state=42
            ))
        ]
        
        # Voting ensemble
        voting_ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft'  # Use probabilities
        )
        
        return voting_ensemble

def demonstrate_ensemble_approaches():
    """Demonstrate different ensemble and architecture approaches"""
    
    print("🎯 Advanced Model Architecture Options")
    print("=" * 50)
    
    print("\n1️⃣ SPECIALIZED CLASSIFIERS")
    print("✅ Separate models for each regime type")
    print("✅ Optimized features per regime")
    print("✅ Higher precision for challenging regimes")
    print("✅ Ensemble of specialists")
    
    print("\n2️⃣ HIERARCHICAL CLASSIFICATION")
    print("✅ First classify broad category (trending/volatile/etc)")
    print("✅ Then classify specific regime within category")
    print("✅ Reduces confusion between similar regimes")
    print("✅ Better handling of class imbalance")
    
    print("\n3️⃣ CASCADING CLASSIFIERS")
    print("✅ High-confidence quick decisions first")
    print("✅ Complex models for uncertain cases")
    print("✅ Adaptive computational complexity")
    print("✅ Confidence-based routing")
    
    print("\n4️⃣ ENSEMBLE METHODS")
    print("✅ Voting ensemble of diverse models")
    print("✅ Stacking with meta-learner")
    print("✅ Bagging with specialized features")
    print("✅ Boosting for hard examples")
    
    print("\n🎯 EXPECTED IMPROVEMENTS:")
    print("- Breakout precision: 25% → 60%+")
    print("- Overall accuracy: 73% → 80%+")
    print("- Reduced false positives")
    print("- Better confidence estimates")
    
    # Create example implementations
    specialized = SpecializedRegimeClassifiers()
    models = specialized.create_specialized_models()
    
    print(f"\n📊 Specialized Models Created:")
    for name, config in models.items():
        print(f"- {name}: {len(config['features_focus'])} focused features")
    
    hierarchical = HierarchicalRegimeClassifier()
    hierarchical.create_hierarchical_structure()
    
    print(f"\n🌳 Hierarchical Structure:")
    for category, regimes in hierarchical.regime_hierarchy.items():
        print(f"- {category}: {len(regimes)} regime(s)")

if __name__ == "__main__":
    demonstrate_ensemble_approaches()

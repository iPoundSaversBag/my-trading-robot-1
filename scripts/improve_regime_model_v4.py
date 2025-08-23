#!/usr/bin/env python3
"""
Option 4: Advanced Hyperparameter Optimization & Cross-Validation
Systematic optimization of model parameters for better performance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, classification_report
import json
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

class RegimeClassifierOptimizer:
    """Advanced optimization for regime classification"""
    
    def __init__(self):
        self.best_params = {}
        self.optimization_results = {}
        
    def create_parameter_spaces(self):
        """Create comprehensive parameter spaces for optimization"""
        
        parameter_spaces = {
            'random_forest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 15, 20, 25, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
                'class_weight': ['balanced', 'balanced_subsample', None],
                'bootstrap': [True, False],
                'criterion': ['gini', 'entropy']
            },
            
            'gradient_boosting': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 8, 10, 12],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None]
            },
            
            'neural_network': {
                'hidden_layer_sizes': [
                    (50,), (100,), (150,), (200,),
                    (50, 25), (100, 50), (150, 75), (200, 100),
                    (100, 50, 25), (150, 100, 50), (200, 150, 100)
                ],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [200, 500, 1000]
            }
        }
        
        return parameter_spaces
    
    def create_custom_scoring(self):
        """Create custom scoring function that prioritizes breakout detection"""
        
        def custom_regime_score(y_true, y_pred):
            """Custom scoring that weights breakout detection heavily"""
            
            # Base F1 score
            base_f1 = f1_score(y_true, y_pred, average='weighted')
            
            # Breakout-specific scoring
            breakout_regimes = ['breakout_bullish', 'breakout_bearish']
            breakout_mask = np.isin(y_true, breakout_regimes)
            
            if np.any(breakout_mask):
                breakout_f1 = f1_score(
                    y_true[breakout_mask], 
                    y_pred[breakout_mask], 
                    average='weighted'
                )
                # Weight breakout performance more heavily
                final_score = 0.7 * base_f1 + 0.3 * breakout_f1
            else:
                final_score = base_f1
            
            return final_score
        
        return make_scorer(custom_regime_score)
    
    def optimize_with_bayesian(self):
        """Bayesian optimization approach (requires scikit-optimize)"""
        
        bayesian_space = {
            'n_estimators': (50, 500),
            'max_depth': (5, 30),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'max_features': (0.1, 1.0),
            'learning_rate': (0.01, 0.3)  # for GB
        }
        
        optimization_strategy = '''
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical
        
        # Define search space
        search_space = {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(5, 30),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Real(0.1, 1.0),
            'class_weight': Categorical(['balanced', 'balanced_subsample', None])
        }
        
        # Bayesian optimization
        bayes_search = BayesSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            search_spaces=search_space,
            n_iter=100,  # Number of iterations
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring=custom_scorer,
            n_jobs=-1,
            random_state=42
        )
        
        bayes_search.fit(X_train, y_train)
        '''
        
        return optimization_strategy
    
    def create_regime_specific_optimization(self):
        """Optimize parameters separately for each regime type"""
        
        regime_specific_params = {
            'trending_regimes': {
                'focus_features': ['adx', 'plus_di', 'minus_di', 'trend_5_20'],
                'model_params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [8, 12, 16],
                    'min_samples_split': [2, 5],
                    'criterion': ['gini', 'entropy']
                }
            },
            
            'volatility_regimes': {
                'focus_features': ['bb_width', 'volatility_ratio', 'atr_ratio'],
                'model_params': {
                    'n_estimators': [150, 250, 350],
                    'max_depth': [10, 15, 20],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            
            'breakout_regimes': {
                'focus_features': ['volume_ratio', 'bb_position', 'price_position'],
                'model_params': {
                    'n_estimators': [200, 300, 400],
                    'max_depth': [6, 10, 15],
                    'class_weight': ['balanced', 'balanced_subsample'],
                    'min_samples_split': [5, 10, 15]
                }
            }
        }
        
        return regime_specific_params
    
    def cross_validation_strategy(self):
        """Advanced cross-validation strategies"""
        
        cv_strategies = {
            'time_series_split': '''
            from sklearn.model_selection import TimeSeriesSplit
            
            # Respect temporal order in data
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring=custom_scorer)
            ''',
            
            'stratified_group': '''
            from sklearn.model_selection import StratifiedGroupKFold
            
            # Stratify by regime while grouping by time periods
            sgkf = StratifiedGroupKFold(n_splits=5)
            scores = cross_val_score(model, X, y, groups=time_groups, cv=sgkf)
            ''',
            
            'regime_balanced': '''
            # Custom CV that ensures each fold has balanced regimes
            def regime_balanced_cv(X, y, n_splits=5):
                from collections import Counter
                
                folds = []
                for regime in np.unique(y):
                    regime_indices = np.where(y == regime)[0]
                    regime_folds = np.array_split(regime_indices, n_splits)
                    
                    for i, fold_indices in enumerate(regime_folds):
                        if i >= len(folds):
                            folds.append([])
                        folds[i].extend(fold_indices)
                
                return folds
            '''
        }
        
        return cv_strategies

def demonstrate_optimization_strategies():
    """Demonstrate comprehensive optimization approaches"""
    
    print("⚡ HYPERPARAMETER OPTIMIZATION STRATEGIES")
    print("=" * 55)
    
    optimizer = RegimeClassifierOptimizer()
    
    print("\n1️⃣ GRID SEARCH OPTIMIZATION")
    param_spaces = optimizer.create_parameter_spaces()
    print(f"✅ Parameter combinations: {len(param_spaces['random_forest'])} dimensions")
    print("✅ Exhaustive search of parameter space")
    print("✅ Guaranteed to find optimal combination")
    print("❌ Computationally expensive")
    
    print("\n2️⃣ RANDOMIZED SEARCH")
    print("✅ Faster than grid search")
    print("✅ Good exploration of parameter space")
    print("✅ Can find good solutions quickly")
    print("✅ Configurable number of iterations")
    
    print("\n3️⃣ BAYESIAN OPTIMIZATION")
    print("✅ Intelligent parameter exploration")
    print("✅ Uses past results to guide search")
    print("✅ Efficient for expensive evaluations")
    print("✅ Better than random search")
    
    print("\n4️⃣ CUSTOM SCORING FUNCTIONS")
    custom_scorer = optimizer.create_custom_scoring()
    print("✅ Prioritizes breakout detection")
    print("✅ Weights regime-specific performance")
    print("✅ Balances precision and recall")
    print("✅ Domain-specific optimization")
    
    print("\n5️⃣ REGIME-SPECIFIC OPTIMIZATION")
    regime_params = optimizer.create_regime_specific_optimization()
    print(f"✅ Optimized parameters for {len(regime_params)} regime types")
    print("✅ Feature selection per regime")
    print("✅ Specialized model configurations")
    
    print("\n6️⃣ ADVANCED CROSS-VALIDATION")
    cv_strategies = optimizer.cross_validation_strategy()
    print(f"✅ {len(cv_strategies)} validation strategies")
    print("✅ Time-aware validation")
    print("✅ Regime-balanced folds")
    print("✅ Robust performance estimation")
    
    print("\n🎯 EXPECTED IMPROVEMENTS:")
    print("- Fine-tuned model parameters")
    print("- Better generalization performance")
    print("- Reduced overfitting")
    print("- Optimized for trading metrics")
    
    print("\n⚙️ IMPLEMENTATION PRIORITY:")
    print("1. Start with RandomizedSearchCV (quick wins)")
    print("2. Implement custom scoring for breakouts")
    print("3. Use TimeSeriesSplit for validation")
    print("4. Consider Bayesian optimization for final tuning")

if __name__ == "__main__":
    demonstrate_optimization_strategies()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Models Module for Trading System
Enhanced machine learning capabilities for the backtesting engine
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging

class RegimeClassifier:
    """Enhanced regime classification for market conditions"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.regime_labels = {
            0: 'trending_up',
            1: 'trending_down', 
            2: 'sideways',
            3: 'high_volatility',
            4: 'low_volatility'
        }
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime classification"""
        features = df.copy()
        
        # Price-based features
        features['price_change'] = df['close'].pct_change()
        features['price_momentum'] = df['close'].rolling(10).mean() / df['close'].rolling(20).mean() - 1
        features['volatility'] = df['close'].rolling(20).std()
        features['rsi'] = self._calculate_rsi(df['close'])
        
        # Volume features if available
        if 'volume' in df.columns:
            features['volume_ma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_ma']
        
        # Trend features
        features['sma_short'] = df['close'].rolling(10).mean()
        features['sma_long'] = df['close'].rolling(30).mean()
        features['trend_ratio'] = features['sma_short'] / features['sma_long'] - 1
        
        return features.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def classify_regime(self, features: pd.DataFrame) -> np.ndarray:
        """Classify market regime based on features"""
        if self.model is None:
            return np.zeros(len(features))
            
        scaled_features = self.scaler.transform(features[self.feature_columns])
        predictions = self.model.predict(scaled_features)
        return predictions
    
    def train(self, df: pd.DataFrame, labels: np.ndarray = None):
        """Train the regime classifier"""
        features = self.prepare_features(df)
        
        if labels is None:
            # Auto-generate labels based on market conditions
            labels = self._generate_regime_labels(features)
        
        # Select features for training
        feature_cols = ['price_change', 'price_momentum', 'volatility', 'rsi', 'trend_ratio']
        if 'volume_ratio' in features.columns:
            feature_cols.append('volume_ratio')
            
        X = features[feature_cols].dropna()
        y = labels[:len(X)]
        
        self.feature_columns = feature_cols
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        return self.model.score(X_scaled, y)
    
    def _generate_regime_labels(self, features: pd.DataFrame) -> np.ndarray:
        """Generate regime labels based on market conditions"""
        labels = np.zeros(len(features))
        
        for i in range(len(features)):
            momentum = features['price_momentum'].iloc[i]
            volatility = features['volatility'].iloc[i]
            trend = features['trend_ratio'].iloc[i]
            
            # High volatility regime
            if volatility > features['volatility'].quantile(0.8):
                labels[i] = 3  # high_volatility
            # Low volatility regime  
            elif volatility < features['volatility'].quantile(0.2):
                labels[i] = 4  # low_volatility
            # Trending up
            elif trend > 0.02 and momentum > 0:
                labels[i] = 0  # trending_up
            # Trending down
            elif trend < -0.02 and momentum < 0:
                labels[i] = 1  # trending_down
            # Sideways
            else:
                labels[i] = 2  # sideways
                
        return labels

class EnsemblePredictor:
    """Ensemble model for price prediction and signal generation"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.models = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ensemble prediction"""
        features = df.copy()
        
        # Technical indicators
        features['sma_5'] = df['close'].rolling(5).mean()
        features['sma_10'] = df['close'].rolling(10).mean()
        features['sma_20'] = df['close'].rolling(20).mean()
        features['ema_12'] = df['close'].ewm(span=12).mean()
        features['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Price patterns
        features['price_change'] = df['close'].pct_change()
        features['high_low_ratio'] = df['high'] / df['low'] - 1
        features['close_open_ratio'] = df['close'] / df['open'] - 1
        
        # Volatility features
        features['volatility'] = df['close'].rolling(20).std()
        features['atr'] = self._calculate_atr(df)
        
        # Momentum indicators
        features['rsi'] = self._calculate_rsi(df['close'])
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        return features.dropna()
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def train(self, df: pd.DataFrame, target_column: str = 'close'):
        """Train ensemble models"""
        features = self.prepare_features(df)
        
        # Prepare target (next period returns)
        target = df[target_column].pct_change().shift(-1).dropna()
        
        # Align features and target
        min_len = min(len(features), len(target))
        X = features.iloc[:min_len]
        y = target.iloc[:min_len]
        
        # Select feature columns
        feature_cols = [col for col in X.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        X = X[feature_cols].dropna()
        y = y.iloc[:len(X)]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble models
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100, 
            random_state=42,
            n_jobs=-1
        )
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        )
        
        # Fit models
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(feature_cols, model.feature_importances_))
        
        # Evaluate ensemble
        train_score = self._evaluate_ensemble(X_train_scaled, y_train)
        test_score = self._evaluate_ensemble(X_test_scaled, y_test)
        
        self.is_trained = True
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': self.feature_importance
        }
    
    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate ensemble performance"""
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        return {
            'mse': mse,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            return np.zeros(len(X))
        
        predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def save_models(self, directory: str):
        """Save trained models"""
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(directory, f'{name}_model.joblib')
            joblib.dump(model, filepath)
        
        # Save scaler
        scaler_path = os.path.join(directory, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained,
            'model_names': list(self.models.keys())
        }
        metadata_path = os.path.join(directory, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def load_models(self, directory: str):
        """Load pre-trained models"""
        # Load metadata
        metadata_path = os.path.join(directory, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_importance = metadata.get('feature_importance', {})
            self.is_trained = metadata.get('is_trained', False)
            model_names = metadata.get('model_names', [])
            
            # Load models
            for name in model_names:
                filepath = os.path.join(directory, f'{name}_model.joblib')
                if os.path.exists(filepath):
                    self.models[name] = joblib.load(filepath)
            
            # Load scaler
            scaler_path = os.path.join(directory, 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

class MLModelManager:
    """Manager for all ML models used in the trading system"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or 'core/optimization_config.json'
        self.config = self._load_config()
        self.regime_classifier = RegimeClassifier(self.config.get('machine_learning', {}))
        self.ensemble_predictor = EnsemblePredictor(self.config.get('machine_learning', {}))
        self.models_directory = 'ml_models'
        
    def _load_config(self) -> Dict:
        """Load configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def train_all_models(self, data: pd.DataFrame) -> Dict:
        """Train all ML models"""
        results = {}
        
        if self.config.get('machine_learning', {}).get('enabled', False):
            # Train regime classifier
            regime_score = self.regime_classifier.train(data)
            results['regime_classifier'] = regime_score
            
            # Train ensemble predictor
            ensemble_results = self.ensemble_predictor.train(data)
            results['ensemble_predictor'] = ensemble_results
            
            # Save models
            self.save_all_models()
            
        return results
    
    def save_all_models(self):
        """Save all trained models"""
        os.makedirs(self.models_directory, exist_ok=True)
        
        # Save regime classifier
        regime_path = os.path.join(self.models_directory, 'regime_classifier.joblib')
        joblib.dump(self.regime_classifier, regime_path)
        
        # Save ensemble predictor
        self.ensemble_predictor.save_models(self.models_directory)
        
    def load_all_models(self):
        """Load all pre-trained models"""
        # Load regime classifier
        regime_path = os.path.join(self.models_directory, 'regime_classifier.joblib')
        if os.path.exists(regime_path):
            self.regime_classifier = joblib.load(regime_path)
        
        # Load ensemble predictor
        self.ensemble_predictor.load_models(self.models_directory)
    
    def get_regime_prediction(self, data: pd.DataFrame) -> str:
        """Get current market regime prediction"""
        features = self.regime_classifier.prepare_features(data)
        if len(features) > 0:
            regime_id = self.regime_classifier.classify_regime(features.tail(1))
            return self.regime_classifier.regime_labels.get(regime_id[0], 'unknown')
        return 'unknown'
    
    def get_price_prediction(self, data: pd.DataFrame) -> float:
        """Get price movement prediction"""
        features = self.ensemble_predictor.prepare_features(data)
        if len(features) > 0 and self.ensemble_predictor.is_trained:
            feature_cols = [col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            X = features[feature_cols].tail(1).values
            X_scaled = self.ensemble_predictor.scaler.transform(X)
            prediction = self.ensemble_predictor.predict(X_scaled)
            return prediction[0] if len(prediction) > 0 else 0.0
        return 0.0

# Initialize the ML model manager
ml_manager = MLModelManager()

# Export main classes for use in backtest
__all__ = ['RegimeClassifier', 'EnsemblePredictor', 'MLModelManager', 'ml_manager']

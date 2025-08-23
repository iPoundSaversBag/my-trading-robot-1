#!/usr/bin/env python3
"""
Implement Best ML Improvements for Regime Classification
=========================================================

This script implements the most effective improvements based on testing:
1. Advanced Feature Engineering (v2) - Primary approach
2. Class Balancing with SMOTE (v1) - For imbalanced data
3. Hyperparameter optimization (v4) - For fine-tuning
4. Basic ensemble methods (v3) - For robustness

Goal: Improve breakout detection from 25% to 60%+ precision
"""

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
try:  # Guarded import to avoid hard failure in minimal environments
    from imblearn.over_sampling import SMOTE  # type: ignore  # noqa: F401
    from imblearn.combine import SMOTEENN     # type: ignore  # noqa: F401
except ImportError:  # Fallback placeholders; real functionality requires imbalanced-learn
    SMOTE = object  # type: ignore
    SMOTEENN = object  # type: ignore

# Technical analysis
import ta
from core.enums import MarketRegime

class AdvancedRegimeFeatureEngineer:
    """Advanced feature engineering specifically for regime classification"""
    
    def __init__(self):
        self.feature_names = []
        
    def engineer_advanced_features(self, df):
        """Create advanced features optimized for regime detection"""
        print("🔧 Engineering advanced features...")
        
        # Original features (keep existing ones)
        original_features = self._get_original_features(df)
        
        # NEW: Volume-Price Divergence Analysis
        vp_features = self._volume_price_divergence(df)
        
        # NEW: Multi-timeframe Momentum 
        momentum_features = self._multiframe_momentum(df)
        
        # NEW: Volatility Regime Detection
        volatility_features = self._volatility_regime_features(df)
        
        # NEW: Support/Resistance Breakout Strength
        breakout_features = self._breakout_strength_features(df)
        
        # NEW: Order Flow Proxy Indicators
        orderflow_features = self._order_flow_proxy(df)
        
        # Combine all features
        all_features = pd.concat([
            original_features,
            vp_features,
            momentum_features, 
            volatility_features,
            breakout_features,
            orderflow_features
        ], axis=1)
        
        print(f"✅ Created {len(all_features.columns)} features (was {len(original_features.columns)})")
        return all_features
    
    def _get_original_features(self, df):
        """Get the original 31 features used in base model"""
        features = pd.DataFrame(index=df.index)

        required = {'close', 'high', 'low', 'volume'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for feature engineering: {missing}")

        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['price_sma_ratio'] = df['close'] / df['close'].rolling(20).mean()
        features['high_low_ratio'] = df['high'] / df['low']

        # Technical indicators
        try:
            features['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            features['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
            features['bb_width'] = ta.volatility.BollingerBands(df['close']).bollinger_wband()
            features['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        except Exception:
            for col in ['rsi', 'adx', 'bb_width', 'atr']:
                features[col] = np.nan

        # Volume features
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        try:
            features['price_volume_trend'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
        except Exception:
            features['price_volume_trend'] = np.nan

        # Additional indicators (subset of original 31 for brevity; extend as needed)
        try:
            features['macd'] = ta.trend.MACD(df['close']).macd()
            features['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            features['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        except Exception:
            for col in ['macd', 'cci', 'stoch']:
                features[col] = np.nan

        features = features.bfill().ffill().replace([np.inf, -np.inf], 0)
        return features
    
    def _volume_price_divergence(self, df):
        """Volume-Price divergence analysis for better breakout detection"""
        features = pd.DataFrame(index=df.index)
        
        # Price momentum vs volume momentum
        price_mom = df['close'].pct_change(5)
        volume_mom = df['volume'].pct_change(5)
        features['vp_divergence'] = price_mom - volume_mom
        
        # Simple VWAP calculation
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        features['vwap_deviation'] = (df['close'] - vwap) / df['close']
        
        # Accumulation/Distribution divergence
        ad_line = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        features['ad_price_divergence'] = ad_line.pct_change(10) - df['close'].pct_change(10)
        
        # On-Balance Volume momentum
        obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        features['obv_momentum'] = obv.pct_change(5)
        
        return features.fillna(0)
    
    def _multiframe_momentum(self, df):
        """Multi-timeframe momentum convergence"""
        features = pd.DataFrame(index=df.index)
        
        # Short, medium, long-term momentum alignment
        mom_5 = df['close'].pct_change(5)
        mom_20 = df['close'].pct_change(20)
        mom_50 = df['close'].pct_change(50)
        
        features['momentum_alignment'] = np.sign(mom_5) + np.sign(mom_20) + np.sign(mom_50)
        features['momentum_strength'] = abs(mom_5) + abs(mom_20) + abs(mom_50)
        
        # Rate of change acceleration
        features['roc_acceleration'] = mom_5 - mom_20
        
        return features.fillna(0)
    
    def _volatility_regime_features(self, df):
        """Enhanced volatility regime detection"""
        features = pd.DataFrame(index=df.index)
        
        # Rolling volatility with different windows
        vol_5 = df['close'].rolling(5).std()
        vol_20 = df['close'].rolling(20).std()
        vol_50 = df['close'].rolling(50).std()
        
        features['vol_regime'] = vol_5 / vol_20
        features['vol_persistence'] = vol_20 / vol_50
        
        # Volatility clustering score
        features['vol_clustering'] = vol_5.rolling(10).std()
        
        # High-low volatility vs close volatility
        hl_vol = (df['high'] - df['low']) / df['close']
        close_vol = df['close'].pct_change().abs()
        features['intraday_vs_daily_vol'] = hl_vol / close_vol.replace(0, np.nan)
        
        return features.fillna(0)
    
    def _breakout_strength_features(self, df):
        """Support/Resistance breakout strength indicators"""
        features = pd.DataFrame(index=df.index)
        
        # 20-period high/low breakouts
        period_high = df['high'].rolling(20).max()
        period_low = df['low'].rolling(20).min()
        
        features['breakout_strength_up'] = (df['close'] - period_high) / period_high
        features['breakout_strength_down'] = (period_low - df['close']) / period_low
        
        # Volume confirmation for breakouts
        avg_volume = df['volume'].rolling(20).mean()
        features['breakout_volume_confirm'] = df['volume'] / avg_volume
        
        # Price position within recent range
        features['range_position'] = (df['close'] - period_low) / (period_high - period_low)
        
        return features.fillna(0)
    
    def _order_flow_proxy(self, df):
        """Order flow proxy indicators"""
        features = pd.DataFrame(index=df.index)
        
        # Buy/sell pressure proxy
        features['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        features['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        
        # Volume-weighted buy/sell pressure
        features['vw_buy_pressure'] = features['buy_pressure'] * df['volume']
        features['vw_sell_pressure'] = features['sell_pressure'] * df['volume']
        
        return features.fillna(0.5)  # Neutral pressure

class EnhancedRegimeClassifier:
    """Enhanced regime classifier with advanced ML techniques"""
    
    def __init__(self):
        self.feature_engineer = AdvancedRegimeFeatureEngineer()
        self.scaler = StandardScaler()
        self.base_model = None
        self.ensemble_model = None
        self.feature_names = []
        
    def train_enhanced_model(self, data_files):
        """Train enhanced model with all improvements"""
        print("🚀 Training Enhanced Regime Classifier")
        print("=" * 50)
        
        # 1. Load and prepare data
        print("📊 Loading training data...")
        X, y = self._load_and_prepare_data(data_files)
        
        # 2. Apply SMOTE for class balancing
        print("⚖️ Balancing classes with SMOTE...")
        X_balanced, y_balanced = self._balance_classes(X, y)
        
        # 3. Feature scaling
        print("📏 Scaling features...")
        X_scaled = self.scaler.fit_transform(X_balanced)
        
        # 4. Hyperparameter optimization
        print("🎯 Optimizing hyperparameters...")
        best_model = self._optimize_hyperparameters(X_scaled, y_balanced)
        
        # 5. Train ensemble model
        print("🎭 Training ensemble model...")
        ensemble_model = self._train_ensemble(X_scaled, y_balanced)
        
        # 6. Evaluate models
        print("📈 Evaluating models...")
        self._evaluate_models(X_scaled, y_balanced, best_model, ensemble_model)
        
        # 7. Save best model
        self.base_model = best_model
        self.ensemble_model = ensemble_model
        self._save_models()
        
        print("✅ Enhanced model training complete!")
        
    def _load_and_prepare_data(self, data_files):
        """Load data and engineer advanced features"""
        all_features = []
        all_labels = []
        
        for file_path in data_files:
            if Path(file_path).exists():
                print(f"  📁 Loading {file_path}")
                
                # Load parquet or CSV files
                if file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                # Ensure required columns exist
                required_cols = ['close', 'high', 'low', 'open', 'volume']
                if not all(col in df.columns for col in required_cols):
                    print(f"  ⚠️ Skipping {file_path} - missing required columns")
                    continue
                
                # Take a sample for faster processing during testing
                if len(df) > 10000:
                    df = df.sample(n=10000, random_state=42).sort_index()
                    print(f"  📊 Using sample of 10,000 rows for faster processing")
                
                # Engineer advanced features
                features = self.feature_engineer.engineer_advanced_features(df)
                
                # Generate labels using original regime detection logic
                labels = self._generate_regime_labels(df)
                
                # Ensure same length
                min_len = min(len(features), len(labels))
                features = features.iloc[:min_len]
                labels = labels[:min_len]
                
                # Drop rows with NaN values
                mask = ~features.isnull().any(axis=1)
                features = features[mask]
                labels = np.array(labels)[mask]
                
                all_features.append(features)
                all_labels.extend(labels)
        
        if not all_features:
            raise ValueError("No valid data found to train on")
        
        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = np.array(all_labels)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"  ✅ Total samples: {len(X)}, Features: {len(X.columns)}")
        print(f"  📊 Regime distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for regime, count in zip(unique, counts):
            try:
                regime_name = MarketRegime(regime).name
            except:
                regime_name = f"Unknown_{regime}"
            print(f"    {regime_name}: {count} ({count/len(y)*100:.1f}%)")
            
        return X, y
    
    def _generate_regime_labels(self, df):
        """Generate regime labels using simplified rule-based approach"""
        regimes = []
        
        # Calculate required indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        df['bb_width'] = ta.volatility.BollingerBands(df['close']).bollinger_wband()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        for i in range(len(df)):
            if i < 50:  # Skip first 50 rows for indicator calculation
                regimes.append(MarketRegime.RANGING.value)
                continue
            
            # Get current values
            rsi = df['rsi'].iloc[i]
            adx = df['adx'].iloc[i]
            bb_width = df['bb_width'].iloc[i]
            atr = df['atr'].iloc[i]
            volume_ratio = df['volume'].iloc[i] / df['volume_ma'].iloc[i]
            
            # Price momentum
            price_change_5 = (df['close'].iloc[i] - df['close'].iloc[i-5]) / df['close'].iloc[i-5]
            price_change_20 = (df['close'].iloc[i] - df['close'].iloc[i-20]) / df['close'].iloc[i-20]
            
            # Volatility measures
            recent_vol = df['close'].iloc[i-20:i].std()
            historical_vol = df['close'].iloc[i-100:i-20].std()
            
            # Determine regime using rule-based approach
            if pd.isna(rsi) or pd.isna(adx):
                regime = MarketRegime.RANGING
            elif adx > 25 and abs(price_change_20) > 0.1:
                # Strong trend
                if price_change_20 > 0:
                    regime = MarketRegime.TRENDING_BULL
                else:
                    regime = MarketRegime.TRENDING_BEAR
            elif recent_vol > historical_vol * 2:
                regime = MarketRegime.HIGH_VOLATILITY
            elif recent_vol < historical_vol * 0.5:
                regime = MarketRegime.LOW_VOLATILITY
            elif volume_ratio > 2 and abs(price_change_5) > 0.03:
                # Potential breakout
                if price_change_5 > 0:
                    regime = MarketRegime.BREAKOUT_BULLISH
                else:
                    regime = MarketRegime.BREAKOUT_BEARISH
            elif rsi > 70:
                regime = MarketRegime.DISTRIBUTION
            elif rsi < 30:
                regime = MarketRegime.ACCUMULATION
            else:
                regime = MarketRegime.RANGING
                
            regimes.append(regime.value)
            
        return regimes
    
    def _balance_classes(self, X, y):
        """Balance classes using SMOTE with parameters optimized for our data"""
        # Use SMOTEENN for combined over/under sampling
        smote_enn = SMOTEENN(
            smote=SMOTE(random_state=42, k_neighbors=3),
            random_state=42
        )
        
        print("  📊 Before balancing:")
        unique, counts = np.unique(y, return_counts=True)
        for regime, count in zip(unique, counts):
            print(f"    {MarketRegime(regime).name}: {count}")
            
        X_balanced, y_balanced = smote_enn.fit_resample(X, y)
        
        print("  📊 After balancing:")
        unique, counts = np.unique(y_balanced, return_counts=True)
        for regime, count in zip(unique, counts):
            print(f"    {MarketRegime(regime).name}: {count}")
            
        return X_balanced, y_balanced
    
    def _optimize_hyperparameters(self, X, y):
        """Optimize hyperparameters using RandomizedSearchCV"""
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        # Use TimeSeriesSplit for time-aware validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        search = RandomizedSearchCV(
            rf, param_dist, n_iter=20, cv=tscv, 
            scoring='f1_macro', random_state=42, n_jobs=-1
        )
        
        search.fit(X, y)
        
        print(f"  🎯 Best parameters: {search.best_params_}")
        print(f"  📈 Best CV score: {search.best_score_:.3f}")
        
        return search.best_estimator_
    
    def _train_ensemble(self, X, y):
        """Train ensemble of diverse models"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        # Individual models with different strengths
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        svm = SVC(probability=True, random_state=42)
        
        # Voting ensemble
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('lr', lr), ('svm', svm)],
            voting='soft'
        )
        
        ensemble.fit(X, y)
        return ensemble
    
    def _evaluate_models(self, X, y, base_model, ensemble_model):
        """Evaluate both models"""
        from sklearn.model_selection import cross_val_score
        
        print("  📊 Base Model Performance:")
        base_scores = cross_val_score(base_model, X, y, cv=3, scoring='f1_macro')
        print(f"    F1-macro: {base_scores.mean():.3f} (+/- {base_scores.std() * 2:.3f})")
        
        print("  📊 Ensemble Model Performance:")
        ensemble_scores = cross_val_score(ensemble_model, X, y, cv=3, scoring='f1_macro')
        print(f"    F1-macro: {ensemble_scores.mean():.3f} (+/- {ensemble_scores.std() * 2:.3f})")
        
        # Feature importance
        if hasattr(base_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': base_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("  🔝 Top 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
    
    def _save_models(self):
        """Save trained models and metadata"""
        models_dir = Path("ml_models")
        models_dir.mkdir(exist_ok=True)
        
        # Save models
        joblib.dump(self.base_model, models_dir / "enhanced_regime_classifier.pkl")
        joblib.dump(self.ensemble_model, models_dir / "ensemble_regime_classifier.pkl")
        joblib.dump(self.scaler, models_dir / "feature_scaler.pkl")
        
        # Save metadata
        metadata = {
            "model_type": "enhanced_regime_classifier",
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "improvements": [
                "Advanced feature engineering",
                "SMOTE class balancing", 
                "Hyperparameter optimization",
                "Ensemble methods"
            ],
            "target_regimes": [regime.name for regime in MarketRegime]
        }
        
        with open(models_dir / "enhanced_model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  💾 Models saved to {models_dir}/")

def main():
    """Main execution function"""
    # Data files to use for training (updated to use actual files)
    data_files = [
        "data/crypto_data_1h.parquet",
        "data/crypto_data_4h.parquet", 
        "data/crypto_data_15m.parquet"
    ]
    
    # Check which files exist
    existing_files = [f for f in data_files if Path(f).exists()]
    
    if not existing_files:
        print("❌ No data files found. Please ensure data files exist in the data/ directory.")
        print("📂 Available files:")
        data_dir = Path("data")
        if data_dir.exists():
            for file in data_dir.iterdir():
                if file.is_file() and file.suffix in ['.csv', '.parquet']:
                    print(f"   {file}")
        return
        
    print(f"🎯 Training enhanced regime classifier on {len(existing_files)} datasets")
    
    # Train enhanced model
    classifier = EnhancedRegimeClassifier()
    classifier.train_enhanced_model(existing_files)
    
    print("\n🎉 Enhanced ML training complete!")
    print("🔄 The new model should show significant improvements in:")
    print("   • Breakout detection precision (target: 60%+)")
    print("   • Overall classification accuracy (target: 80%+)")
    print("   • Reduced false positives")
    print("   • Better confidence estimates")

if __name__ == "__main__":
    main()

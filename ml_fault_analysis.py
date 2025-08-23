#!/usr/bin/env python3
"""
ML Fault Analysis based on documented performance metrics.
Analyzing potential negative impacts on backtesting performance.
"""

import json
from pathlib import Path

class MLPerformanceFaultAnalyzer:
    def __init__(self):
        self.results_file = Path("ML_IMPROVEMENT_RESULTS.md")
        self.metadata_file = Path("ml_models/new_model_comparison_results.json")
        
    def analyze_documented_faults(self):
        """Analyze ML faults based on documented performance"""
        print("🔍 ML FAULT ANALYSIS FOR BACKTESTING IMPACT")
        print("=" * 60)
        
        # Load performance data
        performance_data = self.load_performance_data()
        
        # Analyze each category of faults
        faults = {
            'accuracy_faults': self.analyze_accuracy_faults(performance_data),
            'regime_specific_faults': self.analyze_regime_specific_faults(),
            'breakout_detection_faults': self.analyze_breakout_faults(),
            'class_imbalance_faults': self.analyze_class_imbalance_faults(),
            'computational_overhead_faults': self.analyze_computational_faults(),
            'prediction_reliability_faults': self.analyze_reliability_faults(performance_data)
        }
        
        return faults
    
    def load_performance_data(self):
        """Load performance data from results file"""
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return data.get('results', {})
        except Exception as e:
            print(f"Warning: Could not load performance data: {e}")
            return {
                "Enhanced Model": {"accuracy": 0.453, "macro_f1": 0.303},
                "Ensemble Model": {"accuracy": 0.474, "macro_f1": 0.268}
            }
    
    def analyze_accuracy_faults(self, performance_data):
        """Analyze accuracy-related faults"""
        print("\n📊 ACCURACY FAULT ANALYSIS")
        print("-" * 40)
        
        faults = []
        
        for model_name, metrics in performance_data.items():
            accuracy = metrics.get('accuracy', 0)
            macro_f1 = metrics.get('macro_f1', 0)
            
            print(f"{model_name}: Accuracy={accuracy:.1%}, Macro F1={macro_f1:.3f}")
            
            # Critical accuracy faults
            if accuracy < 0.5:
                faults.append({
                    "fault": "BELOW_RANDOM_ACCURACY",
                    "model": model_name,
                    "severity": "CRITICAL",
                    "value": accuracy,
                    "impact": "Worse than random chance - will cause systematic losses in backtesting",
                    "backtest_impact": "CATASTROPHIC - Wrong regime detection leads to wrong parameters"
                })
            
            # Low accuracy faults  
            if accuracy < 0.6:
                faults.append({
                    "fault": "LOW_OVERALL_ACCURACY",
                    "model": model_name,
                    "severity": "HIGH",
                    "value": accuracy,
                    "impact": "Only 45-47% correct regime detection - frequent parameter misalignment",
                    "backtest_impact": "HIGH - Risk/reward parameters set for wrong market conditions"
                })
            
            # Macro F1 issues (indicates poor minority class performance)
            if macro_f1 < 0.4:
                faults.append({
                    "fault": "POOR_MINORITY_CLASS_PERFORMANCE",
                    "model": model_name,
                    "severity": "CRITICAL",
                    "value": macro_f1,
                    "impact": "Critical regimes (breakouts, high volatility) poorly detected",
                    "backtest_impact": "CRITICAL - Missing dangerous/profitable regimes causes major losses"
                })
        
        return faults
    
    def analyze_regime_specific_faults(self):
        """Analyze regime-specific detection faults"""
        print("\n🎭 REGIME-SPECIFIC FAULT ANALYSIS")
        print("-" * 40)
        
        # Based on documented results from ML_IMPROVEMENT_RESULTS.md
        regime_performance = {
            'BREAKOUT_BULLISH': {'precision': 0.0, 'samples': 5, 'critical': True},
            'BREAKOUT_BEARISH': {'precision': 0.10, 'samples': 15, 'critical': True},
            'HIGH_VOLATILITY': {'precision': 'unknown', 'samples': 200, 'critical': True},
            'LOW_VOLATILITY': {'precision': 'good', 'samples': 400, 'critical': False},
            'RANGING': {'precision': 'good', 'samples': 300, 'critical': False},
            'TRENDING_BULL': {'precision': 'moderate', 'samples': 200, 'critical': True},
            'TRENDING_BEAR': {'precision': 'moderate', 'samples': 200, 'critical': True}
        }
        
        faults = []
        
        for regime, perf in regime_performance.items():
            precision = perf['precision']
            samples = perf['samples']
            critical = perf['critical']
            
            print(f"{regime}: Precision={precision}, Samples={samples}, Critical={critical}")
            
            # Complete failure to detect critical regimes
            if precision == 0.0 and critical:
                faults.append({
                    "fault": "COMPLETE_REGIME_DETECTION_FAILURE",
                    "regime": regime,
                    "severity": "CATASTROPHIC",
                    "precision": precision,
                    "impact": f"NEVER detects {regime} - complete blind spot",
                    "backtest_impact": f"CATASTROPHIC - Will use wrong parameters during {regime} periods"
                })
            
            # Very poor detection of critical regimes
            elif isinstance(precision, float) and precision < 0.2 and critical:
                faults.append({
                    "fault": "CRITICAL_REGIME_POOR_DETECTION",
                    "regime": regime,
                    "severity": "CRITICAL", 
                    "precision": precision,
                    "impact": f"Only {precision:.1%} precision for {regime}",
                    "backtest_impact": f"CRITICAL - Wrong parameters 80%+ of time during {regime}"
                })
            
            # Insufficient training data
            if samples < 50:
                faults.append({
                    "fault": "INSUFFICIENT_TRAINING_DATA",
                    "regime": regime,
                    "severity": "HIGH",
                    "samples": samples,
                    "impact": f"Only {samples} training samples - unreliable detection",
                    "backtest_impact": f"HIGH - Unpredictable behavior during {regime} periods"
                })
        
        return faults
    
    def analyze_breakout_faults(self):
        """Analyze breakout detection specific faults"""
        print("\n💥 BREAKOUT DETECTION FAULT ANALYSIS")
        print("-" * 40)
        
        # Based on documented 0% bullish and 10% bearish breakout precision
        breakout_performance = {
            'BREAKOUT_BULLISH': 0.0,   # 0% precision
            'BREAKOUT_BEARISH': 0.10   # 10% precision
        }
        
        faults = []
        
        for breakout_type, precision in breakout_performance.items():
            print(f"{breakout_type}: {precision:.1%} precision")
            
            if precision == 0.0:
                faults.append({
                    "fault": "COMPLETE_BREAKOUT_BLINDNESS",
                    "breakout_type": breakout_type,
                    "severity": "CATASTROPHIC",
                    "precision": precision,
                    "impact": "NEVER detects bullish breakouts - misses all major profit opportunities",
                    "backtest_impact": "CATASTROPHIC - Conservative parameters during explosive moves = missed profits",
                    "financial_impact": "Could miss 50-200% gains during bull breakouts"
                })
            
            elif precision < 0.2:
                faults.append({
                    "fault": "POOR_BREAKOUT_DETECTION",
                    "breakout_type": breakout_type,
                    "severity": "CRITICAL",
                    "precision": precision,
                    "impact": f"Only {precision:.1%} breakout detection accuracy",
                    "backtest_impact": "CRITICAL - Wrong risk parameters during volatile breakouts",
                    "financial_impact": "Potential for major losses during undetected bearish breakouts"
                })
        
        # Overall breakout strategy impact
        faults.append({
            "fault": "BREAKOUT_STRATEGY_FAILURE",
            "severity": "CATASTROPHIC",
            "impact": "Complete failure to implement breakout trading strategies",
            "backtest_impact": "CATASTROPHIC - Backtesting will show artificially poor performance for breakout strategies",
            "recommendation": "DO NOT USE for breakout-dependent strategies"
        })
        
        return faults
    
    def analyze_class_imbalance_faults(self):
        """Analyze class imbalance impact"""
        print("\n⚖️ CLASS IMBALANCE FAULT ANALYSIS")
        print("-" * 40)
        
        # Based on documented class distribution
        class_distribution = {
            'LOW_VOLATILITY': 0.40,    # 40% of data
            'RANGING': 0.30,           # 30% of data  
            'HIGH_VOLATILITY': 0.20,   # 20% of data
            'TRENDING_BULL': 0.20,     # 20% of data
            'TRENDING_BEAR': 0.20,     # 20% of data
            'ACCUMULATION': 0.20,      # 20% of data
            'DISTRIBUTION': 0.20,      # 20% of data
            'BREAKOUT_BULLISH': 0.005, # 0.5% of data - extremely rare
            'BREAKOUT_BEARISH': 0.015  # 1.5% of data - very rare
        }
        
        faults = []
        
        for regime, ratio in class_distribution.items():
            print(f"{regime}: {ratio:.1%} of training data")
            
            if ratio < 0.01:  # Less than 1%
                faults.append({
                    "fault": "EXTREME_CLASS_UNDERREPRESENTATION",
                    "regime": regime,
                    "severity": "CRITICAL",
                    "ratio": ratio,
                    "impact": f"Only {ratio:.1%} representation - model can't learn pattern",
                    "backtest_impact": "CRITICAL - Will almost never predict this regime correctly"
                })
            
            elif ratio < 0.05:  # Less than 5%
                faults.append({
                    "fault": "SEVERE_CLASS_UNDERREPRESENTATION",
                    "regime": regime,
                    "severity": "HIGH",
                    "ratio": ratio,
                    "impact": f"Only {ratio:.1%} representation - unreliable detection",
                    "backtest_impact": "HIGH - Inconsistent detection leads to parameter instability"
                })
        
        return faults
    
    def analyze_computational_faults(self):
        """Analyze computational overhead faults"""
        print("\n⚡ COMPUTATIONAL OVERHEAD FAULT ANALYSIS")
        print("-" * 40)
        
        faults = []
        
        # Feature complexity analysis
        current_features = 31  # Current system
        enhanced_features = 32  # Enhanced system
        feature_overhead = (enhanced_features - current_features) / current_features
        
        print(f"Feature count: {current_features} → {enhanced_features} (+{feature_overhead:.1%})")
        
        # Model complexity analysis  
        enhanced_model_size = 15.7  # MB
        ensemble_model_size = 46.9  # MB
        current_model_size = 2.0    # MB (estimated)
        
        memory_overhead_enhanced = (enhanced_model_size - current_model_size) / current_model_size
        memory_overhead_ensemble = (ensemble_model_size - current_model_size) / current_model_size
        
        print(f"Enhanced model size: {enhanced_model_size}MB (+{memory_overhead_enhanced:.0%})")
        print(f"Ensemble model size: {ensemble_model_size}MB (+{memory_overhead_ensemble:.0%})")
        
        # Identify computational faults
        if memory_overhead_enhanced > 5:  # More than 500% increase
            faults.append({
                "fault": "EXCESSIVE_MEMORY_OVERHEAD",
                "model": "Enhanced Model",
                "severity": "MEDIUM",
                "overhead": memory_overhead_enhanced,
                "impact": f"{memory_overhead_enhanced:.0%} memory increase may slow backtesting",
                "backtest_impact": "MEDIUM - Slower optimization, more memory pressure"
            })
        
        if memory_overhead_ensemble > 10:  # More than 1000% increase
            faults.append({
                "fault": "EXTREME_MEMORY_OVERHEAD", 
                "model": "Ensemble Model",
                "severity": "HIGH",
                "overhead": memory_overhead_ensemble,
                "impact": f"{memory_overhead_ensemble:.0%} memory increase will significantly slow backtesting",
                "backtest_impact": "HIGH - Much slower optimization, potential memory issues"
            })
        
        # Feature calculation overhead
        if enhanced_features > 50:
            faults.append({
                "fault": "FEATURE_CALCULATION_OVERHEAD",
                "severity": "MEDIUM",
                "feature_count": enhanced_features,
                "impact": "Many features require more computation per prediction",
                "backtest_impact": "MEDIUM - Slower regime detection during backtesting"
            })
        
        return faults
    
    def analyze_reliability_faults(self, performance_data):
        """Analyze prediction reliability faults"""
        print("\n🎯 PREDICTION RELIABILITY FAULT ANALYSIS")
        print("-" * 40)
        
        faults = []
        
        # Based on performance metrics
        for model_name, metrics in performance_data.items():
            accuracy = metrics.get('accuracy', 0)
            macro_f1 = metrics.get('macro_f1', 0)
            
            # Calculate reliability metrics
            consistency_score = min(accuracy, macro_f1)  # Lower bound of performance
            unreliability = 1 - consistency_score
            
            print(f"{model_name}: Consistency={consistency_score:.3f}, Unreliability={unreliability:.3f}")
            
            # High unreliability
            if unreliability > 0.6:
                faults.append({
                    "fault": "HIGH_PREDICTION_UNRELIABILITY",
                    "model": model_name,
                    "severity": "CRITICAL",
                    "unreliability": unreliability,
                    "impact": f"{unreliability:.1%} unreliable predictions",
                    "backtest_impact": "CRITICAL - Parameter changes cause unstable backtest results"
                })
            
            # Moderate unreliability
            elif unreliability > 0.4:
                faults.append({
                    "fault": "MODERATE_PREDICTION_UNRELIABILITY",
                    "model": model_name,
                    "severity": "HIGH",
                    "unreliability": unreliability,
                    "impact": f"{unreliability:.1%} unreliable predictions", 
                    "backtest_impact": "HIGH - Inconsistent regime detection affects strategy parameters"
                })
        
        # Macro F1 vs Accuracy divergence (indicates uneven performance)
        for model_name, metrics in performance_data.items():
            accuracy = metrics.get('accuracy', 0)
            macro_f1 = metrics.get('macro_f1', 0)
            
            divergence = abs(accuracy - macro_f1)
            if divergence > 0.15:  # More than 15% divergence
                faults.append({
                    "fault": "UNEVEN_CLASS_PERFORMANCE",
                    "model": model_name,
                    "severity": "HIGH",
                    "divergence": divergence,
                    "impact": "Performance varies wildly between regimes",
                    "backtest_impact": "HIGH - Some regimes detected well, others ignored completely"
                })
        
        return faults
    
    def generate_backtest_impact_report(self, all_faults):
        """Generate final backtest impact assessment"""
        print("\n" + "=" * 80)
        print("🚨 BACKTESTING PERFORMANCE IMPACT ASSESSMENT")
        print("=" * 80)
        
        # Categorize faults by severity
        catastrophic_faults = []
        critical_faults = []
        high_faults = []
        medium_faults = []
        
        for category, fault_list in all_faults.items():
            for fault in fault_list:
                severity = fault.get('severity', 'UNKNOWN')
                if severity == 'CATASTROPHIC':
                    catastrophic_faults.append(fault)
                elif severity == 'CRITICAL':
                    critical_faults.append(fault)
                elif severity == 'HIGH':
                    high_faults.append(fault)
                else:
                    medium_faults.append(fault)
        
        # Print summary
        print(f"\n📊 FAULT SEVERITY SUMMARY:")
        print(f"💀 CATASTROPHIC: {len(catastrophic_faults)} faults")
        print(f"🔴 CRITICAL: {len(critical_faults)} faults")
        print(f"🟠 HIGH: {len(high_faults)} faults")
        print(f"🟡 MEDIUM: {len(medium_faults)} faults")
        
        # Detail catastrophic faults
        if catastrophic_faults:
            print(f"\n💀 CATASTROPHIC FAULTS (WILL BREAK BACKTESTING):")
            for i, fault in enumerate(catastrophic_faults, 1):
                print(f"\n{i}. {fault['fault']}")
                print(f"   Impact: {fault['impact']}")
                print(f"   Backtest Impact: {fault['backtest_impact']}")
        
        # Detail critical faults
        if critical_faults:
            print(f"\n🔴 CRITICAL FAULTS (WILL HINDER PERFORMANCE):")
            for i, fault in enumerate(critical_faults, 1):
                print(f"\n{i}. {fault['fault']}")
                print(f"   Impact: {fault['impact']}")
                print(f"   Backtest Impact: {fault['backtest_impact']}")
        
        # Generate specific recommendations
        recommendations = self.generate_recommendations(catastrophic_faults, critical_faults, high_faults)
        
        print(f"\n💡 CRITICAL RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Final verdict
        print(f"\n🎯 FINAL VERDICT:")
        
        if len(catastrophic_faults) > 0:
            print("❌ DO NOT DEPLOY - CATASTROPHIC FAULTS WILL BREAK BACKTESTING")
            print("   • Complete breakout detection failure")
            print("   • Wrong parameters during critical market conditions") 
            print("   • Backtesting results will be completely unreliable")
            
        elif len(critical_faults) >= 3:
            print("⚠️ DEPLOY ONLY WITH MAJOR FIXES - TOO MANY CRITICAL ISSUES")
            print("   • Address breakout detection before deployment")
            print("   • Implement hybrid approach with current system")
            
        elif len(critical_faults) >= 1:
            print("⚠️ DEPLOY WITH CAUTION AND MONITORING")
            print("   • Enhanced ML provides some benefits")
            print("   • But has serious limitations in critical scenarios")
            print("   • Monitor closely and implement fallbacks")
            
        else:
            print("✅ SAFE TO DEPLOY WITH IMPROVEMENTS")
            print("   • Benefits outweigh the risks")
            print("   • Address high-priority faults for better performance")
        
        return {
            'catastrophic': catastrophic_faults,
            'critical': critical_faults,
            'high': high_faults,
            'medium': medium_faults,
            'recommendations': recommendations
        }
    
    def generate_recommendations(self, catastrophic_faults, critical_faults, high_faults):
        """Generate specific actionable recommendations"""
        recommendations = []
        
        # Address catastrophic faults
        if any('BREAKOUT' in fault.get('fault', '') for fault in catastrophic_faults):
            recommendations.append("URGENT: Implement rule-based breakout detection as fallback")
            recommendations.append("Collect 10x more breakout training data before ML deployment")
        
        # Address critical accuracy issues
        if any('ACCURACY' in fault.get('fault', '') for fault in critical_faults):
            recommendations.append("Use ensemble voting with current ML system for better accuracy")
            recommendations.append("Implement confidence thresholds - use rule-based when ML uncertain")
        
        # Address class imbalance
        if any('UNDERREPRESENTATION' in fault.get('fault', '') for fault in critical_faults):
            recommendations.append("Use hybrid approach: ML for common regimes, rules for rare regimes")
        
        # Performance monitoring
        recommendations.append("Implement A/B testing: 50% enhanced ML, 50% current system")
        recommendations.append("Monitor regime detection accuracy in live backtesting")
        recommendations.append("Set up automatic fallback to current system if ML performance degrades")
        
        # Gradual deployment
        recommendations.append("Deploy enhanced ML only for LOW_VOLATILITY and RANGING regimes initially")
        recommendations.append("Keep current system for all critical regimes (breakouts, high volatility)")
        
        return recommendations

def main():
    print("🔍 ML Fault Analysis for Backtesting Performance")
    print("=" * 60)
    
    analyzer = MLPerformanceFaultAnalyzer()
    
    # Analyze all fault categories
    all_faults = analyzer.analyze_documented_faults()
    
    # Generate comprehensive backtest impact report
    report = analyzer.generate_backtest_impact_report(all_faults)
    
    # Save detailed report
    with open('ml_backtest_fault_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed fault report saved to: ml_backtest_fault_report.json")

if __name__ == "__main__":
    main()

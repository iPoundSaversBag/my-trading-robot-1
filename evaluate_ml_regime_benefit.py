"""Evaluate net benefit of ML layer in ProductionRegimeDetector.

Runs a sample of regime detections across historical 5m data twice:
  1. With ML enabled (default)
  2. With ML disabled (REGIME_USE_ML=0)

Collects:
  - Regime distribution deltas
  - ML acceptance rate
  - Average rule time vs ML overhead
  - Agreement rate between ML-enabled and rule-only outputs
  - Divergence regimes & confidence summary

Heuristic benefit score combines (accept_rate * divergence_precision_proxy) - overhead_penalty.
"""
import os
import json
import argparse
import copy
import pandas as pd
from core.production_regime_detector import ProductionRegimeDetector, MarketRegime  # noqa: F401
from utilities.utils import load_master_config, atomic_update_master_config

def load_data(path: str = "data/crypto_data_5m.parquet"):
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df

def sample_indices(df, stride: int, max_points: int):
    start = 3000  # warm-up for indicators
    idx = list(range(start, len(df), stride))
    if max_points:
        idx = idx[:max_points]
    return idx

def _set_config_ml_enabled(enabled: bool):
    """Temporarily toggle ML flag in master config using atomic update; restore caller ensures revert."""
    cfg = load_master_config()
    regime_block = cfg.setdefault('regime_detection', {}).setdefault('ml', {})
    prev = regime_block.get('enabled', False)
    if prev == enabled:
        return prev
    regime_block['enabled'] = enabled
    atomic_update_master_config(cfg, reason=f"temp_ml_toggle_{enabled}", suppress_trigger=True)
    return prev

def run_pass(df, indices, use_ml: bool):
    # Toggle via config (preferred) instead of env var.
    prev = _set_config_ml_enabled(use_ml)
    det = ProductionRegimeDetector()  # reads config-based flag
    regimes = []
    confidences = []
    for i in indices:
        window = df.iloc[:i]
        r, c = det.detect_regime_ml(window)
        regimes.append(r)
        confidences.append(c)
    stats = det.get_runtime_stats()
    # Restore previous flag state if changed
    _set_config_ml_enabled(prev)
    return regimes, confidences, stats

def distribution(regimes):
    total = len(regimes)
    counts = {}
    for r in regimes:
        key = getattr(r, 'name', str(r))
        counts[key] = counts.get(key, 0)+1
    return {k: v/total for k,v in counts.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=60, help='Stride between evaluation points')
    ap.add_argument('--max-points', type=int, default=1000, help='Maximum evaluation points')
    ap.add_argument('--data', type=str, default='data/crypto_data_5m.parquet')
    args = ap.parse_args()

    df = load_data(args.data)
    indices = sample_indices(df, args.stride, args.max_points)
    # Pass 1: ML enabled
    ml_regimes, ml_conf, ml_stats = run_pass(df, indices, use_ml=True)
    # Pass 2: ML disabled
    rule_regimes, rule_conf, rule_stats = run_pass(df, indices, use_ml=False)

    # Agreement & divergence
    agree = sum(1 for a,b in zip(ml_regimes, rule_regimes) if a==b)
    agreement_rate = agree/len(ml_regimes)
    divergences = [i for i,(a,b) in enumerate(zip(ml_regimes, rule_regimes)) if a!=b]

    ml_dist = distribution(ml_regimes)
    rule_dist = distribution(rule_regimes)

    # Simple precision proxy: proportion of ML divergences that are high confidence
    high_conf_div = 0
    for i in divergences:
        if ml_conf[i] >= 0.75:
            high_conf_div += 1
    precision_proxy = (high_conf_div / len(divergences)) if divergences else 0

    # Benefit heuristic (tunable):
    # benefit = (ml_accept_rate * precision_proxy) - overhead_ratio
    overhead_ratio = (ml_stats.get('avg_ml_overhead_ms',0) / max(1e-6, ml_stats.get('avg_rule_time_ms',1)))
    ml_accept = ml_stats.get('ml_accept_rate',0)
    benefit_score = (ml_accept * precision_proxy) - overhead_ratio * 0.1

    report = {
        'points_evaluated': len(indices),
        'agreement_rate': agreement_rate,
        'divergence_count': len(divergences),
        'ml_distribution': ml_dist,
        'rule_distribution': rule_dist,
        'ml_stats': ml_stats,
        'rule_stats': rule_stats,
        'high_conf_divergence_fraction': precision_proxy,
        'benefit_score': benefit_score,
        'overhead_ratio': overhead_ratio,
        'ml_accept_rate': ml_accept,
    }
    print(json.dumps(report, indent=2, default=str))

if __name__ == '__main__':
    main()

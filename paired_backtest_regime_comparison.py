"""Paired Backtest Regime Comparison Harness
============================================

Runs two backtests (rule-only vs rule+ML) and reports PnL / risk metric deltas.

Workflow:
 1. Run baseline (REGIME_USE_ML=0)
 2. Run ML-enabled (REGIME_USE_ML=1 / unset)
 3. Locate newest metrics JSON artifacts (heuristic search)
 4. Extract key metrics (Sharpe, CAGR, Return %, MaxDD %, WinRate, Trades, ProfitFactor if present)
 5. Compute absolute / relative deltas, regime entropy (if regime distribution file found), and summary benefit score.

Usage:
  python paired_backtest_regime_comparison.py --config configs/optimization_config.json \
      --baseline-tag rule --ml-tag ml --fast --out comparison_report.json

Flags:
  --fast : sets env hints to speed backtest (if respected by engine)
  --skip-ml : only run baseline (debug)
  --only-ml : only run ml variant (debug)

Environment Variables Set (hints):
  REGIME_USE_ML=0/1 toggles ML inside ProductionRegimeDetector.
  REGIME_VALIDATE_STRIDE / REGIME_VALIDATE_MAX_STEPS may be used by downstream validation (optional).

Note: This harness treats the backtest as a black box; adjust the `METRIC_FILE_CANDIDATES` list if your
engine writes metrics under different filenames.
"""
from __future__ import annotations
import os, sys, json, time, argparse, pathlib, subprocess, math
from typing import Dict, Any, Optional, Tuple
from utilities.utils import load_master_config, atomic_update_master_config

ROOT = pathlib.Path(__file__).parent.resolve()
BACKTEST_SCRIPT = ROOT / 'core' / 'backtest.py'

METRIC_FILE_CANDIDATES = [
    'backtest_metrics.json',
    'final_metrics.json',
    'final_results.json',
    'summary_metrics.json',
    'performance_metrics.json'
]

def _set_config_ml_enabled(enabled: bool) -> bool:
    cfg = load_master_config()
    regime_block = cfg.setdefault('regime_detection', {}).setdefault('ml', {})
    prev = regime_block.get('enabled', False)
    if prev == enabled:
        return prev
    regime_block['enabled'] = enabled
    atomic_update_master_config(cfg, reason=f"paired_compare_toggle_{enabled}", suppress_trigger=True)
    return prev

def run_backtest(env_overrides: Dict[str,str], tag: str, config_path: str, extra_args: Optional[list]=None, ml_enabled: Optional[bool]=None) -> Tuple[Optional[pathlib.Path], float]:
    """Run backtest subprocess adjusting config ML flag (preferred) and returning metrics path & duration."""
    if not BACKTEST_SCRIPT.exists():
        print(f"❌ Backtest script not found at {BACKTEST_SCRIPT}")
        return None, 0.0
    start = time.time()
    env = os.environ.copy()
    env.update(env_overrides)
    # Apply ML toggle via config if specified
    prev_state = None
    if ml_enabled is not None:
        prev_state = _set_config_ml_enabled(ml_enabled)
    # Heuristic: capture pre-run dirs to detect new
    before_dirs = set(p.name for p in ROOT.glob('backtest_runs_*'))
    cmd = [sys.executable, str(BACKTEST_SCRIPT), '--config', config_path]
    if extra_args:
        cmd.extend(extra_args)
    print(f"▶️  Running {tag} backtest: {' '.join(cmd)} (config.ml.enabled={ml_enabled})")
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=None)
    except Exception as e:
        print(f"💥 Backtest {tag} failed to start: {e}")
        if prev_state is not None:
            _set_config_ml_enabled(prev_state)
        return None, 0.0
    duration = time.time() - start
    if proc.returncode != 0:
        print(f"💥 Backtest {tag} exited with code {proc.returncode}")
        print(proc.stderr[-1000:])
    # Attempt to locate metrics file(s): search for candidates modified in last 15 min
    metrics_file = locate_metrics_file()
    if metrics_file:
        print(f"📄 {tag} metrics file: {metrics_file}")
    else:
        print(f"⚠️  {tag} metrics file not found (searched {METRIC_FILE_CANDIDATES})")
    # restore previous ML flag
    if prev_state is not None:
        _set_config_ml_enabled(prev_state)
    return metrics_file, duration

def locate_metrics_file() -> Optional[pathlib.Path]:
    newest: Optional[pathlib.Path] = None
    for cand in METRIC_FILE_CANDIDATES:
        for p in ROOT.rglob(cand):
            try:
                if time.time() - p.stat().st_mtime < 60*30:  # modified last 30 min
                    if newest is None or p.stat().st_mtime > newest.stat().st_mtime:
                        newest = p
            except FileNotFoundError:
                continue
    return newest

KEYS = [
    ('sharpe','Sharpe'), ('sharpe_ratio','Sharpe'), ('cagr','CAGR'), ('annual_return','AnnualReturn'),
    ('total_return','TotalReturn'), ('return_pct','ReturnPct'), ('max_drawdown','MaxDD'), ('max_drawdown_pct','MaxDDPct'),
    ('win_rate','WinRate'), ('winrate','WinRate'), ('trades','Trades'), ('trade_count','Trades'),
    ('profit_factor','ProfitFactor')
]

def load_metrics(path: Optional[pathlib.Path]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path,'r',encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"⚠️  Failed reading metrics {path}: {e}")
        return {}

def extract_standard_metrics(raw: Dict[str,Any]) -> Dict[str,float]:
    flat = {}
    # Flatten nested dicts shallowly
    for k,v in raw.items():
        if isinstance(v, dict):
            for k2,v2 in v.items():
                flat[f"{k}.{k2}"] = v2
        else:
            flat[k] = v
    out = {}
    for keyspec, std in KEYS:
        for fk,val in flat.items():
            if fk.lower().endswith(keyspec.lower()):
                try:
                    out[std] = float(val)
                except Exception:
                    continue
    return out

def pct_change(new: float, old: float) -> float:
    if old == 0 or math.isclose(old,0.0):
        return float('inf') if new>0 else 0.0
    return (new-old)/abs(old)

def regime_entropy(distribution: Dict[str,float]) -> float:
    import math
    h=0.0
    for p in distribution.values():
        if p>0:
            h -= p*math.log(p,2)
    return h

def compute_benefit(baseline: Dict[str,float], ml: Dict[str,float]) -> Dict[str,Any]:
    report = {}
    for k in set(baseline.keys())|set(ml.keys()):
        b = baseline.get(k)
        m = ml.get(k)
        if b is None or m is None:
            continue
        report[k] = {
            'baseline': b,
            'ml': m,
            'abs_delta': m-b,
            'pct_delta': pct_change(m,b)
        }
    # Simple composite score: prioritize Sharpe, drawdown improvement, return
    sharpe_gain = report.get('Sharpe',{}).get('abs_delta',0)
    dd_improve = -(report.get('MaxDD',{}).get('abs_delta',0))  # negative delta is improvement
    return_gain = report.get('TotalReturn',{}).get('abs_delta',0) or report.get('ReturnPct',{}).get('abs_delta',0)
    composite = sharpe_gain*1.5 + dd_improve*0.7 + return_gain*0.3
    report['__composite_score'] = composite
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Backtest config path')
    ap.add_argument('--fast', action='store_true', help='Attempt faster runs via env hints')
    ap.add_argument('--baseline-tag', default='baseline')
    ap.add_argument('--ml-tag', default='ml')
    ap.add_argument('--out', default='paired_backtest_comparison.json')
    ap.add_argument('--skip-ml', action='store_true')
    ap.add_argument('--only-ml', action='store_true')
    ap.add_argument('--extra-arg', action='append', default=[], help='Extra arg to pass to backtest (repeatable)')
    args = ap.parse_args()

    # argparse converts '--only-ml' to 'only_ml'
    if args.skip_ml and args.only_ml:
        print('Conflicting flags: both --skip-ml and --only-ml set.')
        sys.exit(1)

    extra = args.extra_arg
    env_fast = {}
    if args.fast:
        env_fast.update({
            'BT_FAST_MODE':'1',
            'REGIME_VALIDATE_STRIDE':'20',
            'REGIME_VALIDATE_MAX_STEPS':'4000'
        })

    baseline_metrics_path = None
    ml_metrics_path = None
    baseline_duration = 0.0
    ml_duration = 0.0

    if not args.only_ml:
        env_base = {**env_fast}
        baseline_metrics_path, baseline_duration = run_backtest(env_base, args.baseline_tag, args.config, extra, ml_enabled=False)
    if not args.skip_ml:
        env_ml = {**env_fast}
        ml_metrics_path, ml_duration = run_backtest(env_ml, args.ml_tag, args.config, extra, ml_enabled=True)

    baseline_raw = load_metrics(baseline_metrics_path)
    ml_raw = load_metrics(ml_metrics_path)
    baseline_std = extract_standard_metrics(baseline_raw)
    ml_std = extract_standard_metrics(ml_raw)

    benefit = compute_benefit(baseline_std, ml_std) if baseline_std and ml_std else {}

    # Regime distribution entropy if present
    baseline_regime_dist = baseline_raw.get('regime_distribution') if isinstance(baseline_raw, dict) else None
    ml_regime_dist = ml_raw.get('regime_distribution') if isinstance(ml_raw, dict) else None
    entropies = {}
    if isinstance(baseline_regime_dist, dict) and isinstance(ml_regime_dist, dict):
        entropies = {
            'baseline_entropy': regime_entropy(baseline_regime_dist),
            'ml_entropy': regime_entropy(ml_regime_dist),
            'entropy_delta': regime_entropy(ml_regime_dist) - regime_entropy(baseline_regime_dist)
        }
    report = {
        'config': args.config,
        'baseline_metrics_file': str(baseline_metrics_path) if baseline_metrics_path else None,
        'ml_metrics_file': str(ml_metrics_path) if ml_metrics_path else None,
        'baseline_duration_sec': baseline_duration,
        'ml_duration_sec': ml_duration,
        'baseline_metrics_extracted': baseline_std,
        'ml_metrics_extracted': ml_std,
        'pnl_delta_report': benefit,
        'regime_entropy': entropies,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(args.out,'w',encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Comparison report written: {args.out}")
    if benefit:
        print(f"Composite Benefit Score: {benefit.get('__composite_score'):.4f}")

if __name__ == '__main__':
    main()

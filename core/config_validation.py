import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["optimization_settings"],
    "properties": {
        "optimization_settings": {
            "type": "object",
            "required": ["adaptive_master"],
            "properties": {
                "adaptive_master": {"type": "object"}
            }
        }
    }
}

ADAPTIVE_KEY_DOCS: Dict[str, str] = {
    "search_space_contraction": "Enable shrinking parameter bounds based on convergence.",
    "contraction_every": "Apply contraction every N windows once minimum windows reached.",
    "contraction_factor": "Fraction of current width retained during contraction (toward center).",
    "min_relative_width": "Hard floor for relative width vs original hard bounds.",
    "search_space_expansion": "Allow expansion of previously contracted bounds.",
    "expansion_factor": "Base proportion of distance back to original added during expansion.",
    "expansion_factor_max": "Ceiling for adaptive expansion scaling.",
    "expand_overfit_events": "Number of overfit or penalty triggers required to force expansion.",
    "expansion_stagnation_patience": "Windows without improvement before expansion considered.",
    "expansion_penalty_ratio_threshold": "If penalties / raw_score exceed this, consider expansion.",
    "expansion_false_signal_relax_threshold": "False signal EMA level where expansion bias increases.",
    "targeted_boundary_tolerance_pct": "Percent of width defining a boundary hug event.",
    "boundary_hugging_ratio_threshold": "Ratio of boundary hugs / observations to trigger targeted expansion.",
    "targeted_expansion_factor": "Fraction of remaining distance used for targeted (per-param) expansion.",
    "expand_cooldown_windows": "Param-level cooldown windows before re-expanding same param.",
    "global_expand_cooldown_windows": "Global cooldown before any further expansion actions.",
    "directional_min_step_fraction": "Minimum directional step fraction when expanding only one side.",
    "skip_if_overfit": "Block expansion when overfit condition detected.",
    "dynamic_calls_enabled": "Enable dynamic adjustment of optimization trials per window.",
    "calls_scaling_weights": "Weights for composite metric guiding calls multiplier adjustments.",
    "regime_auto_tune": "Enable automatic adjustments of regime multipliers.",
    "regime_tune_every": "Frequency (windows) of regime tuning cycle.",
    "regime_min_trades": "Minimum trades in regime to be considered for tuning.",
    "regime_step": "Increment size for regime multiplier adjustments.",
    "min_regime_mult": "Lower bound for regime multipliers.",
    "max_regime_mult": "Upper bound for regime multipliers.",
    "regime_ema_alpha": "Ewma smoothing for regime performance signals.",
    "risk_scaling_enabled": "Enable adaptive risk tier scaling.",
    "risk_scaling_lookback": "Lookback windows for risk tier evaluation.",
    "risk_scaling_sharpe_aggressive": "Sharpe threshold to promote aggressive tier.",
    "risk_scaling_sharpe_conservative": "Sharpe threshold below which conservative tier enforced.",
    "risk_scaling_drawdown_conservative": "Drawdown level above which conservative tier triggered.",
    "risk_scaling_false_rate_conservative": "False signal rate above which conservative tier enforced.",
    "risk_scaling_false_rate_aggressive": "False signal rate below which aggressive tier considered.",
    "risk_scaling_multiplier_conservative": "Position/risk multiplier for conservative tier.",
    "risk_scaling_multiplier_normal": "Position/risk multiplier for normal tier.",
    "risk_scaling_multiplier_aggressive": "Position/risk multiplier for aggressive tier.",
    "allowed_max_drawdown_dynamic_shift": "Fraction of allowed DD at which score weighting shifts (e.g., Sharpe->Calmar).",
    "false_signal_estimation_settings": "Nested settings controlling false signal classification & smoothing.",
    "false_signal_control": "Penalty bases and amplification limits for false signal penalties.",
    "penalty_auto_calibration": "Enable dynamic tuning of penalty base magnitudes.",
    "penalty_auto_calibration_settings": "Parameters for penalty auto calibration routine.",
    "penalty_component_rebalance": "Settings to normalize contribution of distinct penalty categories.",
    "advanced_variance": "Confidence interval & drift-based variance control settings.",
    "drift_detection_enabled": "Master toggle for drift detection module.",
    "drift_detection_settings": "Parameters guiding detection of performance drift.",
    "adaptive_dashboard_export": "Controls lightweight adaptive state snapshot export.",
    "fragility_penalty_base": "Base multiplier applied to fragility penalty component.",
    "overfit_sharpe_delta": "Sharpe delta between train and validation indicating overfit risk."
}

MISSING_SEVERITY = {
    # key: (required, default)
    # For now treat most as optional; schema extension future work.
}


def collect_adaptive_master_keys(cfg: Dict[str, Any]) -> List[str]:
    return list(cfg.get('optimization_settings', {}).get('adaptive_master', {}).keys())


def find_unknown_keys(cfg: Dict[str, Any]) -> List[str]:
    am = cfg.get('optimization_settings', {}).get('adaptive_master', {})
    return [k for k in am.keys() if k not in ADAPTIVE_KEY_DOCS]


def generate_documentation_table() -> List[Tuple[str, str]]:
    return sorted(ADAPTIVE_KEY_DOCS.items(), key=lambda x: x[0])


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    issues: Dict[str, List[str]] = {"missing": [], "unknown": []}
    am = cfg.get('optimization_settings', {}).get('adaptive_master')
    if am is None:
        issues['missing'].append('optimization_settings.adaptive_master')
        return issues
    unknown = find_unknown_keys(cfg)
    if unknown:
        issues['unknown'].extend(unknown)
    return issues


def write_docs_markdown(path: str = 'docs/adaptive_master_keys.md') -> None:
    lines = ["# Adaptive Master Configuration Keys", "", "Key | Description", "--- | ---"]
    for k, desc in generate_documentation_table():
        lines.append(f"{k} | {desc}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding='utf-8')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='core/optimization_config.json')
    p.add_argument('--write-docs', action='store_true')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    issues = validate_config(config)
    if issues['missing'] or issues['unknown']:
        print('[CONFIG VALIDATION] Issues detected:')
        if issues['missing']:
            print(' Missing:', ', '.join(issues['missing']))
        if issues['unknown']:
            print(' Unknown:', ', '.join(issues['unknown']))
    else:
        print('[CONFIG VALIDATION] No structural issues found.')
    if args.write_docs:
        write_docs_markdown()
        print('[CONFIG VALIDATION] Documentation markdown generated.')

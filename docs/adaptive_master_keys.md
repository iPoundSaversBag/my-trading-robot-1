# Adaptive Master Configuration Keys

Key | Description
--- | ---
allowed_max_drawdown_dynamic_shift | Fraction of allowed DD at which score weighting shifts (e.g., Sharpe->Calmar).
adaptive_dashboard_export | Controls lightweight adaptive state snapshot export.
adjust_rate | (See penalty_component_rebalance.adjust_rate) Rate of adjustment toward target component share.
advanced_variance | Confidence interval & drift-based variance control settings.
boundary_hugging_ratio_threshold | Ratio of boundary hugs / observations to trigger targeted expansion.
calls_scaling_weights | Weights for composite metric guiding calls multiplier adjustments.
contraction_every | Apply contraction every N windows once minimum windows reached.
contraction_factor | Fraction of current width retained during contraction (toward center).
directional_min_step_fraction | Minimum directional step fraction when expanding only one side.
drift_detection_enabled | Master toggle for drift detection module.
drift_detection_settings | Parameters guiding detection of performance drift.
expansion_factor | Base proportion of distance back to original added during expansion.
expansion_factor_max | Ceiling for adaptive expansion scaling.
expansion_false_signal_relax_threshold | False signal EMA level where expansion bias increases.
expansion_penalty_ratio_threshold | If penalties / raw_score exceed this, consider expansion.
expand_cooldown_windows | Param-level cooldown windows before re-expanding same param.
expand_overfit_events | Number of overfit or penalty triggers required to force expansion.
fragility_penalty_base | Base multiplier applied to fragility penalty component.
false_signal_control | Penalty bases and amplification limits for false signal penalties.
false_signal_estimation_settings | Nested settings controlling false signal classification & smoothing.
max_contractions | Maximum number of cumulative contractions.
max_expansions | Maximum number of cumulative expansions.
min_regime_mult | Lower bound for regime multipliers.
max_regime_mult | Upper bound for regime multipliers.
min_relative_width | Hard floor for relative width vs original hard bounds.
overfit_sharpe_delta | Sharpe delta between train and validation indicating overfit risk.
penalty_auto_calibration | Enable dynamic tuning of penalty base magnitudes.
penalty_auto_calibration_settings | Parameters for penalty auto calibration routine.
penalty_component_rebalance | Settings to normalize contribution of distinct penalty categories.
regime_auto_tune | Enable automatic adjustments of regime multipliers.
regime_ema_alpha | Ewma smoothing for regime performance signals.
regime_min_trades | Minimum trades in regime to be considered for tuning.
regime_step | Increment size for regime multiplier adjustments.
regime_tune_every | Frequency (windows) of regime tuning cycle.
risk_scaling_drawdown_conservative | Drawdown level above which conservative tier triggered.
risk_scaling_enabled | Enable adaptive risk tier scaling.
risk_scaling_false_rate_aggressive | False signal rate below which aggressive tier considered.
risk_scaling_false_rate_conservative | False signal rate above which conservative tier enforced.
risk_scaling_lookback | Lookback windows for risk tier evaluation.
risk_scaling_multiplier_aggressive | Position/risk multiplier for aggressive tier.
risk_scaling_multiplier_conservative | Position/risk multiplier for conservative tier.
risk_scaling_multiplier_normal | Position/risk multiplier for normal tier.
risk_scaling_sharpe_aggressive | Sharpe threshold to promote aggressive tier.
risk_scaling_sharpe_conservative | Sharpe threshold below which conservative tier enforced.
search_space_contraction | Enable shrinking parameter bounds based on convergence.
search_space_expansion | Allow expansion of previously contracted bounds.
skip_if_overfit | Block expansion when overfit condition detected.
targeted_boundary_tolerance_pct | Percent of width defining a boundary hug event.
targeted_expansion_factor | Fraction of remaining distance used for targeted (per-param) expansion.
overfit_sharpe_delta | Sharpe delta threshold to classify overfit risk.
expand_always_global_reasons | Reasons that force global expansion regardless of targeted heuristics.

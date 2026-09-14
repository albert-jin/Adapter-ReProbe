"""Compact public entry points for the Adapter-ReProbe source snapshot.

The full upstream package eagerly imports optional ``lm_polygraph`` calculator
modules.  Those calculators are outside this focused release, so keep the
claim-head import usable without requiring the optional application package.
"""

from .auto_uncertainty_head import AutoUncertaintyHead

try:  # Optional upstream calculator compatibility.
    from .luh_claim_estimator import LuhClaimEstimator
    from .calculator_infer_luh import CalculatorInferLuh
    from .causal_lm_with_uncertainty import CausalLMWithUncertainty
except ModuleNotFoundError:  # pragma: no cover - optional dependency boundary
    LuhClaimEstimator = None
    CalculatorInferLuh = None
    CausalLMWithUncertainty = None

"""Causal Decompiler package.

Public ATE helpers stay importable from `src.engine.causal` so existing
experiments keep working. New code should go through CausalDecompiler.
"""

from .algebra import (
    CausalOp,
    delete_memory,
    lesion,
    observe_lock,
    override_event,
    resample,
    set_policy_lambda,
    skip_event,
)
from .ate import CausalResult, compute_ate, run_causal_experiment
from .decompiler import CausalDecompiler, CausalMRIReport
from .twin import identity_holds, load_factual, run_factual, run_replay, run_twin, sim_config_from_log

__all__ = [
    "CausalDecompiler",
    "CausalMRIReport",
    "CausalOp",
    "CausalResult",
    "compute_ate",
    "delete_memory",
    "identity_holds",
    "lesion",
    "observe_lock",
    "override_event",
    "resample",
    "run_causal_experiment",
    "run_factual",
    "run_replay",
    "sim_config_from_log",
    "load_factual",
    "set_policy_lambda",
    "skip_event",
]

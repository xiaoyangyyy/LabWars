"""LabWars Part 4 鈥?experiment runners and analysis."""

from .aggregate import aggregate_experiment
from .benchmark_tasks import list_benchmark_tasks, run_benchmark_task
from .batch import run_batch
from .conditions import EXPERIMENT_MATRIX, build_sim_config, get_condition
from .llm_mix_ablation import run_dual_engine_ablation, run_llm_mix_ablation
from .organization_ablation import run_organization_ablation
from .policy_mode_comparison import run_policy_mode_comparison
from .report import generate_report
from .runner import run_single
from .scale import ScaleExperimentResult, run_scale_experiment
from .scientific_protocol import ScientificProtocolResult, run_scientific_protocol
from .social_potential_ablation import run_social_potential_ablation, summarize_social_potential_ablation

__all__ = [
    "EXPERIMENT_MATRIX",
    "aggregate_experiment",
    "build_sim_config",
    "generate_report",
    "get_condition",
    "run_batch",
    "list_benchmark_tasks",
    "run_benchmark_task",
    "run_dual_engine_ablation",
    "run_llm_mix_ablation",
    "run_organization_ablation",
    "run_policy_mode_comparison",
    "run_scale_experiment",
    "ScaleExperimentResult",
    "run_scientific_protocol",
    "ScientificProtocolResult",
    "run_single",
    "run_social_potential_ablation",
    "summarize_social_potential_ablation",
]

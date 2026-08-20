"""LabWars cognition layer — continuous social dynamics."""

from .authorship import authorship_dispute_index, compute_authorship_scores, rank_authors
from .belief import behavior_tendency_multipliers, update_beliefs
from .divergence import compute_divergence, mean_divergence
from .emotion import update_emotion
from .memory import RecallResult, compute_valence, recall_memories, write_memory
from .pipeline import CognitiveStepResult, apply_action_cognition, process_event_phase
from .relationship import coalition_strength, credit_threat_density, trust_fragmentation

__all__ = [
    "compute_pressure_fields",
    "summarize_pressure_fields",
    "current_reputation",
    "compute_social_potential",
    "SocialPotential",
    "SOCIAL_POTENTIAL_DIMENSIONS",
    "CognitiveStepResult",
    "RecallResult",
    "apply_action_cognition",
    "authorship_dispute_index",
    "behavior_tendency_multipliers",
    "coalition_strength",
    "compute_authorship_scores",
    "compute_divergence",
    "compute_valence",
    "credit_threat_density",
    "mean_divergence",
    "process_event_phase",
    "rank_authors",
    "recall_memories",
    "trust_fragmentation",
    "update_beliefs",
    "update_emotion",
    "write_memory",
]

from .social_potential import (
    SOCIAL_POTENTIAL_DIMENSIONS,
    SocialPotential,
    compute_social_potential,
    summarize_action_social_potential,
)
from .pressure_fields import compute_pressure_fields, summarize_pressure_fields
from .reputation import current_reputation, update_reputation_from_action

"""LangGraph-ready orchestration contract and transition skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .spec_schema import REQUIRED_ORCHESTRATION_STATE_KEYS, RuntimeConfig, StageId

TRANSITION_GRAPH: dict[StageId, set[StageId]] = {
    "stage_1": {"stage_2"},
    "stage_2": {"stage_2", "stage_3"},
    "stage_3": {"stage_3", "stage_4"},
    "stage_4": {"stage_4", "stage_5"},
    "stage_5": {"stage_5", "stage_6"},
    "stage_6": {"stage_6", "stage_7"},
    "stage_7": set(),
}

HOOK_TO_STAGE: dict[str, StageId] = {
    "low_retrieval_confidence": "stage_2",
    "low_mask_confidence": "stage_3",
    "identity_ambiguity": "stage_4",
    "missing_evidence": "stage_6",
}


@dataclass(frozen=True)
class OrchestratorContract:
    """Minimal orchestration contract that can be wired to LangGraph nodes/edges."""

    required_state_keys: frozenset[str]
    branching_hooks: frozenset[str]

    @classmethod
    def from_runtime_config(cls, config: RuntimeConfig) -> "OrchestratorContract":
        hooks = frozenset(config.phase_a.orchestration.branching_hooks)
        return cls(
            required_state_keys=frozenset(config.phase_a.orchestration.required_state_keys),
            branching_hooks=hooks,
        )

    def validate_state_snapshot(self, snapshot: Mapping[str, object]) -> None:
        missing = self.required_state_keys - set(snapshot.keys())
        if missing:
            raise ValueError(f"state snapshot missing required keys: {sorted(missing)}")

    def can_transition(self, current_stage: StageId, next_stage: StageId) -> bool:
        return next_stage in TRANSITION_GRAPH[current_stage]

    def next_stage_for_hook(self, hook: str) -> StageId:
        if hook not in self.branching_hooks:
            raise ValueError(f"hook '{hook}' is not enabled in this configuration")
        if hook not in HOOK_TO_STAGE:
            raise ValueError(f"hook '{hook}' has no stage mapping")
        return HOOK_TO_STAGE[hook]


def default_required_state_keys() -> frozenset[str]:
    """Export required state keys for integration and test consistency."""

    return frozenset(REQUIRED_ORCHESTRATION_STATE_KEYS)

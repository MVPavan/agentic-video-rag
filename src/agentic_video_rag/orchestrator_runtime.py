"""Orchestration runtime that applies stage transitions and branching hooks."""

from __future__ import annotations

from dataclasses import dataclass, field

from .orchestrator_contract import OrchestratorContract
from .spec_schema import StageId


@dataclass(frozen=True)
class TransitionEvent:
    """Recorded stage transition event for auditability."""

    from_stage: StageId
    to_stage: StageId
    reason: str


@dataclass
class OrchestrationRuntime:
    """LangGraph-like runtime skeleton backed by the orchestration contract."""

    contract: OrchestratorContract
    events: list[TransitionEvent] = field(default_factory=list)

    def transition(self, current_stage: StageId, next_stage: StageId, reason: str) -> StageId:
        if not self.contract.can_transition(current_stage, next_stage):
            raise ValueError(f"invalid transition {current_stage} -> {next_stage}")
        self.events.append(TransitionEvent(current_stage, next_stage, reason))
        return next_stage

    def apply_branching_hook(self, current_stage: StageId, hook: str) -> StageId:
        next_stage = self.contract.next_stage_for_hook(hook)
        return self.transition(current_stage, next_stage, reason=f"hook:{hook}")

    def canonical_flow(self) -> list[StageId]:
        """Return standard forward stage execution order."""

        return [
            "stage_1",
            "stage_2",
            "stage_3",
            "stage_4",
            "stage_5",
            "stage_6",
            "stage_7",
        ]

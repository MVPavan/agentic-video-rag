from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_video_rag import load_runtime_config  # noqa: E402
from agentic_video_rag.orchestrator_contract import OrchestratorContract  # noqa: E402
from agentic_video_rag.orchestrator_runtime import OrchestrationRuntime  # noqa: E402

BASE_CONFIG = ROOT / "config/spec/groundtruth.yaml"


class OrchestrationRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_runtime_config(BASE_CONFIG)
        contract = OrchestratorContract.from_runtime_config(config)
        self.runtime = OrchestrationRuntime(contract=contract)

    def test_canonical_transitions_are_valid(self) -> None:
        flow = self.runtime.canonical_flow()
        for current, next_stage in zip(flow, flow[1:]):
            self.runtime.transition(current, next_stage, reason="forward")

        self.assertEqual(len(self.runtime.events), 6)

    def test_branching_hooks_map_to_expected_stages(self) -> None:
        target = self.runtime.apply_branching_hook("stage_2", "low_retrieval_confidence")
        self.assertEqual(target, "stage_2")

        target = self.runtime.apply_branching_hook("stage_3", "low_mask_confidence")
        self.assertEqual(target, "stage_3")

    def test_invalid_transition_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.runtime.transition("stage_1", "stage_3", reason="invalid")


if __name__ == "__main__":
    unittest.main()

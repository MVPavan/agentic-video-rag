from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_video_rag import OrchestratorContract, load_runtime_config  # noqa: E402

BASE_CONFIG = ROOT / "config/spec/groundtruth.yaml"


class OrchestratorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_runtime_config(BASE_CONFIG)
        self.contract = OrchestratorContract.from_runtime_config(self.config)

    def test_required_state_keys_enforced(self) -> None:
        snapshot = {key: None for key in self.contract.required_state_keys}

        self.contract.validate_state_snapshot(snapshot)

        snapshot.pop("query_id")
        with self.assertRaises(ValueError):
            self.contract.validate_state_snapshot(snapshot)

    def test_transition_rules(self) -> None:
        self.assertTrue(self.contract.can_transition("stage_1", "stage_2"))
        self.assertFalse(self.contract.can_transition("stage_1", "stage_3"))
        self.assertTrue(self.contract.can_transition("stage_6", "stage_7"))

    def test_branching_hook_to_stage_mapping(self) -> None:
        self.assertEqual(self.contract.next_stage_for_hook("low_mask_confidence"), "stage_3")

        with self.assertRaises(ValueError):
            self.contract.next_stage_for_hook("non_existent_hook")


if __name__ == "__main__":
    unittest.main()

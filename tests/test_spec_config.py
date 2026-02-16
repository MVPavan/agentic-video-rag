from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_video_rag import load_runtime_config  # noqa: E402

BASE_CONFIG = ROOT / "config/spec/groundtruth.yaml"
DEV_OVERRIDE = ROOT / "config/spec/overrides/dev.yaml"


class RuntimeConfigTests(unittest.TestCase):
    def test_base_config_loads(self) -> None:
        config = load_runtime_config(BASE_CONFIG)

        self.assertEqual(config.meta.spec_id, "agentic_video_rag")
        self.assertEqual(len(config.phase_b.stages), 7)
        self.assertEqual(config.phase_b.stages[0].stage_name, "activity_ingestion")

    def test_override_merge_precedence_is_deterministic(self) -> None:
        config_a = load_runtime_config(BASE_CONFIG, [DEV_OVERRIDE])
        config_b = load_runtime_config(BASE_CONFIG, [DEV_OVERRIDE])

        self.assertEqual(config_a.constants.retrieval.stage2_validated_top_k_windows, 12)
        self.assertEqual(config_a.constants.retrieval.stage2_min_validation_confidence, 0.55)
        self.assertEqual(config_a.model_dump(), config_b.model_dump())

    def test_unknown_keys_fail_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            override_path = Path(temp_dir) / "bad_override.yaml"
            override_path.write_text("unexpected_top_level:\n  value: 1\n", encoding="utf-8")

            with self.assertRaises(ValidationError):
                load_runtime_config(BASE_CONFIG, [override_path])

    def test_stage_name_mismatch_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            broken_path = Path(temp_dir) / "broken.yaml"

            cfg = OmegaConf.load(BASE_CONFIG)
            cfg.phase_b.stages[0].stage_name = "wrong_name"
            OmegaConf.save(cfg, broken_path)

            with self.assertRaises(ValidationError):
                load_runtime_config(broken_path)


if __name__ == "__main__":
    unittest.main()

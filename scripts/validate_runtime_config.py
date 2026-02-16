from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_video_rag import load_runtime_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate runtime config with OmegaConf + Pydantic")
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "config/spec/groundtruth.yaml",
        help="Base config YAML path",
    )
    parser.add_argument(
        "--override",
        type=Path,
        action="append",
        default=[],
        help="Override YAML path; can be provided multiple times",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.base, args.override)
    print(
        "validated",
        f"spec={config.meta.spec_id}",
        f"version={config.meta.spec_version}",
        f"stages={len(config.phase_b.stages)}",
    )


if __name__ == "__main__":
    main()

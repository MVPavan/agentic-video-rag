from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_video_rag.demo_data import build_red_suv_query_request  # noqa: E402
from agentic_video_rag.pipeline import AgenticVideoRAGEngine  # noqa: E402
from agentic_video_rag.spec_loader import load_runtime_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Agentic Video RAG demo")
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "config/spec/groundtruth.yaml",
        help="Base runtime config path",
    )
    parser.add_argument(
        "--override",
        type=Path,
        action="append",
        default=[],
        help="Optional override config path(s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.base, args.override)
    request = build_red_suv_query_request()
    engine = AgenticVideoRAGEngine(config=config)
    result = engine.run(request)

    payload = {
        "query_id": result.query_id,
        "validated_windows": [window.window_id for window in result.validated_windows],
        "entity_links": [link.entity_id for link in result.entity_links],
        "claims": [claim.text for claim in result.synthesis.claims],
        "metrics": {
            "stage_durations_ms": result.metrics.stage_durations_ms,
            "cache_hits": result.metrics.cache_hits,
            "cache_misses": result.metrics.cache_misses,
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

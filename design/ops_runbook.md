# Operations Runbook

## Purpose
Operational guidance for running the deterministic Agentic Video RAG reference pipeline.

## Runtime Commands
1. Validate config: `.venv/bin/python scripts/validate_runtime_config.py --base config/spec/groundtruth.yaml`
2. Run demo pipeline: `.venv/bin/python scripts/run_pipeline_demo.py --base config/spec/groundtruth.yaml`
3. Run tests: `.venv/bin/python -m unittest discover -s tests -v`

## Incident Response
1. If config validation fails, rollback to last passing `config/spec/groundtruth.yaml` revision.
2. If E2E tests fail, freeze milestone progression and fix failing stage tests first.
3. If evidence completeness regresses, block Stage 7 synthesis from release.

## Observability Checks
1. Confirm stage durations are reported in pipeline result metrics.
2. Confirm cache hits increase on repeated query runs.
3. Confirm all synthesis claims include evidence references.

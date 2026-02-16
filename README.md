# Agentic Video RAG

Reference implementation of a 7-stage agentic Video RAG pipeline with:

1. Strict YAML config management via OmegaConf + Pydantic.
2. Deterministic end-to-end execution for contract validation.
3. Evidence-grounded synthesis outputs with uncertainty handling.

This repository is structured for reliable handoff between coding agents and incremental replacement of deterministic adapters with real model/service integrations.

## Status

Current implementation is a deterministic, testable reference path with milestones `M0.1` through `M8.3` marked complete in `design/execution_plan.md`.

## Source-of-Truth Docs

1. Product/system contracts: `design/spec_groundtruth.md`
2. Live execution tracker: `design/execution_plan.md`
3. Agent operating rules: `AGENTS.md`

## Quickstart

## 1. Environment

```bash
uv sync
```

If you already use the project virtualenv:

```bash
.venv/bin/python -V
```

## 2. Validate Config

```bash
.venv/bin/python scripts/validate_runtime_config.py \
  --base config/spec/groundtruth.yaml \
  --override config/spec/overrides/dev.yaml
```

## 3. Run Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 4. Run End-to-End Demo

```bash
.venv/bin/python scripts/run_pipeline_demo.py \
  --base config/spec/groundtruth.yaml
```

The demo returns:

1. Validated window IDs.
2. Linked entity IDs.
3. Grounded claim text.
4. Stage timings + cache hit/miss metrics.

## Architecture (Stage Map)

| Stage ID | Stage Name | Implementation Anchor |
|---|---|---|
| `stage_1` | `activity_ingestion` | `src/agentic_video_rag/pipeline.py` |
| `stage_2` | `temporal_retrieval` | `src/agentic_video_rag/pipeline.py` |
| `stage_3` | `spatial_grounding` | `src/agentic_video_rag/pipeline.py` |
| `stage_4` | `entity_resolution` | `src/agentic_video_rag/pipeline.py` |
| `stage_5` | `temporal_localization` | `src/agentic_video_rag/pipeline.py` |
| `stage_6` | `graph_memory` | `src/agentic_video_rag/pipeline.py` |
| `stage_7` | `multimodal_synthesis` | `src/agentic_video_rag/pipeline.py` |

Core modules:

1. Config schema/validation: `src/agentic_video_rag/spec_schema.py`
2. Config merge/load: `src/agentic_video_rag/spec_loader.py`
3. Orchestration contract/runtime: `src/agentic_video_rag/orchestrator_contract.py`, `src/agentic_video_rag/orchestrator_runtime.py`
4. Deterministic stage adapters: `src/agentic_video_rag/adapters.py`
5. In-memory store abstractions: `src/agentic_video_rag/stores.py`
6. Typed data contracts: `src/agentic_video_rag/types.py`
7. Synthetic fixtures: `src/agentic_video_rag/demo_data.py`

## Config Governance

Mandatory rules enforced by code:

1. YAML config only (`config/spec/groundtruth.yaml` + overrides).
2. OmegaConf merge order: `base <- overrides`.
3. Pydantic strict validation (`extra="forbid"`).
4. `stage_catalog` is the canonical `stage_id -> stage_name` map.
5. Stage completeness must include exactly `stage_1`..`stage_7`.
6. Stage model/datastore/resource references must resolve.

## Testing Strategy

Test files:

1. `tests/test_spec_config.py`: config schema and merge behavior.
2. `tests/test_orchestrator_contract.py`: orchestration contract validation.
3. `tests/test_orchestration_runtime.py`: transition/hook runtime behavior.
4. `tests/test_pipeline_phases.py`: stage-level acceptance tests for `P2`..`P8`.

Key checks include:

1. Stage routing coverage across all 4 ingestion routes.
2. Retrieval + cache behavior.
3. Grounding fallback.
4. ReID ambiguity handling.
5. Temporal failure flags.
6. Graph evidence completeness.
7. Synthesis redaction for insufficient evidence.
8. End-to-end red SUV regression flow.

## Replacing Deterministic Adapters with Real Models

Use this order to reduce risk:

1. Keep all tests green with deterministic adapters first.
2. Replace one adapter at a time in `src/agentic_video_rag/adapters.py`.
3. Preserve existing data contracts in `src/agentic_video_rag/types.py`.
4. Keep stage outputs evidence-complete before enabling Stage 7.
5. Add/adjust tests for each replacement adapter.

Recommended adapter replacement sequence:

1. `SigLIP2Adapter`
2. `InternVideoNextAdapter` + `IVNextLiTTextHeadAdapter`
3. `SAM3Adapter`
4. `ReIDResolver`
5. `TemporalGroundingAdapter`
6. `MultimodalSynthesizerAdapter`

## Operations and Release Docs

1. Ops runbook: `design/ops_runbook.md`
2. Known limitations: `design/known_limitations.md`
3. Rollback notes: `design/rollback_notes.md`
4. Release sign-off checklist: `design/release_signoff_checklist.md`

# Agentic Video RAG - Execution Plan (Live)

## 1. Purpose

This is the live execution tracker for implementation progress.

1. Product and architecture source of truth remains `design/spec_groundtruth.md`.
2. This file tracks delivery state (phases, milestones, owners, dates, risks).
3. Update this file in every PR/commit that materially changes milestone status.

## 2. Tracking Rules

1. Keep milestone IDs stable (`M0.1` ... `M8.3`).
2. Allowed status values: `not_started`, `in_progress`, `blocked`, `done`.
3. When status changes:
- update `Status`
- update `Start Date` if entering `in_progress`
- update `Completed Date` if entering `done`
- add a brief `Notes / Risks` update
4. Do not mark `done` unless milestone acceptance gate is met.
5. If blocked, include unblock condition in `Notes / Risks`.

## 3. Phase Plan (High-Level)

| Phase | Goal | Primary Stage Coverage |
|---|---|---|
| `P0` | Project foundation and governance setup | All (contracts only) |
| `P1` | Config/schema/orchestration skeleton | Phase A + stage contracts |
| `P2` | Retrieval backbone operational | `stage_1`, `stage_2` |
| `P3` | Spatial grounding operational | `stage_3` |
| `P4` | Entity resolution operational | `stage_4` |
| `P5` | Temporal localization operational | `stage_5` |
| `P6` | Graph memory operational | `stage_6` |
| `P7` | Multimodal synthesis operational | `stage_7` |
| `P8` | End-to-end hardening and release readiness | All |

## 4. Milestone Catalog

### Phase `P0` - Foundation and Governance

| Milestone ID | Description | Deliverables | Acceptance Gate |
|---|---|---|---|
| `M0.1` | Repository baseline finalized | Directory structure (`src/`, `tests/`, `config/`, `scripts/`), root `AGENTS.md`, SSOT spec | Repo conventions match Section 17 of `design/spec_groundtruth.md` and `AGENTS.md` |
| `M0.2` | Execution tracking scaffolding | `design/execution_plan.md` initialized | Status table exists with owners/dates |

### Phase `P1` - Config, Schema, and Orchestration Skeleton

| Milestone ID | Description | Deliverables | Acceptance Gate |
|---|---|---|---|
| `M1.1` | Canonical YAML config schema defined | Base YAML schema with `stage_catalog`, model/store/resource registries, constants | Strict Pydantic validation passes with `extra="forbid"` |
| `M1.2` | OmegaConf merge workflow implemented | Base + env + runtime override merge path | Deterministic merge tests pass |
| `M1.3` | Orchestrator state contract implemented | LangGraph skeleton with required state keys and branching hooks | Contract tests verify required state keys and transitions |

### Phase `P2` - Retrieval Backbone (`activity_ingestion`, `temporal_retrieval`)

| Milestone ID | Description | Deliverables | Acceptance Gate |
|---|---|---|---|
| `M2.1` | Stage 1 route framework complete | 4-route trigger framework (`meta_sync`, `sig_ex_adaptive`, `cv_state`, `bg_motion_trigger`) | Route-selection and fallback tests pass |
| `M2.2` | SigLIP2 indexing pipeline complete | Keyframe extraction, embedding generation, Milvus writes to `FrameIndex_SigLIP2` | Index contract and metadata integrity tests pass |
| `M2.3` | Stage 2 calibrated retrieval complete | Candidate search, IVNext extraction, LiT re-ranking, L1 caching | Top-K validation and confidence-gate tests pass |

### Phase `P3` - Spatial Grounding (`spatial_grounding`)

| Milestone ID | Description | Deliverables | Acceptance Gate |
|---|---|---|---|
| `M3.1` | SAM3 prompting and masklet generation | Query-prompted segmentation + track IDs + confidence | Mask confidence gate tests pass |
| `M3.2` | Grounding fallback logic complete | Decomposition retries and detector+tracker fallback interface | Failure-mode tests pass for low-confidence scenarios |
| `M3.3` | Evidence overlays persisted | Mask/bbox overlay artifact pipeline with refs | Artifact linkage tests pass |

### Phase `P4` - Entity Resolution (`entity_resolution`)

| Milestone ID | Description | Deliverables | Acceptance Gate |
|---|---|---|---|
| `M4.1` | Object/vehicle ReID flow complete | DINOv3 mask-weighted embeddings + clustering output `ObjectClusterID` | Cluster quality and conflict-rate tests pass |
| `M4.2` | Person ReID flow complete | Dedicated person ReID + topology/time fusion output `PersonEntityID` | Cross-camera consistency tests pass |
| `M4.3` | Ambiguity handling complete | Unresolved entity state contract | No forced-link behavior in ambiguity tests |

### Phase `P5` - Temporal Localization (`temporal_localization`)

| Milestone ID | Description | Deliverables | Acceptance Gate |
|---|---|---|---|
| `M5.1` | Foreground feature isolation complete | L2-token-first and L1-fallback isolation path | Feature-path fallback tests pass |
| `M5.2` | Similarity curve and boundary extraction complete | `S(t)`, smoothing, hysteresis segmentation | Temporal-boundary stability tests pass |
| `M5.3` | Failure flag surface complete | Explicit flags (`occlusion`, `low_mask_confidence`, `multi_actor_ambiguity`) | Flag emission tests pass |

### Phase `P6` - Graph Memory (`graph_memory`)

| Milestone ID | Description | Deliverables | Acceptance Gate |
|---|---|---|---|
| `M6.1` | Node upsert contract complete | Object/person/camera/track/zone node pipeline | Node schema and idempotency tests pass |
| `M6.2` | Temporal edge construction complete | `EXITS`, `MOVES_TO` edges with confidence + time ranges | Edge-contract and temporal consistency tests pass |
| `M6.3` | Evidence attachment complete | Clip/frame/overlay/embedding references per claim | Evidence completeness gate meets Section 15 criteria in `design/spec_groundtruth.md` |

### Phase `P7` - Multimodal Synthesis (`multimodal_synthesis`)

| Milestone ID | Description | Deliverables | Acceptance Gate |
|---|---|---|---|
| `M7.1` | Structured evidence package complete | Claim-level bundle for synthesizer inputs | Evidence package schema tests pass |
| `M7.2` | Grounded answer generator complete | Synthesizer output with claim-to-evidence grounding | 100% claim evidence-link test passes |
| `M7.3` | Conservative fallback complete | Redaction/uncertainty behavior for missing evidence | Hallucination-prevention regression tests pass |

### Phase `P8` - End-to-End Hardening and Release

| Milestone ID | Description | Deliverables | Acceptance Gate |
|---|---|---|---|
| `M8.1` | Full pipeline E2E flow complete | End-to-end execution for canonical query blueprint | `red_suv_person_tracking` regression passes |
| `M8.2` | Performance and cache tuning complete | Cache hit-rate metrics, route-cost measurements, bottleneck remediation | Throughput/latency targets defined and met for test workload |
| `M8.3` | Release readiness complete | Ops runbook, known limitations, rollback notes | Sign-off checklist complete |

## 5. Milestone Dependency Order

Implementation should follow this dependency chain:

1. `P0 -> P1`
2. `P1 -> P2`
3. `P2 -> P3`
4. `P3 -> P4`
5. `P4 + P3 -> P5`
6. `P4 + P5 -> P6`
7. `P6 -> P7`
8. `P2..P7 -> P8`

## 6. Execution Tracking Board (Update In-Place)

| Milestone ID | Status | Owner | Start Date | Target Date | Completed Date | Notes / Risks |
|---|---|---|---|---|---|---|
| `M0.1` | `done` | `tbd` | `2026-02-16` | `2026-02-16` | `2026-02-16` | SSOT and agent governance are present. |
| `M0.2` | `done` | `tbd` | `2026-02-16` | `2026-02-16` | `2026-02-16` | Tracking section moved into dedicated execution plan file. |
| `M1.1` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M1.2` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M1.3` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M2.1` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M2.2` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M2.3` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M3.1` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M3.2` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M3.3` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M4.1` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M4.2` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M4.3` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M5.1` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M5.2` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M5.3` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M6.1` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M6.2` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M6.3` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M7.1` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M7.2` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M7.3` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M8.1` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M8.2` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |
| `M8.3` | `not_started` | `tbd` | `tbd` | `tbd` | `tbd` |  |

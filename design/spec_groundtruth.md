# Agentic Video RAG - Ground Truth Specification (SSOT)

## 1. Document Control

- **Document ID:** `AVR-SSOT-V1`
- **Status:** Draft (implementation-ready)
- **Date:** `2026-02-16`
- **Source:** `design/spec_v1.txt`
- **Canonical Purpose:** Single-source, implementation-grade specification for the Agentic Video RAG system.
- **Supersedes:** Informal stage description in `design/spec_v1.txt`.

---

## 2. Problem Statement

The system must answer complex video queries in unconstrained real-world settings, including:

- Object/entity retrieval across cameras.
- Activity/action localization with precise temporal bounds.
- Cross-camera identity continuity.
- Evidence-grounded multimodal synthesis.

The system must avoid ungrounded claims and surface uncertainty when evidence is weak.

---

## 3. Product Goals

1. **Grounded answers:** Every claim must map to verifiable clips, frames, timestamps, and tracked entities.
2. **Temporal precision:** Action events must provide `t_start` and `t_end`, not only coarse retrieval windows.
3. **Entity consistency:** The same person/object should preserve identity across cameras when confidence allows.
4. **Operational efficiency:** Use adaptive ingestion and feature caching to minimize unnecessary heavy compute.
5. **Auditability:** Persist evidence references and versioned embeddings for reproducibility.

## 4. Non-Goals

1. Full end-to-end model training pipeline for all models in this document.
2. Real-time sub-second streaming guarantees across all deployments.
3. Guaranteed identity closure when visual evidence is insufficient.

---

## 5. Core Design Principles

1. **Grounding before language:** Retrieval confidence alone is not proof.
2. **Separation of concerns:** Retrieval, grounding, ReID, memory graphing, and synthesis remain isolated stages with explicit contracts.
3. **DRY by registry references:** Model names, store names, and thresholds are defined once and reused everywhere.
4. **Explicit uncertainty:** Ambiguous links are represented as unresolved or low-confidence, never silently collapsed.
5. **Deterministic config governance:** YAML + OmegaConf + strict Pydantic validation is mandatory.

---

## 6. System Overview

The system is a **7-stage agentic pipeline** orchestrated by LangGraph and backed by:

- **Vector index:** Milvus
- **Graph memory:** Neo4j or FalkorDB
- **Feature cache:** Tiered cache (L1/L2)
- **Evidence assets:** Overlay masks/bboxes and references to raw media

Flow summary:

1. Adaptive ingestion and keyframe indexing
2. Coarse retrieval and temporal re-ranking
3. Spatial grounding and tracking
4. Cross-clip entity resolution (split object/person ReID)
5. Fine temporal grounding
6. Dynamic graph memory construction
7. Grounded multimodal answer synthesis

### 6.1 Canonical Stage IDs and Crisp Names

These names are mandatory for runtime config and schema validation.

| Stage ID | Crisp Stage Name |
|---|---|
| `stage_1` | `activity_ingestion` |
| `stage_2` | `temporal_retrieval` |
| `stage_3` | `spatial_grounding` |
| `stage_4` | `entity_resolution` |
| `stage_5` | `temporal_localization` |
| `stage_6` | `graph_memory` |
| `stage_7` | `multimodal_synthesis` |

---

## 7. Component Registry (Canonical)

| ID | Component | Role | Trainability |
|---|---|---|---|
| `qwen3_30b_a3b_thinking` | Qwen3-30B-A3B-Thinking | Orchestrator Phase 1 | Frozen |
| `kimi_k2_5` | Kimi K2.5 | Orchestrator Phase 2 | Frozen |
| `qwen3_vl` | Qwen3-VL | Multimodal synthesis | Frozen |
| `siglip2` | SigLIP2 | Keyframe image-text retrieval embeddings | Frozen |
| `internvideo_next` | InternVideo-Next | Window and timestep video features | Frozen |
| `ivnext_lit_text_head` | IVNext-LiT-TextHead | Text alignment to InternVideo feature space | Trainable |
| `sam3` | SAM 3 | Prompted segmentation and tracklets | Frozen |
| `dinov3_vit_b16` | DINOv3 ViT-B/16 | Object/vehicle ReID features | Frozen |
| `person_reid_embedder` | Dedicated Person ReID | Cross-camera person identity | Frozen |
| `cv_state` | CV-State | Static-camera activity trigger | Rule-based |
| `bg_motion_trigger` | BG-Motion-Trigger | Moving-background activity trigger | Trainable lightweight |

### 7.1 Initialization Rule

- `ivnext_lit_text_head` should initialize from SigLIP2 text encoder/projection geometry where possible.

---

## 8. Data Stores and Resources

### 8.1 Datastores

| ID | Type | Engine | Purpose |
|---|---|---|---|
| `milvus` | Vector DB | Milvus | Frame and optional window embedding indices |
| `neo4j_or_falkordb` | Graph DB | Neo4j/FalkorDB | Event Knowledge Graph (EKG) |
| `feature_cache` | Tiered cache | In-memory/disk | Expensive video feature reuse |
| `evidence_asset_store` | Artifact store | FS/S3-compatible | Overlay assets and visual references |

### 8.2 Logical Resources

- `FrameIndex_SigLIP2`: keyframe embeddings + frame metadata.
- `WindowIndex_IVNext` (optional): pooled window embeddings for faster re-ranking.
- `L1 cache`: per-window pooled + per-timestep IVNext features.
- `L2 cache`: foreground token slices/grid for high-precision temporal grounding.
- `EKG`: entities, edges, timestamps, evidence pointers, confidence.

---

## 9. Stage-by-Stage Functional Spec

## Stage 1 (`activity_ingestion`) - 4-Path Activity Ingestion and Indexing

### Objective

Detect active windows using route-specific strategies and build keyframe retrieval index.

### Inputs

- Raw video stream
- Optional metadata stream (bboxes/motion vectors)
- Optional audio peaks

### Routes

1. **Meta-Sync:** use existing metadata; skip heavy triggering.
2. **Sig-Ex Adaptive:** base 1 FPS + burst sampling near motion/cut/audio peaks.
3. **CV-State:** RMSE/SSIM + debounce + illumination guard for static CCTV.
4. **BG-Motion-Trigger:** classify meaningful motion vs background noise.

### Outputs

- Active windows
- Keyframes + SigLIP2 embeddings in `FrameIndex_SigLIP2`

### Quality Gates

- Coverage of active windows above configured threshold.
- Duplicate/near-duplicate keyframe rate below threshold.

### Failure Handling

- Low route confidence -> parallel safety route (Sig-Ex Adaptive).

---

## Stage 2 (`temporal_retrieval`) - Coarse Temporal Retrieval and Re-ranking

### Objective

Retrieve candidate windows and calibrate confidence using InternVideo-space alignment.

### Inputs

- Text query
- `FrameIndex_SigLIP2`

### Processing

1. Text-to-frame candidate search in Milvus.
2. InternVideo-Next feature extraction on candidate windows.
3. Cache L1 features (`D` pooled + `T x D` timestep).
4. LiT-head scoring with pooled and timestep fusion.

### Outputs

- Top-K validated windows
- Candidate action time ranges
- Confidence scores

### Quality Gates

- Max validated confidence must cross minimum threshold.

### Failure Handling

- Query decomposition and retrieval retry.

---

## Stage 3 (`spatial_grounding`) - Spatial Grounding and Tracking

### Objective

Convert query concepts into spatio-temporal masklets and track IDs.

### Inputs

- Validated clip windows
- Query text or sub-queries

### Processing

1. Prompt SAM 3 with entity/action concepts.
2. Generate masks/tracklets and frame-level confidence.
3. Materialize overlay artifacts for auditability.

### Outputs

- Masklets
- Track IDs
- Per-frame mask confidence

### Quality Gates

- Median mask confidence above threshold for required entities.

### Failure Handling

- Retry with decomposed prompts.
- Fallback to detector+tracker if available.

---

## Stage 4 (`entity_resolution`) - Cross-Clip Entity Resolution (Split ReID)

### Objective

Resolve object and person identities across clips/cameras.

### Object/Vehicle Path

- DINOv3 embeddings on mask-weighted crops.
- Cluster by cosine + density clustering (DBSCAN/HDBSCAN family).
- Emit `ObjectClusterID`.

### Person Path

- PersonReID embeddings on mask-weighted person crops.
- Fuse with camera topology + travel-time constraints + temporal overlap.
- Emit `PersonEntityID` + confidence.

### Quality Gates

- Identity conflict rate must remain below threshold.

### Failure Handling

- Keep unresolved entities explicit; do not force linkage.

---

## Stage 5 (`temporal_localization`) - Fine Temporal Grounding

### Objective

Localize precise action boundaries (`t_start`, `t_end`) with confidence and ambiguity flags.

### Inputs

- L1 timestep features
- L2 foreground slices when available
- Tracklets and query/action text

### Processing

1. Foreground feature isolation (L2 preferred, fallback to track-aligned pooling).
2. Per-timestep similarity curve `S(t)` using LiT-aligned text embedding.
3. Smoothing + hysteresis segmentation to extract contiguous action segments.

### Outputs

- `t_start`, `t_end`
- Temporal confidence
- Failure flags (`occlusion`, `low_mask_confidence`, `multi_actor_ambiguity`)

### Quality Gates

- Segment stability across neighboring windows above threshold.

---

## Stage 6 (`graph_memory`) - Dynamic Memory Construction (EKG)

### Objective

Persist entities, actions, and evidence into graph memory with uncertainty.

### Node Types

- `ObjectClusterID`
- `PersonEntityID`
- `CameraID`
- `TrackID` (optional)
- `Zone` (optional)

### Edge Types

- `PersonEntityID -[EXITS]-> ObjectClusterID {t_start, t_end, confidence, camera_id, evidence_refs}`
- `PersonEntityID -[MOVES_TO]-> CameraID {time_range, confidence}`

### Evidence Attachments

- Clip IDs
- Frame ranges
- Overlay asset URIs
- Embedding IDs/model versions

### Quality Gates

- Required evidence refs present for all synthesizable claims.

---

## Stage 7 (`multimodal_synthesis`) - Multimodal Synthesis

### Objective

Generate grounded natural-language answer tied to evidence package.

### Inputs

- Query-scoped EKG slice
- Verified raw clips
- Claim-level evidence records (`camera_id`, `t_start`, `t_end`, overlays, entity IDs, confidence)

### Output Contract

- Human-readable answer
- Explicit time/camera references
- Entity continuity statements
- Optional evidence appendix

### Quality Gates

- 100% claims must be evidence-linked; unverifiable claims are redacted.

---

## 10. Orchestration and Agent Behavior

### Runtime Controller

- LangGraph state machine drives routing, retries, decomposition, and halting policies.

### Required State Keys

- `query_id`
- `normalized_query`
- `candidate_windows`
- `grounded_tracks`
- `entity_links`
- `temporal_segments`
- `evidence_package`

### Branching Rules

1. Low Stage 2 confidence -> decompose query and retry.
2. Stage 3 low mask confidence -> prompt decomposition retry and fallback stack.
3. Stage 4 ambiguity -> unresolved entity state (no forced merge).
4. Stage 7 missing evidence -> conservative answer generation.

---

## 11. Config Governance (Mandatory)

Although this spec is authored in Markdown, **runtime configuration must use YAML and strict schema validation**.

### 11.1 Required Stack

1. **YAML** for environment-agnostic configuration documents.
2. **OmegaConf** for hierarchical merge and interpolation.
3. **Pydantic (strict mode)** for end-to-end validation of merged configs.

### 11.2 Merge Policy

- Base config defines canonical defaults.
- Environment/tenant overrides are delta-only.
- Merge order: `base <- env <- runtime override`.
- Unknown keys fail validation.

### 11.3 DRY Rules

1. Define model/store IDs once in a registry block.
2. Reference IDs from stage specs; never copy model/store names inline.
3. Keep thresholds in one constants block.
4. Define stage ID to crisp stage-name mapping once in `stage_catalog` and reuse.
5. Keep repeated contracts (claim schema, evidence schema) as reusable templates.

### 11.4 Validation Rules

1. `extra="forbid"` on all Pydantic models.
2. Cross-reference validation: stage model IDs and resource IDs must exist.
3. Stage set completeness validation: all Stage 1..7 must be present exactly once.
4. Stage `stage_name` must match the canonical `stage_catalog` mapping.
5. Confidence thresholds constrained to `[0,1]`.
6. Hysteresis high threshold must be greater than low threshold.

### 11.5 Minimal Reference Structure

```yaml
meta:
  spec_id: agentic_video_rag
stage_catalog:
  stage_1: activity_ingestion
  stage_2: temporal_retrieval
  stage_3: spatial_grounding
  stage_4: entity_resolution
  stage_5: temporal_localization
  stage_6: graph_memory
  stage_7: multimodal_synthesis
registry:
  models: {...}
  datastores: {...}
  resources: {...}
constants: {...}
phase_a: {...}
phase_b:
  stages:
    - stage_id: stage_1
      stage_name: activity_ingestion
      ...
```

---

## 12. Data and Evidence Contracts

### 12.1 Claim Record Contract

Required fields for each synthesized claim:

- `claim_id`
- `entity_ids`
- `camera_id`
- `t_start`
- `t_end`
- `confidence`
- `evidence_refs`

### 12.2 Evidence Ref Contract

- `clip_id`
- `frame_range`
- `overlay_uri`
- `embedding_id`
- `model_version`

### 12.3 Reproducibility Contract

All persisted embeddings and events must include model/version fingerprints and processing profile IDs.

---

## 13. Reliability and Safety

1. Never output high-certainty language for low-confidence links.
2. Surface ambiguity states explicitly.
3. Keep retry budgets bounded.
4. Reject synthesis if evidence package is incomplete.

---

## 14. Performance and Cost Strategy

1. Triggered ingestion to avoid dense full-video processing.
2. L1/L2 feature cache to amortize repeated queries.
3. Optional window index for faster future re-ranking.
4. Query decomposition only when confidence gates fail.

---

## 15. Testing and Acceptance Criteria

### 15.1 Configuration Tests

- YAML merge tests for base + override precedence.
- Pydantic schema tests for strict key rejection.
- Cross-reference integrity tests for stage/model/resource IDs.

### 15.2 Pipeline Tests

- Stage contract tests: required inputs/outputs present per stage.
- Regression test for example query flow (red SUV -> person exit -> interior tracking).
- Graph integrity tests: edge evidence refs must be present.

### 15.3 Product Acceptance

A run is acceptable only if:

1. Final response includes camera/time grounding for each claim.
2. Entity continuity is supported by ReID confidence + topology/time constraints.
3. Unverifiable claims are omitted or marked uncertain.

---

## 16. Example Query Execution Blueprint

**Query:** "Find the red SUV, identify the person who got out, and track that specific person across the interior cameras."

Execution path:

1. Stage 2 finds red SUV candidate windows.
2. Stage 3 grounds SUV + emerging person tracklets.
3. Stage 5 localizes "person exits SUV" temporal segment.
4. Stage 4 links the person across cameras via PersonReID + topology/time fusion.
5. Stage 6 writes graph nodes/edges with evidence refs.
6. Stage 7 generates final grounded narrative with citations.

---

## 17. Implementation Guardrails

1. Keep stage logic modular and side-effect-light.
2. Separate orchestration policy from model inference code.
3. Keep all thresholds and model IDs in config, not hardcoded.
4. Any new model/store must update registry + schema before use.
5. Any new stage behavior must include quality gate and fallback definition.

---

## 18. Change Management

1. All spec changes update this file first.
2. Any config/schema changes must reference a spec section ID.
3. Breaking changes require spec version bump and migration notes.
4. Execution progress and milestone tracking must be maintained in `design/execution_plan.md`.

---

## 19. Execution Plan Reference

Execution phases, milestones, owners, dates, and status are maintained in:

- `design/execution_plan.md` (live tracking document)

Governance rule:

1. Update `design/execution_plan.md` whenever implementation status changes.
2. Keep this document (`design/spec_groundtruth.md`) focused on stable product/system contracts.

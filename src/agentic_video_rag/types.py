"""Typed runtime data contracts for the Agentic Video RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RouteId = Literal["meta_sync", "sig_ex_adaptive", "cv_state", "bg_motion_trigger"]
EntityType = Literal["object", "person"]


@dataclass(frozen=True)
class FrameObservation:
    """Single frame observation used by deterministic stage adapters."""

    timestamp: int
    objects: tuple[str, ...]
    actions: tuple[str, ...]
    background_motion: float


@dataclass(frozen=True)
class Clip:
    """Clip-level metadata and frame observations."""

    clip_id: str
    camera_id: str
    camera_type: Literal["static", "moving"]
    location: Literal["exterior", "interior"]
    duration_seconds: int
    frames: tuple[FrameObservation, ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class QueryRequest:
    """Input request for running the full pipeline."""

    query_id: str
    query_text: str
    clips: tuple[Clip, ...]
    camera_topology: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ActiveWindow:
    """Stage 1 output window selected for downstream processing."""

    window_id: str
    clip_id: str
    camera_id: str
    route_id: RouteId
    t_start: int
    t_end: int
    reason: str
    semantic_tokens: tuple[str, ...]


@dataclass(frozen=True)
class KeyframeRecord:
    """Frame index record stored in frame-vector index."""

    frame_id: str
    window_id: str
    clip_id: str
    camera_id: str
    timestamp: int
    embedding: tuple[float, ...]
    embedding_id: str
    semantic_tokens: tuple[str, ...]
    route_id: RouteId


@dataclass(frozen=True)
class WindowFeatures:
    """Window-level features derived from InternVideo-like adapter."""

    window_id: str
    clip_id: str
    camera_id: str
    t_start: int
    t_end: int
    pooled_embedding: tuple[float, ...]
    per_timestep_embeddings: tuple[tuple[float, ...], ...]
    semantic_tokens: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedWindow:
    """Stage 2 validated window candidate."""

    window_id: str
    clip_id: str
    camera_id: str
    t_start: int
    t_end: int
    confidence: float
    query_text: str


@dataclass(frozen=True)
class Tracklet:
    """Stage 3 grounded entity tracklet."""

    track_id: str
    clip_id: str
    camera_id: str
    window_id: str
    entity_type: EntityType
    label: str
    t_start: int
    t_end: int
    mask_confidence: float
    overlay_uri: str


@dataclass(frozen=True)
class EntityLink:
    """Stage 4 resolved (or unresolved) cross-clip entity identity."""

    entity_id: str
    entity_type: EntityType
    label: str
    track_ids: tuple[str, ...]
    confidence: float
    resolved: bool


@dataclass(frozen=True)
class TemporalSegment:
    """Stage 5 action-level temporal localization output."""

    segment_id: str
    clip_id: str
    camera_id: str
    track_id: str
    action: str
    t_start: int
    t_end: int
    confidence: float
    failure_flags: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceRef:
    """Evidence pointer attached to claims and graph edges."""

    clip_id: str
    camera_id: str
    frame_range: tuple[int, int]
    overlay_uri: str
    embedding_id: str
    model_version: str


@dataclass(frozen=True)
class GraphNode:
    """Graph memory node representation."""

    node_id: str
    node_type: str
    properties: dict[str, object]


@dataclass(frozen=True)
class GraphEdge:
    """Graph memory edge representation with evidence and uncertainty."""

    edge_id: str
    edge_type: Literal["EXITS", "MOVES_TO"]
    source_id: str
    target_id: str
    t_start: int
    t_end: int
    camera_id: str
    confidence: float
    evidence_refs: tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class ClaimRecord:
    """Stage 7 claim contract."""

    claim_id: str
    text: str
    entity_ids: tuple[str, ...]
    camera_id: str
    t_start: int
    t_end: int
    confidence: float
    evidence_refs: tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class SynthesisOutput:
    """Final grounded response bundle."""

    summary: str
    claims: tuple[ClaimRecord, ...]
    evidence_appendix: tuple[str, ...]
    redacted_claim_count: int


@dataclass
class StageMetrics:
    """Per-stage runtime metrics captured for profiling and tuning."""

    stage_durations_ms: dict[str, float] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class PipelineResult:
    """Complete pipeline output including intermediate artifacts."""

    query_id: str
    query_text: str
    active_windows: list[ActiveWindow]
    validated_windows: list[ValidatedWindow]
    tracklets: list[Tracklet]
    entity_links: list[EntityLink]
    temporal_segments: list[TemporalSegment]
    graph_nodes: list[GraphNode]
    graph_edges: list[GraphEdge]
    synthesis: SynthesisOutput
    metrics: StageMetrics

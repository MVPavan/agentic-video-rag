"""In-memory datastore abstractions for deterministic pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import EvidenceRef, GraphEdge, GraphNode, KeyframeRecord, WindowFeatures
from .utils import cosine_similarity


@dataclass
class FrameIndexStore:
    """Milvus-like frame embedding index abstraction."""

    records: list[KeyframeRecord] = field(default_factory=list)

    def add(self, record: KeyframeRecord) -> None:
        self.records.append(record)

    def search(self, query_embedding: tuple[float, ...], top_k: int) -> list[tuple[float, KeyframeRecord]]:
        ranked = [
            (cosine_similarity(query_embedding, record.embedding), record)
            for record in self.records
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:top_k]


@dataclass
class WindowIndexStore:
    """Optional pooled-window vector index."""

    vectors: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def upsert(self, window_id: str, vector: tuple[float, ...]) -> None:
        self.vectors[window_id] = vector


@dataclass
class FeatureCacheStore:
    """Tiered cache for L1 and L2 feature payloads."""

    l1: dict[str, WindowFeatures] = field(default_factory=dict)
    l2: dict[str, tuple[str, ...]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get_l1(self, key: str) -> WindowFeatures | None:
        item = self.l1.get(key)
        if item is None:
            self.misses += 1
            return None
        self.hits += 1
        return item

    def set_l1(self, key: str, value: WindowFeatures) -> None:
        self.l1[key] = value

    def get_l2(self, key: str) -> tuple[str, ...] | None:
        item = self.l2.get(key)
        if item is None:
            self.misses += 1
            return None
        self.hits += 1
        return item

    def set_l2(self, key: str, value: tuple[str, ...]) -> None:
        self.l2[key] = value


@dataclass
class ArtifactStore:
    """Artifact repository for overlays and synthetic visual evidence."""

    artifacts: dict[str, str] = field(default_factory=dict)

    def put(self, uri: str, payload: str) -> None:
        self.artifacts[uri] = payload


@dataclass
class GraphStore:
    """Graph memory store for nodes, edges, and query-scoped evidence retrieval."""

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: dict[str, GraphEdge] = field(default_factory=dict)

    def upsert_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges[edge.edge_id] = edge

    def edges_with_evidence(self) -> list[GraphEdge]:
        return [edge for edge in self.edges.values() if edge.evidence_refs]


@dataclass
class EvidenceRegistry:
    """Maps track/segment identifiers to evidence references."""

    by_track_id: dict[str, list[EvidenceRef]] = field(default_factory=dict)

    def append(self, track_id: str, evidence_ref: EvidenceRef) -> None:
        self.by_track_id.setdefault(track_id, []).append(evidence_ref)

    def get(self, track_id: str) -> tuple[EvidenceRef, ...]:
        return tuple(self.by_track_id.get(track_id, []))


@dataclass
class PipelineStores:
    """Top-level store container used by the pipeline engine."""

    frame_index: FrameIndexStore = field(default_factory=FrameIndexStore)
    window_index: WindowIndexStore = field(default_factory=WindowIndexStore)
    feature_cache: FeatureCacheStore = field(default_factory=FeatureCacheStore)
    artifact_store: ArtifactStore = field(default_factory=ArtifactStore)
    graph_store: GraphStore = field(default_factory=GraphStore)
    evidence_registry: EvidenceRegistry = field(default_factory=EvidenceRegistry)

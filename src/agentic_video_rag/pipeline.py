"""Executable 7-stage deterministic Agentic Video RAG pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .adapters import (
    IVNextLiTTextHeadAdapter,
    InternVideoNextAdapter,
    MultimodalSynthesizerAdapter,
    QueryDecomposer,
    ReIDResolver,
    SAM3Adapter,
    SigLIP2Adapter,
    TemporalGroundingAdapter,
    TriggerRouter,
)
from .orchestrator_contract import OrchestratorContract
from .spec_schema import RuntimeConfig, StageId
from .stores import PipelineStores
from .types import (
    ActiveWindow,
    ClaimRecord,
    Clip,
    EntityLink,
    EvidenceRef,
    GraphEdge,
    GraphNode,
    KeyframeRecord,
    PipelineResult,
    QueryRequest,
    StageMetrics,
    TemporalSegment,
    Tracklet,
    ValidatedWindow,
)
from .utils import cosine_similarity, mean, overlap_score, stable_id, tokenize


@dataclass
class OrchestrationState:
    """Mutable orchestration state used through stage execution."""

    values: dict[str, object]


class AgenticVideoRAGEngine:
    """Deterministic implementation of all seven stages and orchestration flow."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.stores = PipelineStores()
        self.contract = OrchestratorContract.from_runtime_config(config)

        self.trigger_router = TriggerRouter()
        self.decomposer = QueryDecomposer()

        self.siglip = SigLIP2Adapter()
        self.internvideo = InternVideoNextAdapter()
        self.lit_head = IVNextLiTTextHeadAdapter()
        self.sam3 = SAM3Adapter()
        self.reid = ReIDResolver()
        self.temporal = TemporalGroundingAdapter()
        self.synthesizer = MultimodalSynthesizerAdapter()

    def run(self, request: QueryRequest) -> PipelineResult:
        clips_by_id = {clip.clip_id: clip for clip in request.clips}
        metrics = StageMetrics()
        state = self._new_state(request)

        active_windows = self._run_stage(
            metrics,
            "stage_1",
            lambda: self._stage_1_activity_ingestion(request),
        )
        state.values["candidate_windows"] = active_windows

        validated_windows = self._run_stage(
            metrics,
            "stage_2",
            lambda: self._stage_2_temporal_retrieval(request, active_windows),
        )
        state.values["candidate_windows"] = validated_windows

        tracklets = self._run_stage(
            metrics,
            "stage_3",
            lambda: self._stage_3_spatial_grounding(request, validated_windows, clips_by_id),
        )
        state.values["grounded_tracks"] = tracklets

        entity_links = self._run_stage(
            metrics,
            "stage_4",
            lambda: self._stage_4_entity_resolution(request, tracklets),
        )
        state.values["entity_links"] = entity_links

        temporal_segments = self._run_stage(
            metrics,
            "stage_5",
            lambda: self._stage_5_temporal_localization(
                request=request,
                tracklets=tracklets,
                clips_by_id=clips_by_id,
            ),
        )
        state.values["temporal_segments"] = temporal_segments

        graph_nodes, graph_edges = self._run_stage(
            metrics,
            "stage_6",
            lambda: self._stage_6_graph_memory(
                request=request,
                tracklets=tracklets,
                entity_links=entity_links,
                temporal_segments=temporal_segments,
            ),
        )

        synthesis = self._run_stage(
            metrics,
            "stage_7",
            lambda: self._stage_7_multimodal_synthesis(
                query_text=request.query_text,
                graph_edges=graph_edges,
            ),
        )
        state.values["evidence_package"] = synthesis.evidence_appendix

        self.contract.validate_state_snapshot(state.values)

        metrics.cache_hits = self.stores.feature_cache.hits
        metrics.cache_misses = self.stores.feature_cache.misses

        return PipelineResult(
            query_id=request.query_id,
            query_text=request.query_text,
            active_windows=active_windows,
            validated_windows=validated_windows,
            tracklets=tracklets,
            entity_links=entity_links,
            temporal_segments=temporal_segments,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            synthesis=synthesis,
            metrics=metrics,
        )

    def _new_state(self, request: QueryRequest) -> OrchestrationState:
        default = {
            "query_id": request.query_id,
            "normalized_query": request.query_text.lower().strip(),
            "candidate_windows": [],
            "grounded_tracks": [],
            "entity_links": [],
            "temporal_segments": [],
            "evidence_package": [],
        }
        return OrchestrationState(values=default)

    def _run_stage(self, metrics: StageMetrics, stage_id: StageId, runner):
        stage_key = self.config.stage_catalog.as_map()[stage_id]
        start = time.perf_counter()
        result = runner()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics.stage_durations_ms[stage_key] = round(elapsed_ms, 3)
        return result

    def _stage_1_activity_ingestion(self, request: QueryRequest) -> list[ActiveWindow]:
        active_windows: list[ActiveWindow] = []

        for clip in request.clips:
            route_id = self.trigger_router.choose_route(clip)
            window_spans = self.trigger_router.extract_active_windows(clip, route_id)

            for idx, (t_start, t_end, reason) in enumerate(window_spans):
                frames = [
                    frame
                    for frame in clip.frames
                    if t_start <= frame.timestamp <= t_end
                ]
                semantic_tokens = tuple(
                    sorted(
                        {
                            token
                            for frame in frames
                            for token in tokenize(" ".join([*frame.objects, *frame.actions]))
                        }
                    )
                )
                window_id = stable_id("WIN", clip.clip_id, route_id, idx, t_start, t_end)
                window = ActiveWindow(
                    window_id=window_id,
                    clip_id=clip.clip_id,
                    camera_id=clip.camera_id,
                    route_id=route_id,
                    t_start=t_start,
                    t_end=t_end,
                    reason=reason,
                    semantic_tokens=semantic_tokens,
                )
                active_windows.append(window)

                for frame in frames:
                    frame_id = stable_id("FRAME", clip.clip_id, frame.timestamp)
                    embedding = self.siglip.embed_frame(
                        clip_id=clip.clip_id,
                        timestamp=frame.timestamp,
                        semantic_tokens=semantic_tokens,
                    )
                    embedding_id = stable_id("EMB", frame_id, "siglip2")
                    self.stores.frame_index.add(
                        KeyframeRecord(
                            frame_id=frame_id,
                            window_id=window_id,
                            clip_id=clip.clip_id,
                            camera_id=clip.camera_id,
                            timestamp=frame.timestamp,
                            embedding=embedding,
                            embedding_id=embedding_id,
                            semantic_tokens=semantic_tokens,
                            route_id=route_id,
                        )
                    )

        return active_windows

    def _stage_2_temporal_retrieval(
        self,
        request: QueryRequest,
        active_windows: list[ActiveWindow],
    ) -> list[ValidatedWindow]:
        query_embedding = self.siglip.embed_text(request.query_text)
        query_lit_embedding = self.lit_head.embed_text(request.query_text)
        query_tokens = tokenize(request.query_text)

        top_k_initial = self.config.constants.retrieval.stage2_initial_top_k_windows
        frame_hits = self.stores.frame_index.search(query_embedding, top_k=top_k_initial)

        windows_by_id = {window.window_id: window for window in active_windows}
        candidate_ids = [record.window_id for _, record in frame_hits if record.window_id in windows_by_id]

        scored: list[ValidatedWindow] = self._score_windows(
            candidate_ids=candidate_ids,
            windows_by_id=windows_by_id,
            request=request,
            query_text=request.query_text,
            query_tokens=query_tokens,
            query_lit_embedding=query_lit_embedding,
        )

        threshold = self.config.constants.retrieval.stage2_min_validation_confidence
        validated = [item for item in scored if item.confidence >= threshold]

        for subquery in self.decomposer.decompose(request.query_text)[1:]:
            sub_tokens = tokenize(subquery)
            sub_embed = self.lit_head.embed_text(subquery)
            rescored = self._score_windows(
                candidate_ids=candidate_ids,
                windows_by_id=windows_by_id,
                request=request,
                query_text=subquery,
                query_tokens=sub_tokens,
                query_lit_embedding=sub_embed,
            )
            validated.extend(
                item for item in rescored if item.confidence >= max(0.35, threshold - 0.15)
            )
            if rescored:
                # Keep the best decomposed candidate for recall robustness.
                validated.append(rescored[0])

        if not validated and scored:
            best_by_clip: dict[str, ValidatedWindow] = {}
            for item in scored:
                if item.clip_id not in best_by_clip:
                    best_by_clip[item.clip_id] = item
            validated.extend(best_by_clip.values())

        unique: dict[str, ValidatedWindow] = {}
        for item in validated:
            existing = unique.get(item.window_id)
            if existing is None or item.confidence > existing.confidence:
                unique[item.window_id] = item

        top_k_validated = self.config.constants.retrieval.stage2_validated_top_k_windows
        ranked = sorted(unique.values(), key=lambda item: item.confidence, reverse=True)
        # Preserve clip diversity before filling remaining slots by confidence.
        selected: list[ValidatedWindow] = []
        seen_clips: set[str] = set()
        for item in ranked:
            if item.clip_id in seen_clips:
                continue
            selected.append(item)
            seen_clips.add(item.clip_id)
            if len(selected) >= top_k_validated:
                break

        for item in ranked:
            if len(selected) >= top_k_validated:
                break
            if item.window_id in {candidate.window_id for candidate in selected}:
                continue
            selected.append(item)

        return selected[:top_k_validated]

    def _score_windows(
        self,
        candidate_ids: list[str],
        windows_by_id: dict[str, ActiveWindow],
        request: QueryRequest,
        query_text: str,
        query_tokens: list[str],
        query_lit_embedding: tuple[float, ...],
    ) -> list[ValidatedWindow]:
        clip_map = {clip.clip_id: clip for clip in request.clips}
        scores: list[ValidatedWindow] = []

        for window_id in dict.fromkeys(candidate_ids):
            window = windows_by_id.get(window_id)
            if window is None:
                continue

            cache_key = f"l1:{window.window_id}"
            cached_features = self.stores.feature_cache.get_l1(cache_key)
            if cached_features is None:
                clip = clip_map[window.clip_id]
                features = self.internvideo.extract_window_features(window, clip)
                self.stores.feature_cache.set_l1(cache_key, features)
            else:
                features = cached_features

            self.stores.window_index.upsert(window.window_id, features.pooled_embedding)

            pooled_sim = (cosine_similarity(query_lit_embedding, features.pooled_embedding) + 1.0) / 2.0
            step_sims = [
                (cosine_similarity(query_lit_embedding, step) + 1.0) / 2.0
                for step in features.per_timestep_embeddings
            ]
            step_sim = max(step_sims) if step_sims else 0.0
            token_sim = overlap_score(query_tokens, list(features.semantic_tokens))
            confidence = round((0.45 * pooled_sim) + (0.35 * step_sim) + (0.20 * token_sim), 3)

            scores.append(
                ValidatedWindow(
                    window_id=window.window_id,
                    clip_id=window.clip_id,
                    camera_id=window.camera_id,
                    t_start=window.t_start,
                    t_end=window.t_end,
                    confidence=confidence,
                    query_text=query_text,
                )
            )

        scores.sort(key=lambda item: item.confidence, reverse=True)
        return scores

    def _stage_3_spatial_grounding(
        self,
        request: QueryRequest,
        validated_windows: list[ValidatedWindow],
        clips_by_id: dict[str, Clip],
    ) -> list[Tracklet]:
        threshold = self.config.constants.grounding.sam3_min_mask_confidence
        max_retries = self.config.constants.grounding.sam3_retry_max_attempts
        all_tracklets: list[Tracklet] = []

        for window in validated_windows:
            clip = clips_by_id[window.clip_id]
            best_tracklets: list[Tracklet] = []
            best_score = -1.0

            variants = self.decomposer.decompose(request.query_text)
            for attempt in range(max_retries + 1):
                query_variant = variants[min(attempt, len(variants) - 1)]
                candidate_tracklets = self.sam3.ground_window(
                    validated_window=window,
                    clip=clip,
                    query_variant=query_variant,
                    artifact_store=self.stores.artifact_store,
                )

                if candidate_tracklets:
                    median_conf = mean([track.mask_confidence for track in candidate_tracklets])
                    if median_conf > best_score:
                        best_score = median_conf
                        best_tracklets = candidate_tracklets
                    if median_conf >= threshold:
                        break

            if best_score < threshold:
                fallback_tracklets = self._detector_tracker_fallback(window=window, clip=clip)
                if fallback_tracklets:
                    best_tracklets = fallback_tracklets

            for tracklet in best_tracklets:
                if tracklet.mask_confidence >= threshold:
                    self.stores.feature_cache.set_l2(
                        key=f"l2:{tracklet.track_id}",
                        value=tuple(tokenize(tracklet.label)),
                    )
                all_tracklets.append(tracklet)

        return all_tracklets

    def _detector_tracker_fallback(self, window: ValidatedWindow, clip: Clip) -> list[Tracklet]:
        frames = [
            frame
            for frame in clip.frames
            if window.t_start <= frame.timestamp <= window.t_end
        ]

        labels = sorted(
            {
                obj
                for frame in frames
                for obj in frame.objects
                if obj.lower().startswith("person") or "suv" in obj.lower() or "car" in obj.lower()
            }
        )
        fallback: list[Tracklet] = []
        for label in labels[:2]:
            entity_type = "person" if label.lower().startswith("person") else "object"
            track_id = stable_id("FALLBACK_TRACK", clip.clip_id, window.window_id, label)
            overlay_uri = f"overlay://{clip.clip_id}/{track_id}.json"
            self.stores.artifact_store.put(overlay_uri, f"fallback_overlay:{label}")
            fallback.append(
                Tracklet(
                    track_id=track_id,
                    clip_id=clip.clip_id,
                    camera_id=clip.camera_id,
                    window_id=window.window_id,
                    entity_type=entity_type,  # type: ignore[arg-type]
                    label=label,
                    t_start=window.t_start,
                    t_end=window.t_end,
                    mask_confidence=0.51,
                    overlay_uri=overlay_uri,
                )
            )
        return fallback

    def _stage_4_entity_resolution(
        self,
        request: QueryRequest,
        tracklets: list[Tracklet],
    ) -> list[EntityLink]:
        return self.reid.resolve(
            tracklets=tracklets,
            camera_topology=request.camera_topology,
            max_cross_camera_travel_seconds=self.config.constants.reid.max_cross_camera_travel_seconds,
        )

    def _stage_5_temporal_localization(
        self,
        request: QueryRequest,
        tracklets: list[Tracklet],
        clips_by_id: dict[str, Clip],
    ) -> list[TemporalSegment]:
        person_tracklets = [track for track in tracklets if track.entity_type == "person"]
        candidate_tracklets = person_tracklets if person_tracklets else tracklets

        segments: list[TemporalSegment] = []
        for tracklet in candidate_tracklets:
            clip = clips_by_id[tracklet.clip_id]
            segment = self.temporal.localize(
                tracklet=tracklet,
                clip=clip,
                query_text=request.query_text,
                smoothing_window_size=self.config.constants.temporal_localization.smoothing_window_size,
                hysteresis_high=self.config.constants.temporal_localization.hysteresis_high,
                hysteresis_low=self.config.constants.temporal_localization.hysteresis_low,
                mask_confidence_floor=self.config.constants.grounding.sam3_min_mask_confidence,
            )
            segments.append(segment)

        # Surface ambiguity if multiple person tracks overlap in same clip.
        for idx, segment in enumerate(segments):
            overlaps = [
                other
                for other in segments
                if other.segment_id != segment.segment_id
                and other.clip_id == segment.clip_id
                and not (other.t_end < segment.t_start or other.t_start > segment.t_end)
            ]
            if overlaps:
                flags = tuple(sorted(set([*segment.failure_flags, "multi_actor_ambiguity"])))
                segments[idx] = TemporalSegment(
                    segment_id=segment.segment_id,
                    clip_id=segment.clip_id,
                    camera_id=segment.camera_id,
                    track_id=segment.track_id,
                    action=segment.action,
                    t_start=segment.t_start,
                    t_end=segment.t_end,
                    confidence=segment.confidence,
                    failure_flags=flags,
                )

        return segments

    def _stage_6_graph_memory(
        self,
        request: QueryRequest,
        tracklets: list[Tracklet],
        entity_links: list[EntityLink],
        temporal_segments: list[TemporalSegment],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        track_by_id = {track.track_id: track for track in tracklets}
        entity_by_track: dict[str, EntityLink] = {}

        for link in entity_links:
            node = GraphNode(
                node_id=link.entity_id,
                node_type="PersonEntityID" if link.entity_type == "person" else "ObjectClusterID",
                properties={
                    "label": link.label,
                    "confidence": link.confidence,
                    "resolved": link.resolved,
                },
            )
            self.stores.graph_store.upsert_node(node)
            for track_id in link.track_ids:
                entity_by_track[track_id] = link

        for track in tracklets:
            self.stores.graph_store.upsert_node(
                GraphNode(
                    node_id=track.track_id,
                    node_type="TrackID",
                    properties={
                        "clip_id": track.clip_id,
                        "camera_id": track.camera_id,
                        "label": track.label,
                    },
                )
            )
            self.stores.graph_store.upsert_node(
                GraphNode(
                    node_id=track.camera_id,
                    node_type="CameraID",
                    properties={"camera_id": track.camera_id},
                )
            )

        edges: list[GraphEdge] = []

        # Build EXITS edges from temporal segments where person and object co-exist.
        for segment in temporal_segments:
            person_link = entity_by_track.get(segment.track_id)
            if person_link is None or person_link.entity_type != "person":
                continue

            same_window_object_entities = [
                entity_by_track.get(track.track_id)
                for track in tracklets
                if track.clip_id == segment.clip_id
                and track.window_id == track_by_id[segment.track_id].window_id
                and track.entity_type == "object"
            ]
            object_link = next(
                (link for link in same_window_object_entities if link is not None and link.entity_type == "object"),
                None,
            )
            if object_link is None:
                continue

            evidence_refs = self._build_evidence_refs_for_track(
                request=request,
                track=track_by_id[segment.track_id],
                segment=segment,
            )

            edge = GraphEdge(
                edge_id=stable_id("EDGE", "EXITS", person_link.entity_id, object_link.entity_id, segment.segment_id),
                edge_type="EXITS",
                source_id=person_link.entity_id,
                target_id=object_link.entity_id,
                t_start=segment.t_start,
                t_end=segment.t_end,
                camera_id=segment.camera_id,
                confidence=segment.confidence,
                evidence_refs=evidence_refs,
            )
            self.stores.graph_store.add_edge(edge)
            edges.append(edge)

        # Build MOVES_TO edges for person entities across cameras.
        for link in entity_links:
            if link.entity_type != "person":
                continue
            linked_tracks = [track_by_id[track_id] for track_id in link.track_ids if track_id in track_by_id]
            linked_tracks.sort(key=lambda item: (item.t_start, item.camera_id))
            for prev, curr in zip(linked_tracks, linked_tracks[1:]):
                evidence_refs = self._build_evidence_refs_for_track(
                    request=request,
                    track=curr,
                    segment=TemporalSegment(
                        segment_id=stable_id("TMPSEG", curr.track_id),
                        clip_id=curr.clip_id,
                        camera_id=curr.camera_id,
                        track_id=curr.track_id,
                        action="moves_to_camera",
                        t_start=curr.t_start,
                        t_end=curr.t_end,
                        confidence=0.8,
                        failure_flags=tuple(),
                    ),
                )

                edge = GraphEdge(
                    edge_id=stable_id("EDGE", "MOVES_TO", link.entity_id, curr.camera_id, curr.track_id),
                    edge_type="MOVES_TO",
                    source_id=link.entity_id,
                    target_id=curr.camera_id,
                    t_start=curr.t_start,
                    t_end=curr.t_end,
                    camera_id=curr.camera_id,
                    confidence=0.8 if link.resolved else 0.45,
                    evidence_refs=evidence_refs,
                )
                self.stores.graph_store.add_edge(edge)
                edges.append(edge)

        return list(self.stores.graph_store.nodes.values()), edges

    def _build_evidence_refs_for_track(
        self,
        request: QueryRequest,
        track: Tracklet,
        segment: TemporalSegment,
    ) -> tuple[EvidenceRef, ...]:
        # Pick one embedding ID from the frame index for reproducibility.
        embedding_id = next(
            (
                record.embedding_id
                for record in self.stores.frame_index.records
                if record.clip_id == track.clip_id and segment.t_start <= record.timestamp <= segment.t_end
            ),
            stable_id("EMB", track.track_id, "fallback"),
        )

        evidence = EvidenceRef(
            clip_id=track.clip_id,
            camera_id=track.camera_id,
            frame_range=(segment.t_start, segment.t_end),
            overlay_uri=track.overlay_uri,
            embedding_id=embedding_id,
            model_version=self.config.meta.spec_version,
        )
        self.stores.evidence_registry.append(track.track_id, evidence)
        return self.stores.evidence_registry.get(track.track_id)

    def _stage_7_multimodal_synthesis(
        self,
        query_text: str,
        graph_edges: list[GraphEdge],
    ):
        claims: list[ClaimRecord] = []
        for edge in graph_edges:
            if edge.edge_type == "EXITS":
                text = (
                    f"Person entity {edge.source_id} exited object entity {edge.target_id} "
                    f"at camera {edge.camera_id} between {edge.t_start}s and {edge.t_end}s."
                )
                entity_ids = (edge.source_id, edge.target_id)
            else:
                text = (
                    f"Person entity {edge.source_id} moved to camera {edge.target_id} "
                    f"during {edge.t_start}s to {edge.t_end}s."
                )
                entity_ids = (edge.source_id, edge.target_id)

            claims.append(
                ClaimRecord(
                    claim_id=stable_id("CLAIM", edge.edge_id),
                    text=text,
                    entity_ids=entity_ids,
                    camera_id=edge.camera_id,
                    t_start=edge.t_start,
                    t_end=edge.t_end,
                    confidence=edge.confidence,
                    evidence_refs=edge.evidence_refs,
                )
            )

        return self.synthesizer.synthesize(query_text=query_text, claims=claims)

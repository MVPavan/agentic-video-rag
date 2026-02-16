"""Deterministic adapter implementations for all pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

from .stores import ArtifactStore
from .types import (
    ActiveWindow,
    ClaimRecord,
    Clip,
    EntityLink,
    FrameObservation,
    RouteId,
    SynthesisOutput,
    TemporalSegment,
    Tracklet,
    ValidatedWindow,
    WindowFeatures,
)
from .utils import (
    deterministic_vector,
    mean,
    overlap_score,
    smooth_curve,
    stable_id,
    tokenize,
)


def _contiguous_spans(timestamps: list[int]) -> list[tuple[int, int]]:
    """Build contiguous integer spans from sorted timestamps."""

    if not timestamps:
        return []

    sorted_ts = sorted(set(timestamps))
    spans: list[tuple[int, int]] = []
    start = sorted_ts[0]
    end = sorted_ts[0]

    for value in sorted_ts[1:]:
        if value == end + 1:
            end = value
            continue
        spans.append((start, end))
        start = value
        end = value

    spans.append((start, end))
    return spans


@dataclass
class QueryDecomposer:
    """Simple deterministic query decomposition helper."""

    def decompose(self, query_text: str) -> list[str]:
        normalized = query_text.replace(",", " and ")
        parts = [part.strip() for part in normalized.split(" and ") if part.strip()]
        if not parts:
            return [query_text]

        decomposed = [query_text]
        for part in parts:
            if part.lower() == query_text.lower():
                continue
            decomposed.append(part)
        return decomposed


@dataclass
class TriggerRouter:
    """Route selector for Stage 1 ingestion paths."""

    def choose_route(self, clip: Clip) -> RouteId:
        if bool(clip.metadata.get("has_motion_vectors")):
            return "meta_sync"
        if clip.camera_type == "moving":
            return "sig_ex_adaptive"

        avg_bg = mean([frame.background_motion for frame in clip.frames])
        if avg_bg > 0.55:
            return "bg_motion_trigger"
        return "cv_state"

    def extract_active_windows(self, clip: Clip, route_id: RouteId) -> list[tuple[int, int, str]]:
        if route_id == "meta_sync":
            predefined = clip.metadata.get("active_windows")
            if isinstance(predefined, list) and predefined:
                windows: list[tuple[int, int, str]] = []
                for raw in predefined:
                    if not isinstance(raw, dict):
                        continue
                    start = int(raw.get("t_start", 0))
                    end = int(raw.get("t_end", start))
                    windows.append((start, end, "metadata_window"))
                if windows:
                    return windows

        active_timestamps: list[int] = []
        for frame in clip.frames:
            has_semantic_signal = bool(frame.actions) or any(
                token in " ".join(frame.objects).lower()
                for token in ("suv", "person", "vehicle", "car", "truck")
            )

            if route_id == "bg_motion_trigger":
                meaningful = has_semantic_signal and frame.background_motion > 0.4
            elif route_id == "cv_state":
                meaningful = has_semantic_signal
            else:
                meaningful = has_semantic_signal

            if meaningful:
                active_timestamps.append(frame.timestamp)

        windows = _contiguous_spans(active_timestamps)
        return [(start, end, "activity_detected") for start, end in windows]


@dataclass
class SigLIP2Adapter:
    """Deterministic frame/text embedding adapter."""

    embedding_dim: int = 16

    def embed_text(self, text: str) -> tuple[float, ...]:
        return deterministic_vector(f"siglip2:text:{text}", dim=self.embedding_dim)

    def embed_frame(self, clip_id: str, timestamp: int, semantic_tokens: tuple[str, ...]) -> tuple[float, ...]:
        seed = f"siglip2:frame:{clip_id}:{timestamp}:{'|'.join(sorted(semantic_tokens))}"
        return deterministic_vector(seed, dim=self.embedding_dim)


@dataclass
class InternVideoNextAdapter:
    """Deterministic InternVideo-like window feature extractor."""

    embedding_dim: int = 16

    def extract_window_features(self, window: ActiveWindow, clip: Clip) -> WindowFeatures:
        frames = [
            frame
            for frame in clip.frames
            if window.t_start <= frame.timestamp <= window.t_end
        ]

        if not frames:
            frames = [
                FrameObservation(
                    timestamp=window.t_start,
                    objects=tuple(),
                    actions=tuple(),
                    background_motion=0.0,
                )
            ]

        semantic_tokens: list[str] = []
        per_timestep_embeddings: list[tuple[float, ...]] = []

        for frame in frames:
            frame_tokens = [*frame.objects, *frame.actions, clip.camera_id, clip.location]
            semantic_tokens.extend(tokenize(" ".join(frame_tokens)))
            seed = f"ivnext:frame:{clip.clip_id}:{frame.timestamp}:{'|'.join(frame_tokens)}"
            per_timestep_embeddings.append(deterministic_vector(seed, dim=self.embedding_dim))

        pooled_seed = f"ivnext:pooled:{clip.clip_id}:{window.t_start}:{window.t_end}:{'|'.join(sorted(set(semantic_tokens)))}"
        pooled_embedding = deterministic_vector(pooled_seed, dim=self.embedding_dim)

        return WindowFeatures(
            window_id=window.window_id,
            clip_id=window.clip_id,
            camera_id=window.camera_id,
            t_start=window.t_start,
            t_end=window.t_end,
            pooled_embedding=pooled_embedding,
            per_timestep_embeddings=tuple(per_timestep_embeddings),
            semantic_tokens=tuple(sorted(set(semantic_tokens))),
        )


@dataclass
class IVNextLiTTextHeadAdapter:
    """Deterministic text encoder aligned to InternVideo feature space."""

    embedding_dim: int = 16

    def embed_text(self, text: str) -> tuple[float, ...]:
        return deterministic_vector(f"lithead:text:{text}", dim=self.embedding_dim)


@dataclass
class SAM3Adapter:
    """Deterministic SAM3-like spatial grounding adapter."""

    def ground_window(
        self,
        validated_window: ValidatedWindow,
        clip: Clip,
        query_variant: str,
        artifact_store: ArtifactStore,
    ) -> list[Tracklet]:
        query_tokens = set(tokenize(query_variant))
        frames = [
            frame
            for frame in clip.frames
            if validated_window.t_start <= frame.timestamp <= validated_window.t_end
        ]

        object_labels = sorted(
            {
                obj
                for frame in frames
                for obj in frame.objects
                if any(token in obj.lower() for token in ("suv", "car", "truck", "vehicle"))
            }
        )
        person_labels = sorted(
            {
                obj
                for frame in frames
                for obj in frame.objects
                if obj.lower().startswith("person")
            }
        )

        tracklets: list[Tracklet] = []

        if "suv" in query_tokens or "vehicle" in query_tokens or "car" in query_tokens:
            for label in object_labels:
                confidence = 0.88 if any(token in label.lower() for token in query_tokens) else 0.62
                tracklets.append(
                    self._new_tracklet(
                        clip=clip,
                        window=validated_window,
                        entity_type="object",
                        label=label,
                        confidence=confidence,
                        artifact_store=artifact_store,
                    )
                )

        if "person" in query_tokens or "who" in query_tokens:
            for label in person_labels:
                confidence = 0.86
                tracklets.append(
                    self._new_tracklet(
                        clip=clip,
                        window=validated_window,
                        entity_type="person",
                        label=label,
                        confidence=confidence,
                        artifact_store=artifact_store,
                    )
                )

        # If the query is broad and we have detections, expose lower-confidence generic tracks.
        if not tracklets:
            for label in object_labels[:1]:
                tracklets.append(
                    self._new_tracklet(
                        clip=clip,
                        window=validated_window,
                        entity_type="object",
                        label=label,
                        confidence=0.42,
                        artifact_store=artifact_store,
                    )
                )

        return tracklets

    def _new_tracklet(
        self,
        clip: Clip,
        window: ValidatedWindow,
        entity_type: str,
        label: str,
        confidence: float,
        artifact_store: ArtifactStore,
    ) -> Tracklet:
        track_id = stable_id("TRACK", clip.clip_id, window.window_id, entity_type, label)
        overlay_uri = f"overlay://{clip.clip_id}/{track_id}.json"
        artifact_store.put(overlay_uri, f"mask_bbox_overlay:{label}:{window.t_start}-{window.t_end}")

        return Tracklet(
            track_id=track_id,
            clip_id=clip.clip_id,
            camera_id=clip.camera_id,
            window_id=window.window_id,
            entity_type=entity_type,  # type: ignore[arg-type]
            label=label,
            t_start=window.t_start,
            t_end=window.t_end,
            mask_confidence=round(min(max(confidence, 0.0), 1.0), 3),
            overlay_uri=overlay_uri,
        )


@dataclass
class ReIDResolver:
    """Deterministic object/person entity linking with topology checks."""

    def resolve(
        self,
        tracklets: list[Tracklet],
        camera_topology: dict[str, tuple[str, ...]],
        max_cross_camera_travel_seconds: int,
    ) -> list[EntityLink]:
        object_links = self._resolve_object_links(tracklets)
        person_links = self._resolve_person_links(
            tracklets=tracklets,
            camera_topology=camera_topology,
            max_cross_camera_travel_seconds=max_cross_camera_travel_seconds,
        )
        return [*object_links, *person_links]

    def _resolve_object_links(self, tracklets: list[Tracklet]) -> list[EntityLink]:
        grouped: dict[str, list[Tracklet]] = {}
        for tracklet in tracklets:
            if tracklet.entity_type != "object":
                continue
            grouped.setdefault(tracklet.label.lower(), []).append(tracklet)

        links: list[EntityLink] = []
        for label, items in grouped.items():
            entity_id = stable_id("OBJ", label)
            links.append(
                EntityLink(
                    entity_id=entity_id,
                    entity_type="object",
                    label=label,
                    track_ids=tuple(sorted(item.track_id for item in items)),
                    confidence=0.83,
                    resolved=True,
                )
            )
        return links

    def _resolve_person_links(
        self,
        tracklets: list[Tracklet],
        camera_topology: dict[str, tuple[str, ...]],
        max_cross_camera_travel_seconds: int,
    ) -> list[EntityLink]:
        grouped: dict[str, list[Tracklet]] = {}
        for tracklet in tracklets:
            if tracklet.entity_type != "person":
                continue
            grouped.setdefault(tracklet.label.lower(), []).append(tracklet)

        links: list[EntityLink] = []
        for label, items in grouped.items():
            sorted_items = sorted(items, key=lambda item: (item.t_start, item.camera_id))
            resolved = True
            confidence = 0.87
            for prev, curr in zip(sorted_items, sorted_items[1:]):
                if prev.camera_id == curr.camera_id:
                    continue
                neighbors = set(camera_topology.get(prev.camera_id, tuple()))
                travel_time = max(0, curr.t_start - prev.t_end)
                if curr.camera_id not in neighbors and travel_time > max_cross_camera_travel_seconds:
                    resolved = False
                    confidence = 0.46
                    break

            entity_id = stable_id("PER", label)
            links.append(
                EntityLink(
                    entity_id=entity_id,
                    entity_type="person",
                    label=label,
                    track_ids=tuple(sorted(item.track_id for item in sorted_items)),
                    confidence=confidence,
                    resolved=resolved,
                )
            )

        return links


@dataclass
class TemporalGroundingAdapter:
    """Deterministic temporal localization with smoothing and hysteresis."""

    def localize(
        self,
        tracklet: Tracklet,
        clip: Clip,
        query_text: str,
        smoothing_window_size: int,
        hysteresis_high: float,
        hysteresis_low: float,
        mask_confidence_floor: float,
    ) -> TemporalSegment:
        query_tokens = set(tokenize(query_text))
        frames = [
            frame
            for frame in clip.frames
            if tracklet.t_start <= frame.timestamp <= tracklet.t_end
        ]

        timestamps: list[int] = []
        scores: list[float] = []
        for frame in frames:
            action_tokens = tokenize(" ".join(frame.actions))
            timestamps.append(frame.timestamp)
            overlap = overlap_score(list(query_tokens), action_tokens)
            if overlap > 0:
                score = 1.0
            elif frame.actions:
                score = 0.3
            else:
                score = 0.1
            scores.append(score)

        smoothed = smooth_curve(scores, window=smoothing_window_size)
        spans = self._extract_spans(timestamps=timestamps, scores=smoothed, high=hysteresis_high, low=hysteresis_low)

        failure_flags: list[str] = []
        if tracklet.mask_confidence < mask_confidence_floor:
            failure_flags.append("low_mask_confidence")

        if not spans:
            failure_flags.append("low_similarity")
            t_start = tracklet.t_start
            t_end = tracklet.t_end
            confidence = 0.35
        else:
            t_start, t_end, confidence = spans[0]

        action = "person_exits_vehicle" if "exit" in query_tokens else "tracked_activity"

        return TemporalSegment(
            segment_id=stable_id("SEG", tracklet.track_id, t_start, t_end, action),
            clip_id=clip.clip_id,
            camera_id=clip.camera_id,
            track_id=tracklet.track_id,
            action=action,
            t_start=t_start,
            t_end=t_end,
            confidence=round(confidence, 3),
            failure_flags=tuple(sorted(set(failure_flags))),
        )

    def _extract_spans(
        self,
        timestamps: list[int],
        scores: list[float],
        high: float,
        low: float,
    ) -> list[tuple[int, int, float]]:
        if not timestamps or not scores:
            return []

        spans: list[tuple[int, int, float]] = []
        in_span = False
        current_start = timestamps[0]
        current_scores: list[float] = []

        for ts, score in zip(timestamps, scores):
            if not in_span and score >= high:
                in_span = True
                current_start = ts
                current_scores = [score]
                continue

            if in_span:
                if score >= low:
                    current_scores.append(score)
                else:
                    spans.append((current_start, ts, mean(current_scores)))
                    in_span = False
                    current_scores = []

        if in_span:
            spans.append((current_start, timestamps[-1], mean(current_scores)))

        return spans


@dataclass
class MultimodalSynthesizerAdapter:
    """Deterministic grounded answer generator."""

    def synthesize(self, query_text: str, claims: list[ClaimRecord]) -> SynthesisOutput:
        grounded_claims: list[ClaimRecord] = []
        redacted = 0

        for claim in claims:
            if not claim.evidence_refs:
                redacted += 1
                continue
            grounded_claims.append(claim)

        if not grounded_claims:
            summary = (
                "Insufficient verified evidence to answer confidently. "
                "Returning conservative output with uncertainty."
            )
            evidence_appendix: list[str] = []
        else:
            lines = [
                f"{idx + 1}. {claim.text}"
                for idx, claim in enumerate(grounded_claims)
            ]
            summary = " ".join(lines)
            evidence_appendix = [
                (
                    f"claim={claim.claim_id} camera={claim.camera_id} "
                    f"t=({claim.t_start},{claim.t_end}) evidence={len(claim.evidence_refs)}"
                )
                for claim in grounded_claims
            ]

        return SynthesisOutput(
            summary=summary,
            claims=tuple(grounded_claims),
            evidence_appendix=tuple(evidence_appendix),
            redacted_claim_count=redacted,
        )

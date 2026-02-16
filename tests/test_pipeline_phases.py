from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_video_rag.demo_data import (  # noqa: E402
    build_ambiguous_person_request,
    build_red_suv_query_request,
    build_route_coverage_request,
)
from agentic_video_rag.pipeline import AgenticVideoRAGEngine  # noqa: E402
from agentic_video_rag.spec_loader import load_runtime_config  # noqa: E402
from agentic_video_rag.types import Clip, FrameObservation, QueryRequest  # noqa: E402

BASE_CONFIG = ROOT / "config/spec/groundtruth.yaml"


class PipelinePhaseTests(unittest.TestCase):
    def _new_engine(self, override_paths: list[Path] | None = None) -> AgenticVideoRAGEngine:
        config = load_runtime_config(BASE_CONFIG, override_paths=override_paths or [])
        return AgenticVideoRAGEngine(config)

    def test_stage1_routes_and_index_contract(self) -> None:
        engine = self._new_engine()
        request = build_route_coverage_request()
        result = engine.run(request)

        route_ids = {window.route_id for window in result.active_windows}
        self.assertEqual(route_ids, {"meta_sync", "sig_ex_adaptive", "cv_state", "bg_motion_trigger"})

        records = engine.stores.frame_index.records
        self.assertGreater(len(records), 0)
        self.assertTrue(all(record.embedding_id.startswith("EMB_") for record in records))
        self.assertTrue(all(record.window_id for record in records))

    def test_stage2_retrieval_and_cache_behavior(self) -> None:
        engine = self._new_engine()
        request = build_red_suv_query_request()

        first = engine.run(request)
        second = engine.run(request)

        threshold = engine.config.constants.retrieval.stage2_min_validation_confidence
        self.assertGreater(len(first.validated_windows), 0)
        self.assertTrue(all(window.confidence >= threshold for window in first.validated_windows))
        self.assertGreater(second.metrics.cache_hits, first.metrics.cache_hits)

    def test_stage3_grounding_fallback_and_artifacts(self) -> None:
        engine = self._new_engine()
        request = build_red_suv_query_request()
        result = engine.run(request)

        self.assertGreater(len(result.tracklets), 0)
        self.assertTrue(all(track.overlay_uri in engine.stores.artifact_store.artifacts for track in result.tracklets))

        with tempfile.TemporaryDirectory() as temp_dir:
            override = Path(temp_dir) / "high_mask_threshold.yaml"
            override.write_text(
                "constants:\n"
                "  grounding:\n"
                "    sam3_min_mask_confidence: 0.95\n"
                "    sam3_retry_max_attempts: 1\n",
                encoding="utf-8",
            )
            fallback_engine = self._new_engine([override])
            fallback_result = fallback_engine.run(build_red_suv_query_request())
            self.assertTrue(
                any(track.track_id.startswith("FALLBACK_TRACK") for track in fallback_result.tracklets)
            )

    def test_stage4_entity_resolution_with_ambiguity(self) -> None:
        engine = self._new_engine()
        result = engine.run(build_red_suv_query_request())

        self.assertTrue(any(link.entity_type == "object" and link.resolved for link in result.entity_links))
        self.assertTrue(any(link.entity_type == "person" and link.resolved for link in result.entity_links))

        ambiguous_engine = self._new_engine()
        ambiguous_result = ambiguous_engine.run(build_ambiguous_person_request())
        self.assertTrue(
            any(link.entity_type == "person" and not link.resolved for link in ambiguous_result.entity_links)
        )

    def test_stage5_temporal_localization_and_flags(self) -> None:
        engine = self._new_engine()
        request = build_red_suv_query_request()
        result_a = engine.run(request)
        result_b = engine.run(request)

        self.assertGreater(len(result_a.temporal_segments), 0)
        for segment in result_a.temporal_segments:
            self.assertLessEqual(segment.t_start, segment.t_end)
            self.assertIsInstance(segment.failure_flags, tuple)

        boundaries_a = [(segment.clip_id, segment.t_start, segment.t_end) for segment in result_a.temporal_segments]
        boundaries_b = [(segment.clip_id, segment.t_start, segment.t_end) for segment in result_b.temporal_segments]
        self.assertEqual(boundaries_a, boundaries_b)

        ambiguity_request = self._build_multi_actor_request()
        ambiguity_result = engine.run(ambiguity_request)
        self.assertTrue(
            any("multi_actor_ambiguity" in segment.failure_flags for segment in ambiguity_result.temporal_segments)
        )

    def test_stage6_graph_memory_and_evidence_integrity(self) -> None:
        engine = self._new_engine()
        result = engine.run(build_red_suv_query_request())

        node_types = {node.node_type for node in result.graph_nodes}
        self.assertIn("PersonEntityID", node_types)
        self.assertIn("ObjectClusterID", node_types)
        self.assertIn("CameraID", node_types)

        edge_types = {edge.edge_type for edge in result.graph_edges}
        self.assertIn("EXITS", edge_types)
        self.assertIn("MOVES_TO", edge_types)
        self.assertTrue(all(edge.evidence_refs for edge in result.graph_edges))
        self.assertEqual(len(result.graph_edges), len(engine.stores.graph_store.edges_with_evidence()))

    def test_stage7_synthesis_grounding_and_conservative_fallback(self) -> None:
        engine = self._new_engine()
        result = engine.run(build_red_suv_query_request())

        self.assertGreater(len(result.synthesis.claims), 0)
        self.assertTrue(all(claim.evidence_refs for claim in result.synthesis.claims))
        self.assertEqual(result.synthesis.redacted_claim_count, 0)

        empty_request = QueryRequest(
            query_id="query_empty",
            query_text="Find a person",
            clips=tuple(),
            camera_topology={},
        )
        empty_result = engine.run(empty_request)
        self.assertEqual(len(empty_result.synthesis.claims), 0)
        self.assertIn("Insufficient verified evidence", empty_result.synthesis.summary)

    def test_phase8_e2e_performance_and_release_readiness(self) -> None:
        engine = self._new_engine()
        request = build_red_suv_query_request()

        start = time.perf_counter()
        first_result = engine.run(request)
        second_result = engine.run(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self.assertGreater(len(first_result.synthesis.claims), 0)
        self.assertTrue(any("exited object entity" in claim.text for claim in first_result.synthesis.claims))
        self.assertTrue(any("moved to camera" in claim.text for claim in first_result.synthesis.claims))

        total_stage_ms = sum(second_result.metrics.stage_durations_ms.values())
        self.assertLess(total_stage_ms, 1500.0)
        self.assertLess(elapsed_ms, 3000.0)
        self.assertGreater(second_result.metrics.cache_hits, first_result.metrics.cache_hits)

        required_docs = [
            ROOT / "design/ops_runbook.md",
            ROOT / "design/known_limitations.md",
            ROOT / "design/rollback_notes.md",
            ROOT / "design/release_signoff_checklist.md",
        ]
        for path in required_docs:
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 20)

    def _build_multi_actor_request(self) -> QueryRequest:
        base = build_red_suv_query_request()
        ext_clip = base.clips[0]

        new_frames: list[FrameObservation] = []
        for frame in ext_clip.frames:
            if frame.timestamp in {10, 11}:
                objects = tuple([*frame.objects, "person_p2"])
            else:
                objects = frame.objects
            new_frames.append(
                FrameObservation(
                    timestamp=frame.timestamp,
                    objects=objects,
                    actions=frame.actions,
                    background_motion=frame.background_motion,
                )
            )

        updated_ext = Clip(
            clip_id=ext_clip.clip_id,
            camera_id=ext_clip.camera_id,
            camera_type=ext_clip.camera_type,
            location=ext_clip.location,
            duration_seconds=ext_clip.duration_seconds,
            frames=tuple(new_frames),
            metadata=dict(ext_clip.metadata),
        )

        return QueryRequest(
            query_id="query_multi_actor",
            query_text=base.query_text,
            clips=(updated_ext, *base.clips[1:]),
            camera_topology=base.camera_topology,
        )


if __name__ == "__main__":
    unittest.main()

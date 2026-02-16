"""Synthetic datasets used by tests and local demo runs."""

from __future__ import annotations

from .types import Clip, FrameObservation, QueryRequest


def _build_frames(
    duration_seconds: int,
    default_background_motion: float,
    objects_by_timestamp: dict[int, tuple[str, ...]],
    actions_by_timestamp: dict[int, tuple[str, ...]],
) -> tuple[FrameObservation, ...]:
    frames: list[FrameObservation] = []
    for ts in range(duration_seconds + 1):
        frames.append(
            FrameObservation(
                timestamp=ts,
                objects=objects_by_timestamp.get(ts, tuple()),
                actions=actions_by_timestamp.get(ts, tuple()),
                background_motion=default_background_motion,
            )
        )
    return tuple(frames)


def build_red_suv_query_request() -> QueryRequest:
    """Canonical end-to-end query fixture from spec blueprint."""

    ext_objects = {
        8: ("red_suv",),
        9: ("red_suv",),
        10: ("red_suv", "person_p1"),
        11: ("red_suv", "person_p1"),
        12: ("red_suv", "person_p1"),
        13: ("red_suv",),
    }
    ext_actions = {
        10: ("person_exits_suv",),
        11: ("person_exits_suv",),
    }

    int1_objects = {
        30: ("person_p1",),
        31: ("person_p1",),
        32: ("person_p1",),
        33: ("person_p1",),
    }
    int1_actions = {
        31: ("person_moves_to_interior",),
        32: ("person_moves_to_interior",),
    }

    int2_objects = {
        45: ("person_p1",),
        46: ("person_p1",),
        47: ("person_p1",),
    }
    int2_actions = {
        46: ("person_moves_to_interior",),
    }

    distractor_objects = {
        4: ("blue_sedan",),
        5: ("blue_sedan",),
    }

    clips = (
        Clip(
            clip_id="clip_ext_1",
            camera_id="cam_ext_1",
            camera_type="static",
            location="exterior",
            duration_seconds=60,
            frames=_build_frames(
                duration_seconds=60,
                default_background_motion=0.35,
                objects_by_timestamp=ext_objects,
                actions_by_timestamp=ext_actions,
            ),
            metadata={
                "has_motion_vectors": True,
                "active_windows": [{"t_start": 8, "t_end": 13}],
            },
        ),
        Clip(
            clip_id="clip_int_1",
            camera_id="cam_int_1",
            camera_type="static",
            location="interior",
            duration_seconds=60,
            frames=_build_frames(
                duration_seconds=60,
                default_background_motion=0.18,
                objects_by_timestamp=int1_objects,
                actions_by_timestamp=int1_actions,
            ),
            metadata={},
        ),
        Clip(
            clip_id="clip_int_2",
            camera_id="cam_int_2",
            camera_type="static",
            location="interior",
            duration_seconds=60,
            frames=_build_frames(
                duration_seconds=60,
                default_background_motion=0.22,
                objects_by_timestamp=int2_objects,
                actions_by_timestamp=int2_actions,
            ),
            metadata={},
        ),
        Clip(
            clip_id="clip_noise_1",
            camera_id="cam_ext_2",
            camera_type="static",
            location="exterior",
            duration_seconds=60,
            frames=_build_frames(
                duration_seconds=60,
                default_background_motion=0.65,
                objects_by_timestamp=distractor_objects,
                actions_by_timestamp={},
            ),
            metadata={},
        ),
    )

    return QueryRequest(
        query_id="query_red_suv_tracking",
        query_text="Find the red SUV, identify the person who got out, and track that specific person across the interior cameras.",
        clips=clips,
        camera_topology={
            "cam_ext_1": ("cam_int_1",),
            "cam_int_1": ("cam_ext_1", "cam_int_2"),
            "cam_int_2": ("cam_int_1",),
            "cam_ext_2": ("cam_ext_1",),
        },
    )


def build_route_coverage_request() -> QueryRequest:
    """Fixture that triggers all four Stage 1 routing paths."""

    clip_meta = Clip(
        clip_id="clip_meta",
        camera_id="cam_meta",
        camera_type="static",
        location="exterior",
        duration_seconds=20,
        frames=_build_frames(
            duration_seconds=20,
            default_background_motion=0.2,
            objects_by_timestamp={4: ("red_suv",), 5: ("person_p1",)},
            actions_by_timestamp={5: ("person_exits_suv",)},
        ),
        metadata={"has_motion_vectors": True, "active_windows": [{"t_start": 4, "t_end": 6}]},
    )

    clip_moving = Clip(
        clip_id="clip_moving",
        camera_id="cam_move",
        camera_type="moving",
        location="exterior",
        duration_seconds=20,
        frames=_build_frames(
            duration_seconds=20,
            default_background_motion=0.5,
            objects_by_timestamp={7: ("person_p2",), 8: ("person_p2",)},
            actions_by_timestamp={8: ("person_runs",)},
        ),
        metadata={},
    )

    clip_static_low = Clip(
        clip_id="clip_static_low",
        camera_id="cam_static_low",
        camera_type="static",
        location="interior",
        duration_seconds=20,
        frames=_build_frames(
            duration_seconds=20,
            default_background_motion=0.1,
            objects_by_timestamp={9: ("person_p3",), 10: ("person_p3",)},
            actions_by_timestamp={10: ("person_walks",)},
        ),
        metadata={},
    )

    clip_static_high = Clip(
        clip_id="clip_static_high",
        camera_id="cam_static_high",
        camera_type="static",
        location="exterior",
        duration_seconds=20,
        frames=_build_frames(
            duration_seconds=20,
            default_background_motion=0.8,
            objects_by_timestamp={11: ("vehicle_unknown",), 12: ("vehicle_unknown",)},
            actions_by_timestamp={12: ("object_moves",)},
        ),
        metadata={},
    )

    return QueryRequest(
        query_id="query_route_coverage",
        query_text="Find person and vehicle movement",
        clips=(clip_meta, clip_moving, clip_static_low, clip_static_high),
        camera_topology={
            "cam_meta": ("cam_move",),
            "cam_move": ("cam_meta", "cam_static_low"),
            "cam_static_low": ("cam_move", "cam_static_high"),
            "cam_static_high": ("cam_static_low",),
        },
    )


def build_ambiguous_person_request() -> QueryRequest:
    """Fixture that introduces ambiguous cross-camera person links."""

    clip_a = Clip(
        clip_id="clip_amb_a",
        camera_id="cam_far_a",
        camera_type="static",
        location="interior",
        duration_seconds=30,
        frames=_build_frames(
            duration_seconds=30,
            default_background_motion=0.2,
            objects_by_timestamp={5: ("person_px",), 6: ("person_px",)},
            actions_by_timestamp={6: ("person_moves",)},
        ),
        metadata={},
    )

    clip_b = Clip(
        clip_id="clip_amb_b",
        camera_id="cam_far_b",
        camera_type="static",
        location="interior",
        duration_seconds=420,
        frames=_build_frames(
            duration_seconds=420,
            default_background_motion=0.2,
            objects_by_timestamp={400: ("person_px",), 401: ("person_px",)},
            actions_by_timestamp={401: ("person_moves",)},
        ),
        metadata={},
    )

    return QueryRequest(
        query_id="query_ambiguous_person",
        query_text="Track this person across cameras",
        clips=(clip_a, clip_b),
        camera_topology={
            "cam_far_a": tuple(),
            "cam_far_b": tuple(),
        },
    )

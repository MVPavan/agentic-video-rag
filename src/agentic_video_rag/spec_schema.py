"""Strict schema definitions for Agentic Video RAG runtime configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

StageId = Literal[
    "stage_1",
    "stage_2",
    "stage_3",
    "stage_4",
    "stage_5",
    "stage_6",
    "stage_7",
]

EXPECTED_STAGE_IDS: set[StageId] = {
    "stage_1",
    "stage_2",
    "stage_3",
    "stage_4",
    "stage_5",
    "stage_6",
    "stage_7",
}

EXTERNAL_IO_IDS = {
    "raw_video_stream",
    "camera_metadata_stream",
    "optional_audio_stream",
    "synthesized_response",
}

REQUIRED_ORCHESTRATION_STATE_KEYS = {
    "query_id",
    "normalized_query",
    "candidate_windows",
    "grounded_tracks",
    "entity_links",
    "temporal_segments",
    "evidence_package",
}


class StrictBaseModel(BaseModel):
    """Base model with strict validation and no unknown keys."""

    model_config = ConfigDict(extra="forbid", strict=True)


class MetaConfig(StrictBaseModel):
    spec_id: str
    spec_version: str
    source_spec: str
    generated_on: str


class StageCatalog(StrictBaseModel):
    stage_1: Literal["activity_ingestion"]
    stage_2: Literal["temporal_retrieval"]
    stage_3: Literal["spatial_grounding"]
    stage_4: Literal["entity_resolution"]
    stage_5: Literal["temporal_localization"]
    stage_6: Literal["graph_memory"]
    stage_7: Literal["multimodal_synthesis"]

    def as_map(self) -> dict[StageId, str]:
        return self.model_dump()  # type: ignore[return-value]


class ModelEntry(StrictBaseModel):
    provider: str
    role: str
    modality: Literal["text", "multimodal", "image_text", "video", "image", "image_video"]


class DatastoreEntry(StrictBaseModel):
    datastore_type: Literal["vector_db", "graph_db", "cache", "artifact_store"]
    engine: str


class ResourceEntry(StrictBaseModel):
    datastore_id: str
    resource_type: Literal["vector_collection", "cache_level", "graph", "artifact_bundle"]


class RegistryConfig(StrictBaseModel):
    models: dict[str, ModelEntry]
    datastores: dict[str, DatastoreEntry]
    resources: dict[str, ResourceEntry]


class RetrievalConstants(StrictBaseModel):
    stage2_initial_top_k_windows: int = Field(gt=0)
    stage2_validated_top_k_windows: int = Field(gt=0)
    stage2_min_validation_confidence: float = Field(ge=0, le=1)


class GroundingConstants(StrictBaseModel):
    sam3_min_mask_confidence: float = Field(ge=0, le=1)
    sam3_retry_max_attempts: int = Field(ge=0)


class ReidConstants(StrictBaseModel):
    object_reid_min_similarity: float = Field(ge=0, le=1)
    person_reid_min_similarity: float = Field(ge=0, le=1)
    max_cross_camera_travel_seconds: int = Field(ge=0)


class TemporalLocalizationConstants(StrictBaseModel):
    smoothing_method: Literal["savitzky_golay", "ema"]
    smoothing_window_size: int = Field(gt=0)
    hysteresis_high: float = Field(ge=0, le=1)
    hysteresis_low: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_hysteresis(self) -> "TemporalLocalizationConstants":
        if self.hysteresis_high <= self.hysteresis_low:
            raise ValueError("hysteresis_high must be greater than hysteresis_low")
        return self


class ConstantsConfig(StrictBaseModel):
    retrieval: RetrievalConstants
    grounding: GroundingConstants
    reid: ReidConstants
    temporal_localization: TemporalLocalizationConstants


class OrchestratorModels(StrictBaseModel):
    phase_1: str
    phase_2: str


class OrchestrationConfig(StrictBaseModel):
    framework: Literal["LangGraph"]
    orchestrator_models: OrchestratorModels
    required_state_keys: list[str]
    branching_hooks: list[str]

    @model_validator(mode="after")
    def validate_state_contract(self) -> "OrchestrationConfig":
        missing = REQUIRED_ORCHESTRATION_STATE_KEYS - set(self.required_state_keys)
        if missing:
            raise ValueError(f"required_state_keys missing required keys: {sorted(missing)}")
        if not self.branching_hooks:
            raise ValueError("branching_hooks must not be empty")
        return self


class PhaseAConfig(StrictBaseModel):
    orchestration: OrchestrationConfig


class StageConfig(StrictBaseModel):
    stage_id: StageId
    stage_name: str
    depends_on: list[StageId]
    models: list[str]
    reads: list[str]
    writes: list[str]


class PhaseBConfig(StrictBaseModel):
    stages: list[StageConfig]


class RuntimeConfig(StrictBaseModel):
    meta: MetaConfig
    stage_catalog: StageCatalog
    registry: RegistryConfig
    constants: ConstantsConfig
    phase_a: PhaseAConfig
    phase_b: PhaseBConfig

    @model_validator(mode="after")
    def validate_references(self) -> "RuntimeConfig":
        catalog = self.stage_catalog.as_map()

        stage_ids = [stage.stage_id for stage in self.phase_b.stages]
        unique_stage_ids = set(stage_ids)
        if len(stage_ids) != len(unique_stage_ids):
            raise ValueError("phase_b.stages must contain unique stage_id values")

        if unique_stage_ids != EXPECTED_STAGE_IDS:
            missing = sorted(EXPECTED_STAGE_IDS - unique_stage_ids)
            extra = sorted(unique_stage_ids - EXPECTED_STAGE_IDS)
            raise ValueError(f"stage completeness failed: missing={missing}, extra={extra}")

        model_ids = set(self.registry.models.keys())
        datastore_ids = set(self.registry.datastores.keys())
        resource_ids = set(self.registry.resources.keys())

        for resource_id, resource in self.registry.resources.items():
            if resource.datastore_id not in datastore_ids:
                raise ValueError(
                    f"resource '{resource_id}' references unknown datastore '{resource.datastore_id}'"
                )

        phase_1_model = self.phase_a.orchestration.orchestrator_models.phase_1
        phase_2_model = self.phase_a.orchestration.orchestrator_models.phase_2
        for orchestrator_model in (phase_1_model, phase_2_model):
            if orchestrator_model not in model_ids:
                raise ValueError(
                    f"orchestration references unknown model '{orchestrator_model}'"
                )

        for stage in self.phase_b.stages:
            expected_name = catalog[stage.stage_id]
            if stage.stage_name != expected_name:
                raise ValueError(
                    f"stage_name mismatch for {stage.stage_id}: "
                    f"expected '{expected_name}', found '{stage.stage_name}'"
                )

            unknown_dependencies = set(stage.depends_on) - unique_stage_ids
            if unknown_dependencies:
                raise ValueError(
                    f"{stage.stage_id} has unknown dependencies {sorted(unknown_dependencies)}"
                )

            unknown_models = set(stage.models) - model_ids
            if unknown_models:
                raise ValueError(
                    f"{stage.stage_id} references unknown models {sorted(unknown_models)}"
                )

            for io_ref in [*stage.reads, *stage.writes]:
                if io_ref in resource_ids or io_ref in EXTERNAL_IO_IDS:
                    continue
                raise ValueError(f"{stage.stage_id} references unknown resource/io '{io_ref}'")

        return self


def validate_config_dict(config: dict) -> RuntimeConfig:
    """Convenience helper that raises a detailed validation error on failure."""

    try:
        return RuntimeConfig.model_validate(config)
    except ValidationError as exc:
        raise exc

"""Dependency-neutral types for enforceable batch STT execution plans."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

SttPlanScalar = str | int | float | bool | None | tuple[str, ...]

_SAFE_LABEL_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._+-]*"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._+-]*)?$"
)
_STABLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_ENDPOINT_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTENT_SHA_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_SNAPSHOT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_MODULE_RE = re.compile(r"^(?:tldw_Server_API|Helper_Scripts)" r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_DISTRIBUTION_RE = re.compile(r"^[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*$")
_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_SECRET_SHAPED_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])"
    r"(?:api[_-]?key|authorization|bearer|password|secret|"
    r"access[_-]?token|refresh[_-]?token|private[_-]?key)"
    r"(?:$|[^A-Za-z0-9])",
    re.IGNORECASE,
)
_MAX_EXECUTION_MISMATCHES = 8


def _require_nonblank(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value


def _reject_hostile_serialized_string(value: str, field_name: str) -> str:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")
    if (
        value.startswith(("/", "\\\\", "//", "~", "./", "../"))
        or _WINDOWS_ABSOLUTE_RE.match(value)
    ):
        raise ValueError(f"{field_name} must not contain an absolute or relative path")
    if (
        _URL_SCHEME_RE.match(value)
        or "://" in value
        or any(character in value for character in ("@", "?", "#"))
    ):
        raise ValueError(f"{field_name} must not contain URL components")
    if _SECRET_SHAPED_RE.search(value):
        raise ValueError(f"{field_name} must not contain secret-shaped values")
    return value


def _require_safe_label(value: object, field_name: str) -> str:
    label = _require_nonblank(value, field_name)
    if not field_name.endswith("model_label"):
        return _require_serialized_id(label, field_name)
    if label.startswith("external:"):
        external_label = label.removeprefix("external:")
        _require_stable_id(external_label, field_name)
        _reject_hostile_serialized_string(external_label, field_name)
        return label
    _reject_hostile_serialized_string(label, field_name)
    if (
        not _SAFE_LABEL_RE.fullmatch(label)
        or label.startswith((".", "/", "~"))
        or "\\" in label
    ):
        raise ValueError(f"{field_name} must be a safe provider/model label")
    return label


def _require_stable_id(value: object, field_name: str) -> str:
    identifier = _require_nonblank(value, field_name)
    if not _STABLE_ID_RE.fullmatch(identifier):
        raise ValueError(f"{field_name} must be a stable identifier")
    return identifier


def _require_serialized_id(value: object, field_name: str) -> str:
    identifier = _require_stable_id(value, field_name)
    return _reject_hostile_serialized_string(identifier, field_name)


def _require_artifact_id(value: object, field_name: str) -> str:
    identifier = _require_nonblank(value, field_name)
    if _CONTENT_SHA_RE.fullmatch(identifier) or _SNAPSHOT_COMMIT_RE.fullmatch(
        identifier
    ):
        return identifier
    return _require_serialized_id(identifier, field_name)


def _require_source_module(value: object, field_name: str) -> str:
    module = _require_nonblank(value, field_name)
    _reject_hostile_serialized_string(module, field_name)
    if not _SOURCE_MODULE_RE.fullmatch(module):
        raise ValueError(f"{field_name} must contain valid project modules")
    return module


def _require_distribution_name(value: object, field_name: str) -> str:
    distribution = _require_nonblank(value, field_name)
    _reject_hostile_serialized_string(distribution, field_name)
    if not _DISTRIBUTION_RE.fullmatch(distribution):
        raise ValueError(f"{field_name} must contain valid distribution names")
    return distribution


def _validate_ordered_unique(
    values: tuple[str, ...],
    field_name: str,
    *,
    validator: Callable[[object, str], str] = _require_stable_id,
) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    for value in values:
        validator(value, field_name)
    if len(set(values)) != len(values) or values != tuple(sorted(values)):
        raise ValueError(f"{field_name} must be unique and lexicographically ordered")


def _validate_scalar(value: object, field_name: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return
    raise ValueError(f"{field_name} contains an unsupported runtime value")


def _validate_safe_scalar(value: object, field_name: str) -> None:
    _validate_scalar(value, field_name)
    string_values = value if isinstance(value, tuple) else (value,)
    for item in string_values:
        if not isinstance(item, str):
            continue
        if item.startswith("fixed:"):
            _require_serialized_id(item.removeprefix("fixed:"), field_name)
            continue
        _require_serialized_id(item, field_name)


def _validate_settings(
    settings: tuple[tuple[str, SttPlanScalar], ...],
    field_name: str,
    *,
    serialized: bool = False,
) -> tuple[str, ...]:
    if not isinstance(settings, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    keys: list[str] = []
    for item in settings:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{field_name} entries must be key/value tuples")
        key, value = item
        key_validator = _require_serialized_id if serialized else _require_stable_id
        key_validator(key, f"{field_name} key")
        if serialized:
            _validate_safe_scalar(value, field_name)
        else:
            _validate_scalar(value, field_name)
        keys.append(key)
    ordered_keys = tuple(keys)
    if len(set(ordered_keys)) != len(ordered_keys) or ordered_keys != tuple(sorted(ordered_keys)):
        raise ValueError(f"{field_name} keys must be unique and lexicographically ordered")
    return ordered_keys


def _safe_scalar(value: SttPlanScalar) -> SttPlanScalar | list[str]:
    return list(value) if isinstance(value, tuple) else value


def _declared_dict(instance: object) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataclass_field in fields(instance):
        value = getattr(instance, dataclass_field.name)
        if isinstance(value, SttAudioEgress):
            value = value.value
        elif isinstance(value, tuple):
            value = list(value)
        result[dataclass_field.name] = value
    return result


class SttAudioEgress(str, Enum):
    """Whether a planned provider sends audio over a network interface."""

    NONE = "none"
    LOOPBACK = "loopback"
    REMOTE = "remote"


@dataclass(frozen=True)
class SttExecutionRoute:
    """One authorized backend route in execution-attempt order."""

    route_id: str
    provider: str
    model_label: str
    artifact_id: str | None
    identity_resolved: bool
    backend: str
    source: str
    audio_egress: SttAudioEgress
    endpoint_id: str | None
    device: str | None
    compute_type: str | None
    dtype: str | None
    decoding_ids: tuple[str, ...]
    local_model_available: bool
    would_download: bool

    def __post_init__(self) -> None:
        _require_serialized_id(self.route_id, "route_id")
        _require_safe_label(self.provider, "provider")
        _require_safe_label(self.model_label, "model_label")
        _require_serialized_id(self.backend, "backend")
        _require_serialized_id(self.source, "source")
        if not isinstance(self.audio_egress, SttAudioEgress):
            raise ValueError("audio_egress must be an SttAudioEgress")
        if self.audio_egress is SttAudioEgress.NONE:
            if self.endpoint_id is not None:
                raise ValueError("endpoint_id is invalid when audio egress is none")
        elif not isinstance(self.endpoint_id, str) or not _ENDPOINT_ID_RE.fullmatch(self.endpoint_id):
            raise ValueError("network routes require an opaque endpoint_id")
        if self.artifact_id is not None:
            _require_artifact_id(self.artifact_id, "artifact_id")
        if not isinstance(self.identity_resolved, bool):
            raise ValueError("identity_resolved must be boolean")
        if self.identity_resolved and (
            self.artifact_id is None
            or not (_CONTENT_SHA_RE.fullmatch(self.artifact_id) or _SNAPSHOT_COMMIT_RE.fullmatch(self.artifact_id))
        ):
            raise ValueError("resolved identity requires an immutable artifact ID")
        for name in ("device", "compute_type", "dtype"):
            value = getattr(self, name)
            if value is not None:
                _require_serialized_id(value, name)
        _validate_ordered_unique(
            self.decoding_ids,
            "decoding_ids",
            validator=_require_serialized_id,
        )
        if not isinstance(self.local_model_available, bool):
            raise ValueError("local_model_available must be boolean")
        if not isinstance(self.would_download, bool):
            raise ValueError("would_download must be boolean")

    def as_safe_dict(self) -> dict[str, Any]:
        """Return only the route's explicitly declared safe fields."""
        return _declared_dict(self)


@dataclass(frozen=True)
class SttExecutionDescriptor:
    """Safe, serializable description of the complete execution authorization."""

    requested_provider: str
    requested_model_label: str
    resolved_provider: str
    resolved_model_label: str
    routes: tuple[SttExecutionRoute, ...]
    honors_task: bool
    honors_language: bool
    honors_prompt_absence: bool
    honors_hotword_absence: bool
    honors_diarization: bool
    honors_word_timestamps: bool
    decoding_settings: tuple[tuple[str, SttPlanScalar], ...]
    source_modules: tuple[str, ...]
    dependency_distributions: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_safe_label(self.requested_provider, "requested_provider")
        _require_safe_label(self.requested_model_label, "requested_model_label")
        _require_safe_label(self.resolved_provider, "resolved_provider")
        _require_safe_label(self.resolved_model_label, "resolved_model_label")
        if (
            not isinstance(self.routes, tuple)
            or not self.routes
            or not all(isinstance(route, SttExecutionRoute) for route in self.routes)
        ):
            raise ValueError("routes must be a non-empty tuple of execution routes")
        route_ids = tuple(route.route_id for route in self.routes)
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("route IDs must be unique")
        route_fields = tuple(field.name for field in fields(SttExecutionRoute))
        material_fields = tuple(name for name in route_fields if name != "route_id")
        seen_routes: set[tuple[Any, ...]] = set()
        for route in self.routes:
            material_route = tuple(getattr(route, name) for name in material_fields)
            if material_route in seen_routes:
                raise ValueError("fallback routes must be materially distinct")
            seen_routes.add(material_route)
        decoding_keys = _validate_settings(
            self.decoding_settings,
            "decoding_settings",
            serialized=True,
        )
        for route in self.routes:
            if not set(route.decoding_ids).issubset(decoding_keys):
                raise ValueError("route decoder IDs must name declared settings")
        _validate_ordered_unique(
            self.source_modules,
            "source_modules",
            validator=_require_source_module,
        )
        _validate_ordered_unique(
            self.dependency_distributions,
            "dependency_distributions",
            validator=_require_distribution_name,
        )
        for name in (
            "honors_task",
            "honors_language",
            "honors_prompt_absence",
            "honors_hotword_absence",
            "honors_diarization",
            "honors_word_timestamps",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")

    @property
    def primary_route(self) -> SttExecutionRoute:
        """Return the first authorized route."""
        return self.routes[0]

    @property
    def fallback_allowed(self) -> bool:
        """Return whether the plan authorizes more than one route."""
        return len(self.routes) > 1

    def as_safe_dict(self) -> dict[str, Any]:
        """Serialize only declared safe descriptor fields."""
        return {
            "requested_provider": self.requested_provider,
            "requested_model_label": self.requested_model_label,
            "resolved_provider": self.resolved_provider,
            "resolved_model_label": self.resolved_model_label,
            "routes": [route.as_safe_dict() for route in self.routes],
            "honors_task": self.honors_task,
            "honors_language": self.honors_language,
            "honors_prompt_absence": self.honors_prompt_absence,
            "honors_hotword_absence": self.honors_hotword_absence,
            "honors_diarization": self.honors_diarization,
            "honors_word_timestamps": self.honors_word_timestamps,
            "decoding_settings": [[key, _safe_scalar(value)] for key, value in self.decoding_settings],
            "source_modules": list(self.source_modules),
            "dependency_distributions": list(self.dependency_distributions),
        }


@dataclass(frozen=True)
class SttBatchExecutionPlan:
    """Safe descriptor plus in-memory runtime values for one batch request."""

    descriptor: SttExecutionDescriptor
    task: str
    language: str | None
    prompt: str | None = field(default=None, repr=False)
    hotwords: tuple[str, ...] = field(default=(), repr=False)
    diarization: bool = False
    word_timestamps: bool = False
    runtime_settings: tuple[tuple[str, SttPlanScalar], ...] = field(
        default=(),
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, SttExecutionDescriptor):
            raise ValueError("descriptor must be an SttExecutionDescriptor")
        _require_stable_id(self.task, "task")
        if self.language is not None:
            _require_stable_id(self.language, "language")
        if self.prompt is not None and not isinstance(self.prompt, str):
            raise ValueError("prompt must be a string or None")
        if not isinstance(self.hotwords, tuple) or not all(isinstance(value, str) for value in self.hotwords):
            raise ValueError("hotwords must be a tuple of strings")
        if not isinstance(self.diarization, bool):
            raise ValueError("diarization must be boolean")
        if not isinstance(self.word_timestamps, bool):
            raise ValueError("word_timestamps must be boolean")
        _validate_settings(self.runtime_settings, "runtime_settings")

    def runtime_values(self) -> dict[str, SttPlanScalar]:
        """Return the secret runtime settings as an in-memory mapping."""
        return dict(self.runtime_settings)


@dataclass(frozen=True)
class SttActualExecution:
    """Verified material execution fields reported by a provider helper."""

    route_id: str
    provider: str
    model_label: str
    artifact_id: str | None
    backend: str
    audio_egress: SttAudioEgress
    endpoint_id: str | None
    source: str
    device: str | None
    compute_type: str | None
    dtype: str | None
    decoding_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_serialized_id(self.route_id, "route_id")
        _require_safe_label(self.provider, "provider")
        _require_safe_label(self.model_label, "model_label")
        _require_serialized_id(self.backend, "backend")
        _require_serialized_id(self.source, "source")
        if self.artifact_id is not None:
            _require_artifact_id(self.artifact_id, "artifact_id")
        if not isinstance(self.audio_egress, SttAudioEgress):
            raise ValueError("audio_egress must be an SttAudioEgress")
        if self.audio_egress is SttAudioEgress.NONE:
            if self.endpoint_id is not None:
                raise ValueError("endpoint_id is invalid when audio egress is none")
        elif not isinstance(self.endpoint_id, str) or not _ENDPOINT_ID_RE.fullmatch(self.endpoint_id):
            raise ValueError("network execution requires an opaque endpoint_id")
        for name in ("device", "compute_type", "dtype"):
            value = getattr(self, name)
            if value is not None:
                _require_serialized_id(value, name)
        _validate_ordered_unique(
            self.decoding_ids,
            "decoding_ids",
            validator=_require_serialized_id,
        )

    def as_safe_dict(self) -> dict[str, Any]:
        """Return only the actual execution's declared safe fields."""
        return _declared_dict(self)


@dataclass(frozen=True)
class SttLoadedRuntime:
    """Provider runtime components paired with their verified execution."""

    components: tuple[Any, ...] = field(repr=False, compare=False)
    actual_execution: SttActualExecution

    def __post_init__(self) -> None:
        if not isinstance(self.components, tuple):
            raise ValueError("components must be a tuple")
        if not isinstance(self.actual_execution, SttActualExecution):
            raise ValueError("actual_execution must be an SttActualExecution")


@dataclass(frozen=True)
class SttTranscriptionOutcome:
    """Provider artifact paired with verified actual execution."""

    artifact: dict[str, Any] = field(repr=False, compare=False)
    actual_execution: SttActualExecution

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, dict):
            raise ValueError("artifact must be a dictionary")
        if not isinstance(self.actual_execution, SttActualExecution):
            raise ValueError("actual_execution must be an SttActualExecution")


def _actual_matches_route(
    actual: SttActualExecution,
    route: SttExecutionRoute,
) -> bool:
    material_fields = (
        "route_id",
        "provider",
        "model_label",
        "artifact_id",
        "backend",
        "source",
        "audio_egress",
        "endpoint_id",
        "device",
        "compute_type",
        "dtype",
        "decoding_ids",
    )
    return all(
        getattr(route, name) is None or getattr(actual, name) == getattr(route, name) for name in material_fields
    )


def _semantic_mismatches(plan: SttBatchExecutionPlan) -> list[str]:
    descriptor = plan.descriptor
    mismatches: list[str] = []
    if not descriptor.honors_task:
        mismatches.append("task")
    if plan.language is not None and not descriptor.honors_language:
        mismatches.append("language")
    if plan.prompt is None and not descriptor.honors_prompt_absence:
        mismatches.append("prompt_absence")
    if not plan.hotwords and not descriptor.honors_hotword_absence:
        mismatches.append("hotword_absence")
    if plan.diarization and not descriptor.honors_diarization:
        mismatches.append("diarization")
    if plan.word_timestamps and not descriptor.honors_word_timestamps:
        mismatches.append("word_timestamps")
    return mismatches[:_MAX_EXECUTION_MISMATCHES]


def finalize_stt_artifact(
    artifact: object,
    *,
    plan: SttBatchExecutionPlan,
    actual: SttActualExecution,
) -> dict[str, Any]:
    """Validate and safely finalize an artifact from a planned STT execution."""
    from tldw_Server_API.app.core.exceptions import (
        STTExecutionPlanError,
        STTTranscriptionError,
    )

    if not isinstance(plan, SttBatchExecutionPlan):
        raise STTExecutionPlanError("Planned STT execution is missing a valid plan")
    if not isinstance(actual, SttActualExecution):
        raise STTExecutionPlanError("Planned STT execution did not report typed actual execution")
    if not any(_actual_matches_route(actual, route) for route in plan.descriptor.routes):
        raise STTExecutionPlanError("Actual STT execution used an undeclared route")
    if not isinstance(artifact, Mapping):
        raise STTTranscriptionError("STT provider returned a non-mapping artifact")
    text = artifact.get("text")
    segments = artifact.get("segments")
    if not isinstance(text, str) or not isinstance(segments, list):
        raise STTTranscriptionError("STT provider artifact requires string text and list segments")

    from .Audio_Transcription_Lib import is_transcription_error_message

    if is_transcription_error_message(text):
        raise STTTranscriptionError(text)

    finalized: dict[str, Any] = {"text": text, "segments": segments}
    for key in ("language", "diarization", "usage"):
        if key in artifact:
            finalized[key] = artifact[key]
    finalized["actual_execution"] = actual.as_safe_dict()
    mismatches = _semantic_mismatches(plan)
    if mismatches:
        finalized["execution_mismatch"] = mismatches
    return finalized

"""Immutable, side-effect-free contracts for research discovery V2."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_QUERY_NAME_RE = re.compile(r"[A-Za-z0-9_.\[\]-]+\Z")
_MISSING = object()


class RouteKind(str, Enum):
    """How a route relates a target to a physical backend."""

    DIRECT = "direct"
    AGGREGATOR = "aggregator"
    SITE_SEARCH = "site_search"


class QueryMode(str, Enum):
    """Supported query semantics for one route."""

    GENERAL_FREE_TEXT = "general_free_text"
    STRUCTURED_QUERY = "structured_query"
    IDENTIFIER_LOOKUP = "identifier_lookup"
    RECENT_FEED = "recent_feed"
    DATE_INTERVAL = "date_interval"
    CATEGORY_BROWSE = "category_browse"


class CredentialRequirement(str, Enum):
    """Static authentication requirement declared by a route."""

    NONE = "none"
    API_KEY = "api_key"


class CredentialStatus(str, Enum):
    """Non-sensitive credential readiness state."""

    NOT_REQUIRED = "not_required"
    OUT_OF_SCOPE = "out_of_scope"


class SourceConstraint(str, Enum):
    """How a route constrains results to a catalog target."""

    NATIVE_CORPUS = "native_corpus"
    PROVIDER_SOURCE_FILTER = "provider_source_filter"
    PROVIDER_DOMAIN_FILTER = "provider_domain_filter"


class PredicateOperator(str, Enum):
    """Supported source-attribution predicate operations."""

    EQUALS_ANY = "equals_any"
    CONTAINS_ANY = "contains_any"


class AttributionMatch(str, Enum):
    """Three-valued result of evaluating a source predicate."""

    MATCH = "match"
    NON_MATCH = "non_match"
    AMBIGUOUS = "ambiguous"


class ExecutionMode(str, Enum):
    """Explicit non-production execution modes for the foundation."""

    OFFLINE_FIXTURE = "offline_fixture"
    SYNTHETIC = "synthetic"


class ReadinessState(str, Enum):
    """Route availability in one immutable overlay."""

    READY = "ready"
    DISABLED = "disabled"
    POLICY_BLOCKED = "policy_blocked"
    UNCERTIFIED = "uncertified"
    CERTIFICATION_EXPIRED = "certification_expired"
    UNHEALTHY = "unhealthy"
    CREDENTIALED_OUT_OF_SCOPE = "credentialed_out_of_scope"


class OperationKind(str, Enum):
    """One declarative physical operation in an attempt."""

    SEARCH = "search"
    CONDITIONAL_SUMMARY = "conditional_summary"


class SkippedStatus(str, Enum):
    """Top-level planning status for a target without executable work."""

    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


class SkippedCode(str, Enum):
    """Typed reason for omitting executable work."""

    CREDENTIALED_OUT_OF_SCOPE = "credentialed_out_of_scope"
    ROUTE_NOT_READY = "route_not_ready"


@dataclass(frozen=True, slots=True)
class QueryPair:
    """One immutable query or filter name/value pair."""

    name: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _QUERY_NAME_RE.fullmatch(self.name):
            raise ValueError("invalid_query_pair_name")
        if not isinstance(self.value, str) or "\x00" in self.value:
            raise ValueError("invalid_query_pair_value")


@dataclass(frozen=True, slots=True)
class ExactOrigin:
    """A normalized route transport origin."""

    scheme: str
    host: str
    port: int

    def __post_init__(self) -> None:
        if self.scheme not in {"http", "https"}:
            raise ValueError("invalid_origin_scheme")
        if (
            not isinstance(self.host, str)
            or self.host != self.host.strip().lower()
            or any(character in self.host for character in "/@?#")
            or not _valid_hostname(self.host)
        ):
            raise ValueError("invalid_origin_host")
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 1 <= self.port <= 65_535:
            raise ValueError("invalid_origin_port")


@dataclass(frozen=True, slots=True)
class RouteLimits:
    """Per-route ceilings copied into each descriptive intent."""

    max_pages: int
    max_redirects: int
    max_retries: int
    timeout_ms: int
    max_response_bytes: int
    max_results: int

    def __post_init__(self) -> None:
        _require_positive_int("max_pages", self.max_pages)
        _require_nonnegative_int("max_redirects", self.max_redirects)
        _require_nonnegative_int("max_retries", self.max_retries)
        _require_positive_int("timeout_ms", self.timeout_ms)
        _require_positive_int("max_response_bytes", self.max_response_bytes)
        _require_positive_int("max_results", self.max_results)


@dataclass(frozen=True, slots=True)
class BudgetCeilings:
    """Independent ceilings accepted by the pure planner."""

    max_route_attempts: int
    max_physical_dispatches: int
    max_pages_per_route: int
    max_redirects: int
    max_retries: int
    max_wall_time_ms: int
    max_results: int

    def __post_init__(self) -> None:
        for name in (
            "max_route_attempts",
            "max_physical_dispatches",
            "max_pages_per_route",
            "max_redirects",
            "max_retries",
            "max_wall_time_ms",
            "max_results",
        ):
            _require_nonnegative_int(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class SourcePredicate:
    """Typed provider-field predicate used only for source attribution."""

    field_path: tuple[str, ...]
    operator: PredicateOperator
    values: tuple[str, ...]
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        _require_tuple("field_path", self.field_path)
        _require_tuple("values", self.values)
        if not self.field_path:
            raise ValueError("empty_source_predicate_field_path")
        for segment in self.field_path:
            _validate_identifier("source_predicate_field", segment)
        _require_enum("predicate_operator", self.operator, PredicateOperator)
        if not self.values or any(not isinstance(value, str) or not value for value in self.values):
            raise ValueError("invalid_source_predicate_values")
        if len(set(self.values)) != len(self.values):
            raise ValueError("duplicate_source_predicate_value")
        if not isinstance(self.case_sensitive, bool):
            raise TypeError("case_sensitive_must_be_bool")


@dataclass(frozen=True, slots=True)
class BackendDefinition:
    """One physical external service identity."""

    backend_id: str
    display_name: str

    def __post_init__(self) -> None:
        _validate_identifier("backend_id", self.backend_id)
        _require_nonempty("backend_display_name", self.display_name)


@dataclass(frozen=True, slots=True)
class RoutePolicy:
    """Immutable exact-origin and request-shape policy for one route."""

    policy_version: str
    origin: ExactOrigin
    methods: tuple[str, ...]
    paths: tuple[str, ...]
    allowed_query_keys: tuple[str, ...]
    limits: RouteLimits
    policy_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("policy_version", self.policy_version)
        if not isinstance(self.origin, ExactOrigin):
            raise TypeError("origin_must_be_exact_origin")
        _require_tuple("methods", self.methods)
        _require_tuple("paths", self.paths)
        _require_tuple("allowed_query_keys", self.allowed_query_keys)
        if not self.methods or any(
            not isinstance(method, str) or method != method.upper() or not method.isalpha() for method in self.methods
        ):
            raise ValueError("invalid_policy_methods")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("duplicate_policy_method")
        if not self.paths or any(not _valid_path(path) for path in self.paths):
            raise ValueError("invalid_policy_paths")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("duplicate_policy_path")
        if any(not isinstance(key, str) or not _QUERY_NAME_RE.fullmatch(key) for key in self.allowed_query_keys):
            raise ValueError("invalid_policy_query_key")
        if len(set(self.allowed_query_keys)) != len(self.allowed_query_keys):
            raise ValueError("duplicate_policy_query_key")
        if not isinstance(self.limits, RouteLimits):
            raise TypeError("limits_must_be_route_limits")

        computed = canonical_policy_digest(self)
        if self.policy_digest:
            if not _DIGEST_RE.fullmatch(self.policy_digest) or self.policy_digest != computed:
                raise ValueError("policy_digest_mismatch")
        else:
            object.__setattr__(self, "policy_digest", computed)


@dataclass(frozen=True, slots=True)
class SourceRouteReference:
    """A catalog target's ordered reference to one access route."""

    route_id: str
    source_predicate: SourcePredicate | None

    def __post_init__(self) -> None:
        _validate_identifier("route_id", self.route_id)
        if self.source_predicate is not None and not isinstance(self.source_predicate, SourcePredicate):
            raise TypeError("source_predicate_must_be_typed")


@dataclass(frozen=True, slots=True)
class SourceDefinition:
    """One stable, user-facing research target."""

    catalog_source_id: str
    display_name: str
    aliases: tuple[str, ...]
    categories: tuple[str, ...]
    content_types: tuple[str, ...]
    surfaces: tuple[str, ...]
    route_references: tuple[SourceRouteReference, ...]
    site_hosts: tuple[str, ...]
    priority: int
    catalog_version: str

    def __post_init__(self) -> None:
        _validate_identifier("catalog_source_id", self.catalog_source_id)
        _require_nonempty("source_display_name", self.display_name)
        for name in ("aliases", "categories", "content_types", "surfaces", "route_references", "site_hosts"):
            _require_tuple(name, getattr(self, name))
        for alias in self.aliases:
            _validate_identifier("source_alias", alias)
        for name, values in (
            ("categories", self.categories),
            ("content_types", self.content_types),
            ("surfaces", self.surfaces),
        ):
            if not values:
                raise ValueError(f"empty_{name}")
            for value in values:
                _validate_identifier(name, value)
        if not self.route_references:
            raise ValueError("source_requires_route_reference")
        if any(not isinstance(reference, SourceRouteReference) for reference in self.route_references):
            raise TypeError("route_references_must_be_source_route_reference_tuple")
        route_ids = tuple(reference.route_id for reference in self.route_references)
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("duplicate_source_route_reference")
        if any(
            not isinstance(host, str) or host != host.strip().lower() or not _valid_hostname(host)
            for host in self.site_hosts
        ):
            raise ValueError("invalid_descriptive_site_host")
        _require_nonnegative_int("source_priority", self.priority)
        _require_nonempty("catalog_version", self.catalog_version)


@dataclass(frozen=True, slots=True)
class AccessRoute:
    """One versioned way to search one or more catalog targets."""

    route_id: str
    backend_id: str
    adapter_id: str
    route_kind: RouteKind
    query_modes: tuple[QueryMode, ...]
    source_constraint: SourceConstraint
    attribution_basis: str
    credential_requirement: CredentialRequirement
    fallback_order: int
    max_physical_dispatches: int
    adapter_version: str
    policy: RoutePolicy

    def __post_init__(self) -> None:
        _validate_identifier("route_id", self.route_id)
        _validate_identifier("backend_id", self.backend_id)
        _validate_identifier("adapter_id", self.adapter_id)
        _require_enum("route_kind", self.route_kind, RouteKind)
        _require_tuple("query_modes", self.query_modes)
        if not self.query_modes:
            raise ValueError("route_requires_query_mode")
        for mode in self.query_modes:
            _require_enum("query_mode", mode, QueryMode)
        if len(set(self.query_modes)) != len(self.query_modes):
            raise ValueError("duplicate_query_mode")
        _require_enum("source_constraint", self.source_constraint, SourceConstraint)
        _require_nonempty("attribution_basis", self.attribution_basis)
        _require_enum("credential_requirement", self.credential_requirement, CredentialRequirement)
        _require_nonnegative_int("fallback_order", self.fallback_order)
        _require_positive_int("max_physical_dispatches", self.max_physical_dispatches)
        _require_nonempty("adapter_version", self.adapter_version)
        if not isinstance(self.policy, RoutePolicy):
            raise TypeError("policy_must_be_route_policy")
        required_dispatches = (
            len(self.policy.paths)
            + self.policy.limits.max_pages
            - 1
            + self.policy.limits.max_redirects
            + self.policy.limits.max_retries
        )
        if required_dispatches > self.max_physical_dispatches:
            raise ValueError("route_policy_exceeds_physical_dispatches")


@dataclass(frozen=True, slots=True)
class RouteReadiness:
    """Readiness for one route without credential material."""

    route_id: str
    state: ReadinessState
    credential_status: CredentialStatus
    reason: str

    def __post_init__(self) -> None:
        _validate_identifier("route_id", self.route_id)
        _require_enum("readiness_state", self.state, ReadinessState)
        _require_enum("credential_status", self.credential_status, CredentialStatus)
        _require_nonempty("readiness_reason", self.reason)


@dataclass(frozen=True, slots=True)
class ReadinessOverlay:
    """Explicit immutable readiness for a non-production execution mode."""

    overlay_version: str
    execution_mode: ExecutionMode
    routes: tuple[RouteReadiness, ...]

    def __post_init__(self) -> None:
        _require_nonempty("overlay_version", self.overlay_version)
        _require_enum("execution_mode", self.execution_mode, ExecutionMode)
        _require_tuple("routes", self.routes)
        if any(not isinstance(route, RouteReadiness) for route in self.routes):
            raise TypeError("routes_must_be_route_readiness_tuple")
        route_ids = tuple(route.route_id for route in self.routes)
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("duplicate_readiness_route")

    def get(self, route_id: str) -> RouteReadiness | None:
        """Return readiness for one route, if declared."""
        return next((entry for entry in self.routes if entry.route_id == route_id), None)


@dataclass(frozen=True, slots=True)
class DispatchIntent:
    """A description of one potential physical operation."""

    route_id: str
    policy_digest: str
    operation_kind: OperationKind
    method: str
    path: str
    query_pairs: tuple[QueryPair, ...]
    limits: RouteLimits

    def __post_init__(self) -> None:
        _validate_identifier("route_id", self.route_id)
        _validate_digest("policy_digest", self.policy_digest)
        _require_enum("operation_kind", self.operation_kind, OperationKind)
        if not isinstance(self.method, str) or self.method != self.method.upper() or not self.method.isalpha():
            raise ValueError("invalid_intent_method")
        if not _valid_path(self.path):
            raise ValueError("invalid_intent_path")
        _require_tuple("query_pairs", self.query_pairs)
        if any(not isinstance(pair, QueryPair) for pair in self.query_pairs):
            raise TypeError("query_pairs_must_be_typed")
        if not isinstance(self.limits, RouteLimits):
            raise TypeError("limits_must_be_route_limits")


@dataclass(frozen=True, slots=True)
class DispatchAllowance:
    """Worst-case physical work the planner permits for one attempt."""

    physical_dispatches: int
    pages: int
    redirects: int
    retries: int

    def __post_init__(self) -> None:
        for name in ("physical_dispatches", "pages", "redirects", "retries"):
            _require_nonnegative_int(name, getattr(self, name))
        if self.physical_dispatches < self.pages + self.redirects + self.retries:
            raise ValueError("physical_dispatches_below_declared_work")


@dataclass(frozen=True, slots=True)
class RequestedTarget:
    """Logical target attribution retained on physical work."""

    catalog_source_id: str
    selection_reason: str
    source_predicate: SourcePredicate | None

    def __post_init__(self) -> None:
        _validate_identifier("catalog_source_id", self.catalog_source_id)
        _require_nonempty("selection_reason", self.selection_reason)
        if self.source_predicate is not None and not isinstance(self.source_predicate, SourcePredicate):
            raise TypeError("source_predicate_must_be_typed")


@dataclass(frozen=True, slots=True)
class PlannedAttempt:
    """One immutable coalesced unit of planned work."""

    attempt_id: str
    route_id: str
    backend_id: str
    policy_digest: str
    normalized_query: str
    filters: tuple[QueryPair, ...]
    requested_targets: tuple[RequestedTarget, ...]
    fallback_order: int
    intents: tuple[DispatchIntent, ...]
    allowance: DispatchAllowance

    def __post_init__(self) -> None:
        _validate_identifier("attempt_id", self.attempt_id)
        _validate_identifier("route_id", self.route_id)
        _validate_identifier("backend_id", self.backend_id)
        _validate_digest("policy_digest", self.policy_digest)
        _require_nonempty("normalized_query", self.normalized_query)
        for name in ("filters", "requested_targets", "intents"):
            _require_tuple(name, getattr(self, name))
        if not self.requested_targets or not self.intents:
            raise ValueError("attempt_requires_targets_and_intents")
        if any(not isinstance(item, QueryPair) for item in self.filters):
            raise TypeError("filters_must_be_query_pair_tuple")
        if any(not isinstance(item, RequestedTarget) for item in self.requested_targets):
            raise TypeError("requested_targets_must_be_requested_target_tuple")
        if any(not isinstance(item, DispatchIntent) for item in self.intents):
            raise TypeError("intents_must_be_dispatch_intent_tuple")
        _require_nonnegative_int("fallback_order", self.fallback_order)
        if not isinstance(self.allowance, DispatchAllowance):
            raise TypeError("allowance_must_be_dispatch_allowance")
        required_dispatches = (
            len(self.intents) + max(self.allowance.pages - 1, 0) + self.allowance.redirects + self.allowance.retries
        )
        if self.allowance.physical_dispatches < required_dispatches:
            raise ValueError("attempt_work_exceeds_physical_dispatches")


@dataclass(frozen=True, slots=True)
class SkippedTarget:
    """Typed target-level planning outcome without executable work."""

    requested_source_id: str
    route_id: str
    status: SkippedStatus
    code: SkippedCode
    reason: str

    def __post_init__(self) -> None:
        _validate_identifier("requested_source_id", self.requested_source_id)
        _validate_identifier("route_id", self.route_id)
        _require_enum("skipped_status", self.status, SkippedStatus)
        _require_enum("skipped_code", self.code, SkippedCode)
        _require_nonempty("skipped_reason", self.reason)


@dataclass(frozen=True, slots=True)
class PlannedBudgetAllowance:
    """Worst-case plan dimensions compared with independent ceilings."""

    route_attempts: int
    physical_dispatches: int
    max_pages_per_route: int
    redirects: int
    retries: int
    aggregate_wall_time_ms: int
    returned_results: int

    def __post_init__(self) -> None:
        for name in (
            "route_attempts",
            "physical_dispatches",
            "max_pages_per_route",
            "redirects",
            "retries",
            "aggregate_wall_time_ms",
            "returned_results",
        ):
            _require_nonnegative_int(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class DiscoveryPlan:
    """Deterministic effective plan produced without side effects."""

    planner_version: str
    catalog_version: str
    registry_version: str
    readiness_version: str
    execution_mode: ExecutionMode
    normalized_query: str
    filters: tuple[QueryPair, ...]
    attempts: tuple[PlannedAttempt, ...]
    skipped: tuple[SkippedTarget, ...]
    ceilings: BudgetCeilings
    allowance: PlannedBudgetAllowance

    def __post_init__(self) -> None:
        for name in ("planner_version", "catalog_version", "registry_version", "readiness_version"):
            _require_nonempty(name, getattr(self, name))
        _require_enum("execution_mode", self.execution_mode, ExecutionMode)
        _require_nonempty("normalized_query", self.normalized_query)
        for name in ("filters", "attempts", "skipped"):
            _require_tuple(name, getattr(self, name))
        if any(not isinstance(item, QueryPair) for item in self.filters):
            raise TypeError("filters_must_be_query_pair_tuple")
        if any(not isinstance(item, PlannedAttempt) for item in self.attempts):
            raise TypeError("attempts_must_be_planned_attempt_tuple")
        if any(not isinstance(item, SkippedTarget) for item in self.skipped):
            raise TypeError("skipped_must_be_skipped_target_tuple")
        if not isinstance(self.ceilings, BudgetCeilings):
            raise TypeError("ceilings_must_be_budget_ceilings")
        if not isinstance(self.allowance, PlannedBudgetAllowance):
            raise TypeError("allowance_must_be_planned_budget_allowance")


@dataclass(frozen=True, slots=True)
class DiscoveryProvenanceV2:
    """Route-aware provenance that preserves logical target attribution."""

    requested_catalog_source_ids: tuple[str, ...]
    route_id: str
    backend_id: str
    transport_origin: ExactOrigin | None
    reported_document_origin: ExactOrigin | None
    retrieval_observed_origin: ExactOrigin | None
    attribution_basis: str
    catalog_version: str
    adapter_version: str
    policy_digest: str

    def __post_init__(self) -> None:
        _require_tuple("requested_catalog_source_ids", self.requested_catalog_source_ids)
        if not self.requested_catalog_source_ids:
            raise ValueError("provenance_requires_requested_source")
        for source_id in self.requested_catalog_source_ids:
            _validate_identifier("catalog_source_id", source_id)
        _validate_identifier("route_id", self.route_id)
        _validate_identifier("backend_id", self.backend_id)
        for name in (
            "transport_origin",
            "reported_document_origin",
            "retrieval_observed_origin",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ExactOrigin):
                raise TypeError(f"{name}_must_be_exact_origin")
        for name in ("attribution_basis", "catalog_version", "adapter_version"):
            _require_nonempty(name, getattr(self, name))
        _validate_digest("policy_digest", self.policy_digest)


@dataclass(frozen=True, slots=True)
class DiscoveryOutcomeIdentity:
    """Additive route-independent V2 document identity."""

    fingerprint: str
    document_id: str

    def __post_init__(self) -> None:
        _require_nonempty("fingerprint", self.fingerprint)
        expected = stable_document_id_v2(self.fingerprint)
        if self.document_id != expected:
            raise ValueError("document_id_mismatch")

    @classmethod
    def from_fingerprint(cls, fingerprint: str) -> DiscoveryOutcomeIdentity:
        """Build an identity solely from the canonical document fingerprint."""
        return cls(fingerprint=fingerprint, document_id=stable_document_id_v2(fingerprint))


def canonical_policy_digest(policy: RoutePolicy) -> str:
    """Hash only immutable route-policy content."""
    payload = {
        "allowed_query_keys": policy.allowed_query_keys,
        "limits": asdict(policy.limits),
        "methods": policy.methods,
        "origin": asdict(policy.origin),
        "paths": policy.paths,
        "policy_version": policy.policy_version,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def stable_document_id_v2(fingerprint: str) -> str:
    """Build a route- and provider-independent document ID."""
    _require_nonempty("fingerprint", fingerprint)
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32]
    return f"research_document_v2:{digest}"


def evaluate_source_predicate(
    predicate: SourcePredicate,
    record: Mapping[str, Any],
) -> AttributionMatch:
    """Evaluate a typed source predicate with explicit ambiguity."""
    if not isinstance(predicate, SourcePredicate) or not isinstance(record, Mapping):
        raise TypeError("typed_predicate_and_mapping_required")

    value: object = record
    for segment in predicate.field_path:
        if not isinstance(value, Mapping) or segment not in value:
            return AttributionMatch.AMBIGUOUS
        value = value.get(segment, _MISSING)
    if value is _MISSING or value is None or isinstance(value, Mapping):
        return AttributionMatch.AMBIGUOUS

    candidates = value if isinstance(value, (tuple, list, set, frozenset)) else (value,)
    if not candidates:
        return AttributionMatch.AMBIGUOUS
    normalized_candidates: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, (str, int, float, bool)):
            return AttributionMatch.AMBIGUOUS
        normalized_candidates.append(_predicate_text(str(candidate), predicate.case_sensitive))
    expected = tuple(_predicate_text(item, predicate.case_sensitive) for item in predicate.values)

    if predicate.operator is PredicateOperator.EQUALS_ANY:
        matched = any(candidate in expected for candidate in normalized_candidates)
    else:
        matched = any(item in candidate for candidate in normalized_candidates for item in expected)
    return AttributionMatch.MATCH if matched else AttributionMatch.NON_MATCH


def _predicate_text(value: str, case_sensitive: bool) -> str:
    normalized = " ".join(value.split())
    return normalized if case_sensitive else normalized.casefold()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"invalid_{name}")


def _validate_digest(name: str, value: object) -> None:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise ValueError(f"invalid_{name}")


def _require_tuple(name: str, value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name}_must_be_tuple")


def _require_enum(name: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{name}_must_be_{enum_type.__name__}")


def _require_nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid_{name}")


def _require_nonnegative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"invalid_{name}")


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"invalid_{name}")


def _valid_hostname(host: str) -> bool:
    if len(host) > 253 or not host or not host.isascii():
        return False
    labels = host.split(".")
    return all(
        label
        and len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isalnum() or character == "-" for character in label)
        for label in labels
    )


def _valid_path(path: object) -> bool:
    return (
        isinstance(path, str)
        and path.startswith("/")
        and "?" not in path
        and "#" not in path
        and "\\" not in path
        and "\x00" not in path
    )

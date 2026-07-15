"""Immutable, side-effect-free contracts for research discovery V2."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_QUERY_NAME_RE = re.compile(r"[A-Za-z0-9_.\[\]-]+\Z")
_MISSING = object()
MAX_PAGINATION_CURSOR = 2_147_483_647
CREDENTIALED_ROUTE_SKIP_REASON = "credentialed_route_not_authorized_for_foundation"


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


class PathSlotKind(str, Enum):
    """Closed dynamic-segment grammars supported by route policy."""

    DATE = "date"
    UINT = "uint"
    DOI_REGISTRANT = "doi_registrant"
    DOI_SUFFIX = "doi_suffix"


_PATH_SLOT_MAX_CHARS = {
    PathSlotKind.DATE: 10,
    PathSlotKind.UINT: 10,
    PathSlotKind.DOI_REGISTRANT: 12,
    PathSlotKind.DOI_SUFFIX: 128,
}
_MAX_LITERAL_TERMS = 16
_MAX_LITERAL_TERM_CHARS = 64
_MAX_BOUNDED_TEXT_CHARS = 128


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
class JSONBodyPair:
    """One immutable key and bounded scalar for a JSON request body."""

    name: str
    value: str | int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _QUERY_NAME_RE.fullmatch(self.name):
            raise ValueError("invalid_json_body_pair_name")
        if type(self.value) is str:
            if "\x00" in self.value:
                raise ValueError("invalid_json_body_pair_value")
        elif type(self.value) is not int or not 0 <= self.value <= MAX_PAGINATION_CURSOR:
            raise ValueError("invalid_json_body_pair_value")


@dataclass(frozen=True, slots=True)
class DeferredNumericCSVQueryBinding:
    """One bounded numeric-CSV query value produced by an earlier operation."""

    binding_id: str
    query_name: str
    max_items: int
    max_item_chars: int

    def __post_init__(self) -> None:
        _validate_identifier("binding_id", self.binding_id)
        if not isinstance(self.query_name, str) or not _QUERY_NAME_RE.fullmatch(self.query_name):
            raise ValueError("invalid_binding_query_name")
        _require_positive_int("binding_max_items", self.max_items)
        _require_positive_int("binding_max_item_chars", self.max_item_chars)


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
    max_request_body_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
        _require_positive_int("max_pages", self.max_pages)
        _require_nonnegative_int("max_redirects", self.max_redirects)
        _require_nonnegative_int("max_retries", self.max_retries)
        _require_positive_int("timeout_ms", self.timeout_ms)
        _require_positive_int("max_response_bytes", self.max_response_bytes)
        _require_positive_int("max_results", self.max_results)
        _require_positive_int("max_request_body_bytes", self.max_request_body_bytes)


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
        if not isinstance(self.case_sensitive, bool):
            raise TypeError("case_sensitive_must_be_bool")
        if not self.values or any(not isinstance(value, str) for value in self.values):
            raise ValueError("invalid_source_predicate_values")
        normalized_values = tuple(sorted(_predicate_text(value, self.case_sensitive) for value in self.values))
        if any(not value for value in normalized_values):
            raise ValueError("invalid_source_predicate_values")
        if len(set(normalized_values)) != len(normalized_values):
            raise ValueError("duplicate_source_predicate_value")
        object.__setattr__(self, "values", normalized_values)


@dataclass(frozen=True, slots=True)
class BackendDefinition:
    """One physical external service identity."""

    backend_id: str
    display_name: str

    def __post_init__(self) -> None:
        _validate_identifier("backend_id", self.backend_id)
        _require_nonempty("backend_display_name", self.display_name)


@dataclass(frozen=True, slots=True)
class PathSlot:
    """One typed dynamic segment in a closed route path template."""

    kind: PathSlotKind
    max_chars: int

    def __post_init__(self) -> None:
        if type(self.kind) is not PathSlotKind:
            raise TypeError("path_slot_kind_must_be_PathSlotKind")
        if type(self.max_chars) is not int or not 1 <= self.max_chars <= _PATH_SLOT_MAX_CHARS[self.kind]:
            raise ValueError("invalid_path_slot_max_chars")


@dataclass(frozen=True, slots=True)
class PathTemplate:
    """One ordered literal/slot path shape."""

    segments: tuple[str | PathSlot, ...]
    pagination_segment_index: int | None = None

    def __post_init__(self) -> None:
        if type(self.segments) is not tuple:
            raise TypeError("path_template_segments_must_be_tuple")
        if not self.segments:
            raise ValueError("invalid_path_template_segments")
        for segment in self.segments:
            if type(segment) is str:
                if (
                    not segment
                    or not segment.isascii()
                    or any(not "!" <= character <= "~" for character in segment)
                    or segment in {".", ".."}
                    or any(character in "%?#" for character in segment)
                    or "/" in segment
                    or "\\" in segment
                ):
                    raise ValueError("invalid_path_template_literal")
            elif type(segment) is not PathSlot:
                raise TypeError("invalid_path_template_segment_type")
        if self.pagination_segment_index is None:
            return
        if (
            type(self.pagination_segment_index) is not int
            or not 0 <= self.pagination_segment_index < len(self.segments)
            or type(self.segments[self.pagination_segment_index]) is not PathSlot
            or self.segments[self.pagination_segment_index].kind is not PathSlotKind.UINT
        ):
            raise ValueError("invalid_pagination_segment_index")


@dataclass(frozen=True, slots=True)
class ExactQueryValuePolicy:
    """Require one query key to carry one exact value."""

    name: str
    value: str
    required: bool = True

    def __post_init__(self) -> None:
        _validate_query_value_policy_common(self.name, self.required)
        if type(self.value) is not str or not self.value or "\x00" in self.value:
            raise ValueError("invalid_exact_query_value")


@dataclass(frozen=True, slots=True)
class BoundedDecimalQueryValuePolicy:
    """Require one canonical unsigned decimal bounded by a ceiling."""

    name: str
    maximum: int
    required: bool = True

    def __post_init__(self) -> None:
        _validate_query_value_policy_common(self.name, self.required)
        if type(self.maximum) is not int or self.maximum <= 0:
            raise ValueError("invalid_bounded_decimal_maximum")


@dataclass(frozen=True, slots=True)
class LiteralTermsQueryValuePolicy:
    """Require literal Unicode terms followed by an immutable suffix."""

    name: str
    fixed_suffix: str
    max_terms: int
    max_term_chars: int
    required: bool = True

    def __post_init__(self) -> None:
        _validate_query_value_policy_common(self.name, self.required)
        if type(self.fixed_suffix) is not str or not self.fixed_suffix or "\x00" in self.fixed_suffix:
            raise ValueError("invalid_literal_terms_fixed_suffix")
        if type(self.max_terms) is not int or not 1 <= self.max_terms <= _MAX_LITERAL_TERMS:
            raise ValueError("invalid_literal_terms_max_terms")
        if type(self.max_term_chars) is not int or not 1 <= self.max_term_chars <= _MAX_LITERAL_TERM_CHARS:
            raise ValueError("invalid_literal_terms_max_term_chars")


@dataclass(frozen=True, slots=True)
class BoundedTextQueryValuePolicy:
    """Require one optional bounded canonical text value."""

    name: str
    max_chars: int
    required: bool = False

    def __post_init__(self) -> None:
        _validate_query_value_policy_common(self.name, self.required)
        if type(self.max_chars) is not int or not 1 <= self.max_chars <= _MAX_BOUNDED_TEXT_CHARS:
            raise ValueError("invalid_bounded_text_max_chars")


QueryValuePolicy = (
    ExactQueryValuePolicy | BoundedDecimalQueryValuePolicy | LiteralTermsQueryValuePolicy | BoundedTextQueryValuePolicy
)

_QUERY_VALUE_POLICY_TYPES = (
    ExactQueryValuePolicy,
    BoundedDecimalQueryValuePolicy,
    LiteralTermsQueryValuePolicy,
    BoundedTextQueryValuePolicy,
)


def _validate_query_value_policy_common(name: object, required: object) -> None:
    if type(name) is not str or not _QUERY_NAME_RE.fullmatch(name):
        raise ValueError("invalid_query_value_policy_name")
    if type(required) is not bool:
        raise TypeError("query_value_policy_required_must_be_bool")


@dataclass(frozen=True, slots=True)
class RoutePolicy:
    """Immutable exact-origin and request-shape policy for one route."""

    policy_version: str
    origin: ExactOrigin
    methods: tuple[str, ...]
    paths: tuple[str, ...]
    allowed_query_keys: tuple[str, ...]
    limits: RouteLimits
    pagination_query_key: str | None = None
    pagination_json_body_key: str | None = None
    allowed_json_body_keys: tuple[str, ...] = ()
    integer_json_body_keys: tuple[str, ...] = ()
    policy_digest: str = ""
    path_template: PathTemplate | None = None
    query_value_policies: tuple[QueryValuePolicy, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty("policy_version", self.policy_version)
        if not isinstance(self.origin, ExactOrigin):
            raise TypeError("origin_must_be_exact_origin")
        _require_tuple("methods", self.methods)
        _require_tuple("paths", self.paths)
        _require_tuple("allowed_query_keys", self.allowed_query_keys)
        _require_tuple("allowed_json_body_keys", self.allowed_json_body_keys)
        if type(self.integer_json_body_keys) is not tuple:
            raise TypeError("integer_json_body_keys_must_be_tuple")
        if type(self.query_value_policies) is not tuple:
            raise TypeError("query_value_policies_must_be_tuple")
        if not self.methods or any(
            not isinstance(method, str) or method != method.upper() or not method.isalpha() for method in self.methods
        ):
            raise ValueError("invalid_policy_methods")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("duplicate_policy_method")
        if bool(self.paths) == (self.path_template is not None):
            raise ValueError("invalid_policy_path_channel")
        if self.paths:
            if any(not _valid_path(path) for path in self.paths):
                raise ValueError("invalid_policy_paths")
            if len(set(self.paths)) != len(self.paths):
                raise ValueError("duplicate_policy_path")
        elif type(self.path_template) is not PathTemplate:
            raise TypeError("path_template_must_be_PathTemplate")
        if any(type(key) is not str or not _QUERY_NAME_RE.fullmatch(key) for key in self.allowed_query_keys):
            raise ValueError("invalid_policy_query_key")
        if len(set(self.allowed_query_keys)) != len(self.allowed_query_keys):
            raise ValueError("duplicate_policy_query_key")
        if any(type(rule) not in _QUERY_VALUE_POLICY_TYPES for rule in self.query_value_policies):
            raise TypeError("query_value_policy_must_be_closed_typed_rule")
        query_policy_names = tuple(rule.name for rule in self.query_value_policies)
        if len(set(query_policy_names)) != len(query_policy_names):
            raise ValueError("duplicate_query_value_policy_name")
        if self.query_value_policies and set(query_policy_names) != set(self.allowed_query_keys):
            raise ValueError("query_value_policy_key_coverage_mismatch")
        if any(not isinstance(key, str) or not _QUERY_NAME_RE.fullmatch(key) for key in self.allowed_json_body_keys):
            raise ValueError("invalid_policy_json_body_key")
        if len(set(self.allowed_json_body_keys)) != len(self.allowed_json_body_keys):
            raise ValueError("duplicate_policy_json_body_key")
        if any(type(key) is not str for key in self.integer_json_body_keys):
            raise ValueError("invalid_integer_json_body_key")
        if len(set(self.integer_json_body_keys)) != len(self.integer_json_body_keys):
            raise ValueError("duplicate_integer_json_body_key")
        if not set(self.integer_json_body_keys).issubset(self.allowed_json_body_keys):
            raise ValueError("invalid_integer_json_body_key")
        if set(self.allowed_query_keys).intersection(self.allowed_json_body_keys):
            raise ValueError("json_body_key_channel_overlap")
        if self.allowed_json_body_keys and "POST" not in self.methods:
            raise ValueError("json_body_requires_post_method")
        if self.pagination_query_key is not None and (
            not isinstance(self.pagination_query_key, str) or self.pagination_query_key not in self.allowed_query_keys
        ):
            raise ValueError("invalid_pagination_query_key")
        if self.pagination_json_body_key is not None and (
            not isinstance(self.pagination_json_body_key, str)
            or self.pagination_json_body_key not in self.allowed_json_body_keys
        ):
            raise ValueError("invalid_pagination_json_body_key")
        if (
            self.pagination_json_body_key is not None
            and self.pagination_json_body_key not in self.integer_json_body_keys
        ):
            raise ValueError("pagination_json_body_key_must_be_integer")
        path_pagination = self.path_template is not None and self.path_template.pagination_segment_index is not None
        if (
            sum(
                (
                    self.pagination_query_key is not None,
                    self.pagination_json_body_key is not None,
                    path_pagination,
                )
            )
            > 1
        ):
            raise ValueError("multiple_pagination_channels")
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
        initial_intents = 1 if self.policy.path_template is not None else len(self.policy.paths)
        required_dispatches = (
            initial_intents
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
    json_body_pairs: tuple[JSONBodyPair, ...] = ()
    query_bindings: tuple[DeferredNumericCSVQueryBinding, ...] = ()

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
        _require_tuple("json_body_pairs", self.json_body_pairs)
        if any(not isinstance(pair, JSONBodyPair) for pair in self.json_body_pairs):
            raise TypeError("json_body_pairs_must_be_typed")
        body_names = tuple(pair.name for pair in self.json_body_pairs)
        if len(set(body_names)) != len(body_names):
            raise ValueError("duplicate_json_body_pair_name")
        _require_tuple("query_bindings", self.query_bindings)
        if any(not isinstance(binding, DeferredNumericCSVQueryBinding) for binding in self.query_bindings):
            raise TypeError("query_bindings_must_be_typed")
        binding_ids = tuple(binding.binding_id for binding in self.query_bindings)
        binding_names = tuple(binding.query_name for binding in self.query_bindings)
        if len(set(binding_ids)) != len(binding_ids) or len(set(binding_names)) != len(binding_names):
            raise ValueError("duplicate_query_binding")
        if set(binding_names).intersection(pair.name for pair in self.query_pairs):
            raise ValueError("binding_query_conflict")


@dataclass(frozen=True, slots=True)
class DispatchAllowance:
    """Worst-case physical work the planner permits for one dispatch group."""

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
class PlannedLogicalAttempt:
    """One stable target/route attempt retained on physical work."""

    logical_attempt_id: str
    catalog_source_id: str
    selection_reason: str
    source_predicate: SourcePredicate | None

    def __post_init__(self) -> None:
        _validate_identifier("logical_attempt_id", self.logical_attempt_id)
        _validate_identifier("catalog_source_id", self.catalog_source_id)
        _require_nonempty("selection_reason", self.selection_reason)
        if self.source_predicate is not None and not isinstance(self.source_predicate, SourcePredicate):
            raise TypeError("source_predicate_must_be_typed")


@dataclass(frozen=True, slots=True)
class PlannedDispatchGroup:
    """One immutable coalesced unit of physical work."""

    dispatch_group_id: str
    route_id: str
    backend_id: str
    adapter_id: str
    adapter_version: str
    policy_digest: str
    limits: RouteLimits
    normalized_query: str
    filters: tuple[QueryPair, ...]
    logical_attempts: tuple[PlannedLogicalAttempt, ...]
    fallback_order: int
    intents: tuple[DispatchIntent, ...]
    allowance: DispatchAllowance

    def __post_init__(self) -> None:
        _validate_identifier("dispatch_group_id", self.dispatch_group_id)
        _validate_identifier("route_id", self.route_id)
        _validate_identifier("backend_id", self.backend_id)
        _validate_identifier("adapter_id", self.adapter_id)
        _require_nonempty("adapter_version", self.adapter_version)
        _validate_digest("policy_digest", self.policy_digest)
        if not isinstance(self.limits, RouteLimits):
            raise TypeError("limits_must_be_route_limits")
        _require_nonempty("normalized_query", self.normalized_query)
        for name in ("filters", "logical_attempts", "intents"):
            _require_tuple(name, getattr(self, name))
        if not self.logical_attempts or not self.intents:
            raise ValueError("dispatch_group_requires_logical_attempts_and_intents")
        if any(not isinstance(item, QueryPair) for item in self.filters):
            raise TypeError("filters_must_be_query_pair_tuple")
        if any(not isinstance(item, PlannedLogicalAttempt) for item in self.logical_attempts):
            raise TypeError("logical_attempts_must_be_planned_logical_attempt_tuple")
        if any(not isinstance(item, DispatchIntent) for item in self.intents):
            raise TypeError("intents_must_be_dispatch_intent_tuple")
        _require_nonnegative_int("fallback_order", self.fallback_order)
        if not isinstance(self.allowance, DispatchAllowance):
            raise TypeError("allowance_must_be_dispatch_allowance")
        logical_attempt_ids = tuple(item.logical_attempt_id for item in self.logical_attempts)
        if len(set(logical_attempt_ids)) != len(logical_attempt_ids):
            raise ValueError("duplicate_logical_attempt_id")
        catalog_source_ids = tuple(item.catalog_source_id for item in self.logical_attempts)
        if len(set(catalog_source_ids)) != len(catalog_source_ids):
            raise ValueError("duplicate_logical_target")
        for intent in self.intents:
            if intent.route_id != self.route_id:
                raise ValueError("intent_route_mismatch")
            if intent.policy_digest != self.policy_digest:
                raise ValueError("intent_policy_mismatch")
            if intent.limits != self.limits:
                raise ValueError("intent_limits_mismatch")
        if (
            self.allowance.pages != self.limits.max_pages
            or self.allowance.redirects != self.limits.max_redirects
            or self.allowance.retries != self.limits.max_retries
        ):
            raise ValueError("allowance_limits_mismatch")
        required_dispatches = (
            len(self.intents) + max(self.allowance.pages - 1, 0) + self.allowance.redirects + self.allowance.retries
        )
        if self.allowance.physical_dispatches < required_dispatches:
            raise ValueError("dispatch_group_work_exceeds_physical_dispatches")


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
    """Worst-case work plus the global post-executor returned-result cap.

    Physical routes may produce more raw candidates before that global cap is
    applied; raw candidate capacity is not a returned-result dimension.
    """

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
    result_limit: int
    dispatch_groups: tuple[PlannedDispatchGroup, ...]
    skipped: tuple[SkippedTarget, ...]
    ceilings: BudgetCeilings
    plan_digest: str = ""
    allowance: PlannedBudgetAllowance = field(init=False)

    def __post_init__(self) -> None:
        for name in ("planner_version", "catalog_version", "registry_version", "readiness_version"):
            _require_nonempty(name, getattr(self, name))
        _require_enum("execution_mode", self.execution_mode, ExecutionMode)
        _require_nonempty("normalized_query", self.normalized_query)
        _require_positive_int("result_limit", self.result_limit)
        for name in ("filters", "dispatch_groups", "skipped"):
            _require_tuple(name, getattr(self, name))
        if any(not isinstance(item, QueryPair) for item in self.filters):
            raise TypeError("filters_must_be_query_pair_tuple")
        if any(not isinstance(item, PlannedDispatchGroup) for item in self.dispatch_groups):
            raise TypeError("dispatch_groups_must_be_planned_dispatch_group_tuple")
        if any(not isinstance(item, SkippedTarget) for item in self.skipped):
            raise TypeError("skipped_must_be_skipped_target_tuple")
        if not isinstance(self.ceilings, BudgetCeilings):
            raise TypeError("ceilings_must_be_budget_ceilings")
        dispatch_group_ids = tuple(group.dispatch_group_id for group in self.dispatch_groups)
        if len(set(dispatch_group_ids)) != len(dispatch_group_ids):
            raise ValueError("duplicate_dispatch_group_id")
        logical_attempt_ids = tuple(
            attempt.logical_attempt_id for group in self.dispatch_groups for attempt in group.logical_attempts
        )
        if len(set(logical_attempt_ids)) != len(logical_attempt_ids):
            raise ValueError("duplicate_logical_attempt_id")
        for group in self.dispatch_groups:
            if group.normalized_query != self.normalized_query:
                raise ValueError("plan_query_mismatch")
            if group.filters != self.filters:
                raise ValueError("plan_filters_mismatch")
        allowance = derive_plan_allowance(self.dispatch_groups, self.result_limit)
        object.__setattr__(self, "allowance", allowance)
        violation = budget_ceiling_violation(allowance, self.ceilings)
        if violation is not None:
            raise ValueError(f"budget_exceeded:{violation}")
        computed_digest = canonical_plan_digest(self)
        if type(self.plan_digest) is not str or (self.plan_digest and self.plan_digest != computed_digest):
            raise ValueError("plan_digest_mismatch")
        if not self.plan_digest:
            object.__setattr__(self, "plan_digest", computed_digest)


def derive_plan_allowance(
    dispatch_groups: tuple[PlannedDispatchGroup, ...],
    result_limit: int,
) -> PlannedBudgetAllowance:
    """Derive aggregate work and the global returned-result cap."""
    _require_tuple("dispatch_groups", dispatch_groups)
    _require_positive_int("result_limit", result_limit)
    if any(not isinstance(group, PlannedDispatchGroup) for group in dispatch_groups):
        raise TypeError("dispatch_groups_must_be_planned_dispatch_group_tuple")
    return PlannedBudgetAllowance(
        route_attempts=sum(len(group.logical_attempts) for group in dispatch_groups),
        physical_dispatches=sum(group.allowance.physical_dispatches for group in dispatch_groups),
        max_pages_per_route=max((group.allowance.pages for group in dispatch_groups), default=0),
        redirects=sum(group.allowance.redirects for group in dispatch_groups),
        retries=sum(group.allowance.retries for group in dispatch_groups),
        aggregate_wall_time_ms=sum(
            group.limits.timeout_ms * group.allowance.physical_dispatches for group in dispatch_groups
        ),
        returned_results=(
            min(result_limit, sum(group.limits.max_results for group in dispatch_groups)) if dispatch_groups else 0
        ),
    )


def budget_ceiling_violation(
    allowance: PlannedBudgetAllowance,
    ceilings: BudgetCeilings,
) -> str | None:
    """Return the first exceeded independent plan dimension, if any."""
    if not isinstance(allowance, PlannedBudgetAllowance):
        raise TypeError("allowance_must_be_planned_budget_allowance")
    if not isinstance(ceilings, BudgetCeilings):
        raise TypeError("ceilings_must_be_budget_ceilings")
    checks = (
        ("route_attempts", allowance.route_attempts, ceilings.max_route_attempts),
        ("physical_dispatches", allowance.physical_dispatches, ceilings.max_physical_dispatches),
        ("pages_per_route", allowance.max_pages_per_route, ceilings.max_pages_per_route),
        ("redirects", allowance.redirects, ceilings.max_redirects),
        ("retries", allowance.retries, ceilings.max_retries),
        ("wall_time_ms", allowance.aggregate_wall_time_ms, ceilings.max_wall_time_ms),
        ("returned_results", allowance.returned_results, ceilings.max_results),
    )
    return next((name for name, planned, ceiling in checks if planned > ceiling), None)


def canonical_plan_digest(plan: DiscoveryPlan) -> str:
    """Hash compiler-owned plan content, excluding live and derived budget state."""
    if type(plan) is not DiscoveryPlan:
        raise TypeError("plan_must_be_discovery_plan")
    payload = {
        "planner_version": plan.planner_version,
        "catalog_version": plan.catalog_version,
        "registry_version": plan.registry_version,
        "readiness_version": plan.readiness_version,
        "execution_mode": plan.execution_mode,
        "normalized_query": plan.normalized_query,
        "filters": tuple(asdict(item) for item in plan.filters),
        "result_limit": plan.result_limit,
        "dispatch_groups": tuple(asdict(group) for group in plan.dispatch_groups),
        "skipped": tuple(asdict(target) for target in plan.skipped),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


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
        "allowed_json_body_keys": policy.allowed_json_body_keys,
        "limits": asdict(policy.limits),
        "methods": policy.methods,
        "origin": asdict(policy.origin),
        "paths": policy.paths,
        "pagination_query_key": policy.pagination_query_key,
        "policy_version": policy.policy_version,
    }
    if policy.pagination_json_body_key is not None:
        payload["pagination_json_body_key"] = policy.pagination_json_body_key
    if policy.integer_json_body_keys:
        payload["integer_json_body_keys"] = policy.integer_json_body_keys
    if policy.path_template is not None:
        payload["path_template"] = {
            "segments": tuple(
                (
                    {"literal": segment}
                    if type(segment) is str
                    else {
                        "slot": {
                            "kind": segment.kind,
                            "max_chars": segment.max_chars,
                        }
                    }
                )
                for segment in policy.path_template.segments
            ),
            "pagination_segment_index": policy.path_template.pagination_segment_index,
        }
    if policy.query_value_policies:
        payload["query_value_policies"] = tuple(
            {
                "kind": type(rule).__name__,
                **asdict(rule),
            }
            for rule in policy.query_value_policies
        )
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

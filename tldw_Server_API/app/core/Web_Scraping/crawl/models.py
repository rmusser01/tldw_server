"""Immutable internal crawl plans and limits, with lower-layer controls only."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Protocol

from ..contracts.execution import CrawlOutcome
from ..contracts.requests import CrawlRequest
from ..runtime.controls import BudgetSnapshot, ParentControls, WorkHandle, positive_duration, positive_integer

PROFILES = frozenset(
    {
        "legacy_recursive",
        "enhanced_recursive",
        "legacy_sitemap",
        "legacy_url_level",
        "enhanced_sitemap",
        "enhanced_url_level",
        "enhanced_batch",
    }
)
SAFE_OPTIONS = frozenset(
    {
        "max_pages",
        "max_depth",
        "include_external",
        "score_threshold",
        "crawl_strategy",
        "delay",
        "resume_file",
        "allow_llm_extraction",
        "method",
        "priority",
        "summarize",
        "max_urls",
        "level",
        "allowed_domains",
        "blocked_domains",
        "respect_robots",
        "enable_keyword_scorer",
        "keywords",
        "enable_domain_map",
        "domain_map",
    }
)
_SOURCES = frozenset({"default", "request", "config", "supplied", "supplied:config", "supplied:default"})


def _freeze_option(name: str, value: Any) -> Any:
    """Freeze only supported option shapes, not arbitrary credential subtrees."""
    if name == "domain_map" and isinstance(value, Mapping):
        if any(type(key) is not str or type(item) not in (int, float, str, bool) for key, item in value.items()):
            raise ValueError("domain_map option must map domain strings to numeric weights")
        try:
            return MappingProxyType({key: float(item) for key, item in value.items()})
        except (ValueError, OverflowError):
            raise ValueError("domain_map option weights must be numeric scalars") from None
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("crawl option must be a scalar or a domain-weight mapping")


@dataclass(frozen=True)
class CrawlLimits:
    """Trusted composition-time finite allowances; never client configuration."""

    page_attempts: int = 1_000
    network_requests: int = 20_000
    browser_attempts: int = 1_000
    active_probes: int = 2_000
    transfer_bytes: int = 1_073_741_824
    discovered_candidates: int = 50_000
    max_url_bytes: int = 65_536
    discovered_url_bytes: int = 67_108_864
    frontier_entries: int = 10_000
    execution_seconds: float = 1_800

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            validator = positive_duration if descriptor.name == "execution_seconds" else positive_integer
            validator(getattr(self, descriptor.name), descriptor.name)


@dataclass(frozen=True)
class CheckpointLimits:
    """Independent load/write bounds; larger crawl injection does not raise these."""

    max_file_bytes: int = 134_217_728
    max_candidates: int = 50_000
    max_frontier: int = 10_000
    max_results: int = 1_000

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            positive_integer(getattr(self, descriptor.name), descriptor.name)

    def validate_for(self, limits: CrawlLimits) -> None:
        """Require explicit checkpoint capacity for all effective crawl dimensions."""
        if (
            self.max_candidates < limits.discovered_candidates
            or self.max_frontier < limits.frontier_entries
            or self.max_results < limits.page_attempts
            or self.max_file_bytes < limits.discovered_url_bytes
            or self.max_file_bytes < limits.max_url_bytes
        ):
            raise ValueError("checkpoint ceilings are smaller than effective crawl limits")


@dataclass(frozen=True)
class CrawlPlan:
    """Safe behavior snapshot; credentials and executable state travel separately."""

    request: CrawlRequest
    profile: str
    options: Mapping[str, Any]
    provenance: Mapping[str, str]
    limits: CrawlLimits

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError("unknown crawl profile")
        options = {key: _freeze_option(key, value) for key, value in self.options.items() if key in SAFE_OPTIONS}
        provenance = {key: value for key, value in self.provenance.items() if key in options and value in _SOURCES}
        object.__setattr__(self, "options", MappingProxyType(options))
        object.__setattr__(self, "provenance", MappingProxyType(provenance))
        object.__setattr__(self, "request", replace(self.request, context=replace(self.request.context, metadata={})))


class CrawlControls(ParentControls, Protocol):
    """Engine-facing controls: only orchestration constructs the concrete context."""

    def cancel(self) -> None: ...
    def attach(self, handle: WorkHandle[Any]) -> None: ...
    def snapshot(self) -> BudgetSnapshot: ...
    def restore_budget(self, snapshot: BudgetSnapshot, *, accounting_complete: bool) -> None: ...
    def note_successful_result(self) -> None: ...
    def publish(self, outcome: CrawlOutcome) -> CrawlOutcome: ...


@dataclass(frozen=True)
class CrawlHooks:
    """Private live callbacks, never part of a serialized plan or checkpoint."""

    url_filter: Callable[[str], bool] | None = field(default=None, repr=False)
    progress_callback: Callable[[str], None] | None = field(default=None, repr=False)
    on_progress: Callable[[Mapping[str, Any]], None] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class FrontierEntry:
    """One admitted URL identity, its traversal position and priority."""

    url: str
    parent_url: str | None
    depth: int
    sequence: int
    score: float


@dataclass(frozen=True)
class FrontierSnapshot:
    """Ordered dispositions and committed article data, without acquisition HTML."""

    queued: tuple[FrontierEntry, ...] = ()
    inflight: tuple[FrontierEntry, ...] = ()
    completed: tuple[FrontierEntry, ...] = ()
    results: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class CheckpointScope:
    """Trusted owner/crawl namespace; never authorized from stored file contents."""

    owner: str
    namespace: str
    trusted_local_path: bool = False


@dataclass(frozen=True)
class CheckpointState:
    """Credential-free continuation contract; Task 7 supplies bounded storage."""

    scope: CheckpointScope
    plan: CrawlPlan
    frontier: FrontierSnapshot
    budget: BudgetSnapshot
    stop_reason: str | None = None
    accounting_complete: bool = True
    schema_version: int = 1


class CheckpointPort(Protocol):
    """Bound trusted checkpoint ownership, independent of storage implementation."""

    @property
    def scope(self) -> CheckpointScope: ...
    def load(self, *, plan: CrawlPlan, scope: CheckpointScope) -> CheckpointState | None: ...
    def save(self, state: CheckpointState) -> None: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...

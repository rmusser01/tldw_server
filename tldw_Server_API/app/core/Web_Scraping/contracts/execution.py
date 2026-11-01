"""Internal discovery and terminal reports, separate from public constructors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

COMPLETION_REASONS = frozenset({"frontier_exhausted", "page_cap_reached"})
INTERRUPTION_REASONS = frozenset(
    {
        "deadline_exhausted",
        "budget_exhausted",
        "transport_unavailable",
        "policy_denied",
        "execution_failed",
        "checkpoint_failed",
        "queue_overloaded",
        "queue_stopping",
        "job_payload_too_large",
        "cancelled",
    }
)
_PRIVATE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "proxy_authorization",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "cookies",
        "custom_cookies",
        "cookie",
        "headers",
        "custom_headers",
        "proxies",
        "config",
        "execution",
        "credentials",
        "_discovered_links",
    }
)
_NORMALIZED_PRIVATE_KEYS = frozenset("".join(char for char in key if char.isalnum()) for key in _PRIVATE_KEYS)
_CREDENTIAL_KEY_SUFFIXES = (
    "apikey",
    "clientsecret",
    "authorization",
    "password",
    "accesstoken",
    "refreshtoken",
    "authtoken",
    "sessiontoken",
    "bearertoken",
    "accesskey",
    "accesskeyid",
    "privatekey",
    "setcookie",
)


def _private_article_key(key: str) -> bool:
    """Recognize credential aliases/provider prefixes without scanning article prose."""
    normalized = "".join(char for char in key.lower() if char.isalnum())
    return normalized in _NORMALIZED_PRIVATE_KEYS or normalized.endswith(_CREDENTIAL_KEY_SUFFIXES)


def _safe_article(value: Any) -> Any:
    """Copy JSON-like article data, retaining non-secret extra fields as plain dicts."""
    if isinstance(value, Mapping):
        return {
            key: _safe_article(item)
            for key, item in value.items()
            if type(key) is str and not _private_article_key(key)
        }
    if isinstance(value, list | tuple):
        return [_safe_article(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    return None


def successful_article(article: Mapping[str, Any]) -> bool:
    """Failure diagnostics alone never count as successfully extracted article data."""
    return (
        bool(article.get("content")) and not article.get("error") and article.get("extraction_successful", True) is True
    )


@dataclass(frozen=True)
class DiscoveredLink:
    """A discovered target and the document location from which it was resolved."""

    url: str
    parent_url: str


@dataclass(frozen=True)
class DiscoveryResult:
    """Acquisition locations and links, not a second unauthenticated fetch."""

    request_url: str
    final_url: str
    links: tuple[DiscoveredLink, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "links", tuple(self.links))


@dataclass(frozen=True)
class PageData:
    """Internal page article plus discovery sidecar (never public history HTML)."""

    article: dict[str, Any]
    discovery: DiscoveryResult


@dataclass(frozen=True)
class CrawlOutcome:
    """Validated terminal value; successful completion never means partial data."""

    state: Literal["completed", "partial", "failed", "cancelled"]
    stop_reason: str
    has_partial_results: bool

    def __post_init__(self) -> None:
        valid = type(self.has_partial_results) is bool
        if self.state == "completed":
            valid &= self.stop_reason in COMPLETION_REASONS and not self.has_partial_results
        elif self.state == "cancelled":
            valid &= self.stop_reason == "cancelled"
        elif self.state in {"partial", "failed"}:
            valid &= self.stop_reason in INTERRUPTION_REASONS - {"cancelled"}
            valid &= self.has_partial_results == (self.state == "partial")
        else:
            valid = False
        if not valid:
            raise ValueError("incoherent crawl outcome")


@dataclass(frozen=True)
class CrawlReport:
    """Safe successful article data and outcome; failed entries become diagnostics."""

    articles: tuple[dict[str, Any], ...]
    outcome: CrawlOutcome
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        articles = tuple(_safe_article(article) for article in self.articles if successful_article(article))
        if self.outcome.has_partial_results != (bool(articles) and self.outcome.state != "completed"):
            raise ValueError("outcome does not match successful partial article data")
        object.__setattr__(self, "articles", articles)
        object.__setattr__(
            self,
            "diagnostics",
            tuple(reason if reason in INTERRUPTION_REASONS else "execution_failed" for reason in self.diagnostics),
        )

    @classmethod
    def interrupted(
        cls, articles: Iterable[dict[str, Any]], stop_reason: str, diagnostics: tuple[str, ...] = ()
    ) -> CrawlReport:
        """Classify interruption without treating failed page attempts as articles."""
        values = tuple(articles)
        successful = tuple(article for article in values if successful_article(article))
        state = "cancelled" if stop_reason == "cancelled" else ("partial" if successful else "failed")
        failures = ("execution_failed",) if len(successful) != len(values) else ()
        return cls(successful, CrawlOutcome(state, stop_reason, bool(successful)), diagnostics + failures)


class CrawlInterrupted(Exception):
    """Non-normal exit for adapters without an established structured failure shape."""

    def __init__(self, report: CrawlReport) -> None:
        if report.outcome.state == "completed":
            raise ValueError("completed crawl is not an interruption")
        self.report = report
        super().__init__(report.outcome.stop_reason)

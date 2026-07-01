from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

PolicyStatus = Literal["allowed", "approval_required", "denied"]
FetchResultStatus = Literal["fetched", "approval_required", "denied", "failed"]
IngestStatus = Literal[
    "created",
    "updated",
    "unchanged",
    "approval_required",
    "denied",
    "failed",
    "capability_disabled",
]


@dataclass(frozen=True)
class NormalizedURL:
    scheme: str
    host: str
    port: int | None
    path: str
    decoded_path: str
    canonical_url: str
    redacted_url: str


@dataclass(frozen=True)
class SourceDecision:
    status: PolicyStatus
    reason: str
    safe_argument_hash: str
    redacted_url: str | None = None
    normalized_url: NormalizedURL | None = None
    matched_rule: str | None = None


@dataclass(frozen=True)
class URLRequest:
    normalized_url: NormalizedURL
    headers: Mapping[str, str] = field(default_factory=dict)
    max_body_bytes: int | None = None
    target: str | None = None

    @property
    def url(self) -> str:
        return self.normalized_url.canonical_url

    @property
    def redacted_url(self) -> str:
        return self.normalized_url.redacted_url


@dataclass(frozen=True)
class ResolvedAddress:
    host: str
    ip: str
    port: int
    family: int | None = None
    is_private: bool = False

    @property
    def address(self) -> str:
        return self.ip


@dataclass(frozen=True)
class FetchResponse:
    status_code: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body_chunks: Sequence[bytes] = ()


@dataclass(frozen=True)
class RedirectHop:
    from_url: str
    to_url: str
    status_code: int


@dataclass(frozen=True)
class FetchResult:
    status: FetchResultStatus
    reason: str
    final_url: str | None = None
    status_code: int | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    redirects: Sequence[RedirectHop] = ()
    warnings: Sequence[str] = ()
    safe_argument_hash: str | None = None

    @property
    def reason_code(self) -> str:
        return self.reason


class Resolver(Protocol):
    def resolve(self, host: str, port: int) -> Iterable[ResolvedAddress]: ...


class Transport(Protocol):
    dials_validated_address: bool

    def request(
        self,
        *,
        address: ResolvedAddress,
        request: URLRequest,
        timeout_seconds: float,
    ) -> FetchResponse: ...

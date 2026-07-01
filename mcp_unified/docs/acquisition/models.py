from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

PolicyStatus = Literal["allowed", "approval_required", "denied"]
FetchResultStatus = Literal["fetched", "blocked", "error"]
IngestStatus = Literal["created", "updated", "skipped", "failed"]


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
    url: str
    redacted_url: str
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedAddress:
    host: str
    address: str
    family: int | None = None
    is_private: bool = False


@dataclass(frozen=True)
class FetchResponse:
    status: FetchResultStatus
    url: str
    redacted_url: str
    status_code: int | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    reason: str | None = None


class Resolver(Protocol):
    def resolve(self, host: str) -> Sequence[ResolvedAddress]:
        ...


class Transport(Protocol):
    def fetch(self, request: URLRequest) -> FetchResponse:
        ...

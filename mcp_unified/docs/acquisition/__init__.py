from __future__ import annotations

from .models import (
    FetchResponse,
    FetchResult,
    FetchResultStatus,
    IngestStatus,
    NormalizedURL,
    PolicyStatus,
    RedirectHop,
    Resolver,
    ResolvedAddress,
    SourceDecision,
    Transport,
    URLRequest,
)
from .policy import (
    DomainRule,
    SourcePolicy,
    URLPolicyError,
    URLPrefixRule,
    has_url_credentials,
    normalize_url,
    safe_argument_hash,
)
from .resolver import StdlibResolver, is_unsafe_egress_ip
from .service import DocsAcquisitionService

__all__ = [
    "DocsAcquisitionService",
    "DomainRule",
    "FetchResponse",
    "FetchResult",
    "FetchResultStatus",
    "IngestStatus",
    "NormalizedURL",
    "PolicyStatus",
    "RedirectHop",
    "Resolver",
    "ResolvedAddress",
    "SourceDecision",
    "SourcePolicy",
    "StdlibResolver",
    "Transport",
    "URLPolicyError",
    "URLPrefixRule",
    "URLRequest",
    "has_url_credentials",
    "is_unsafe_egress_ip",
    "normalize_url",
    "safe_argument_hash",
]

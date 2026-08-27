from __future__ import annotations

"""Process-local guard callbacks for one canonical Notes product mutation."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..models import SyncDomain

GuardBefore = Callable[[Any], None]
GuardAfter = Callable[[Any, str], None]


class GuardedProductMutationIdentityError(ValueError):
    """A process-local guard does not identify exactly one materialized object."""


@dataclass(frozen=True, slots=True)
class GuardedProductMutation:
    """Trusted callbacks bound to one Sync domain/object product transaction."""

    expected_domain: SyncDomain
    expected_object_id: str
    before: GuardBefore
    after: GuardAfter

    def __post_init__(self) -> None:
        if self.expected_domain not in {
            "notes.link",
            "notes.keyword",
            "notes.keyword_link",
        }:
            raise GuardedProductMutationIdentityError("Guarded product mutation domain is unsupported")
        if not self.expected_object_id.strip():
            raise GuardedProductMutationIdentityError("Guarded product mutation object identity is empty")
        if not callable(self.before) or not callable(self.after):
            raise TypeError("Guarded product mutation callbacks must be callable")

    def matches(self, domain: SyncDomain, object_id: str) -> bool:
        """Return whether the guard is bound to this exact Sync identity."""

        return domain == self.expected_domain and object_id == self.expected_object_id

    def require_identity(self, domain: SyncDomain, object_id: str) -> None:
        """Reject any attempt to apply this guard to another product object."""

        if not self.matches(domain, object_id):
            raise GuardedProductMutationIdentityError(
                "Guarded product mutation identity does not match the Sync envelope"
            )


__all__ = [
    "GuardAfter",
    "GuardBefore",
    "GuardedProductMutation",
    "GuardedProductMutationIdentityError",
]

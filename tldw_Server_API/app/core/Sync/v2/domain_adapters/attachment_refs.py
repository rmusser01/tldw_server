"""Strict Sync v2 adapter for canonical Notes attachment references."""

from __future__ import annotations

from dataclasses import dataclass

from ..adapters import AttachmentRefAdapter


@dataclass(slots=True)
class AttachmentRefDomainAdapter(AttachmentRefAdapter):
    """Dedicated domain adapter for the versioned attachment-ref contract."""


__all__ = ["AttachmentRefDomainAdapter"]

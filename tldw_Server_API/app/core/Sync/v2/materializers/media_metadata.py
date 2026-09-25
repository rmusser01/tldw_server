from __future__ import annotations

"""Materialize metadata-only media Sync domains into restoreable object state."""

from ..models import SyncDomain
from .metadata_only import MetadataOnlyMaterializer


def MediaMetadataMaterializer(  # noqa: N802 - preserves the original class call sites
    domain: SyncDomain = "media.item",
) -> MetadataOnlyMaterializer:
    """Record accepted media metadata envelopes as Sync object state.

    This was a 187-line class byte-identical to ``source_cache.py`` apart from its
    docstrings, error-code prefix and message strings. It is now a construction of the
    shared :class:`MetadataOnlyMaterializer`; every emitted code, metadata key and
    message is unchanged. Registered for ``media.item``, ``media.keyword`` and
    ``media.keyword_link`` in ``factory.py``.

    Named as a function rather than a class deliberately: ``SyncMaterializer`` is a
    structural :class:`typing.Protocol`, nothing performs an ``isinstance`` check, and
    keeping the CapWords name leaves every call site untouched.
    """
    return MetadataOnlyMaterializer(
        domain=domain,
        code_prefix="media_metadata",
        label="Media metadata",
        lower_label="media metadata",
        noun="object",
    )


__all__ = ["MediaMetadataMaterializer"]

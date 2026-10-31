from __future__ import annotations

"""Materialize Sync v2 source-cache entries into restoreable object state."""

from ..models import SyncDomain
from .metadata_only import MetadataOnlyMaterializer


def SourceCacheMaterializer(  # noqa: N802 - preserves the original class call sites
    domain: SyncDomain = "source_cache.entry",
) -> MetadataOnlyMaterializer:
    """Record accepted ``source_cache.entry`` envelopes as Sync object state.

    See :func:`..media_metadata.MediaMetadataMaterializer` for why this is a function.
    Every emitted error code, metadata key and message is unchanged from the previous
    187-line implementation.
    """
    return MetadataOnlyMaterializer(
        domain=domain,
        code_prefix="source_cache",
        label="source_cache.entry",
        lower_label="source_cache.entry",
        noun="entry",
    )


__all__ = ["SourceCacheMaterializer"]

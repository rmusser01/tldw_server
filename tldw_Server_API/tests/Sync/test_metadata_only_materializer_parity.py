"""Characterisation test for the two metadata-only materializers.

media_metadata.py and source_cache.py are the same 187 lines; every difference is a
docstring, an error-code prefix, or a message string. This pins the externally visible
behaviour of BOTH -- every error code and message, on every failure path -- so the two
can be collapsed into one parameterised materializer without changing what a client sees.

source_cache has no dedicated materializer suite of its own, which is exactly the risk:
a rule fixed on the media side and missed on the other would not be caught.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sync.v2.materializers.media_metadata import (
    MediaMetadataMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.materializers.source_cache import (
    SourceCacheMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelope, SyncObjectState


class _FakeStore:
    def __init__(self, state: SyncObjectState | None = None) -> None:
        self._state = state
        self.marks: list[dict] = []
        self.upserts: list[SyncObjectState] = []

    def get_object_state(self, dataset_id, domain, object_id):
        return self._state

    def mark_envelope_apply_status(self, cursor, **kwargs):
        self.marks.append({"cursor": cursor, **kwargs})

    def upsert_object_state(self, state):
        self.upserts.append(state)


def _env(domain: str, *, operation="upsert", payload_hash="h1", cursor=10):
    return SyncEnvelope(
        dataset_id="ds-1",
        client_envelope_id="cid-1",
        domain=domain,
        operation=operation,
        object_id="obj-1",
        payload_hash=payload_hash,
        server_cursor=cursor,
    )


CASES = [
    pytest.param(MediaMetadataMaterializer(), "media.item", "media_metadata", "Media metadata", "object", id="media"),
    pytest.param(SourceCacheMaterializer(), "source_cache.entry", "source_cache", "source_cache.entry", "entry", id="source_cache"),
]


@pytest.mark.parametrize("mat,domain,prefix,label,noun", CASES)
def test_missing_payload_hash(mat, domain, prefix, label, noun) -> None:
    store = _FakeStore()
    res = mat.apply(_env(domain, payload_hash=""), store=store)
    assert res.status == "failed"
    assert res.error_code == f"{prefix}_projection_failed"
    assert res.message == f"{label} envelopes require payload_hash"
    assert store.marks[0]["apply_error_code"] == f"{prefix}_projection_failed"
    assert store.marks[0]["apply_error_message"] == f"{label} envelopes require payload_hash"


# NOTE: the `envelope.server_cursor is None` branch in both materializers is defensive
# and unreachable through the model -- SyncEnvelope.__post_init__ raises
# "server_cursor is required" -- so it cannot be characterised here. It is preserved
# verbatim by the consolidation rather than dropped.


@pytest.mark.parametrize("mat,domain,prefix,label,noun", CASES)
def test_upsert_over_tombstone_conflicts(mat, domain, prefix, label, noun) -> None:
    state = SyncObjectState(
        dataset_id="ds-1", domain=domain, object_id="obj-1",
        object_revision=3, object_hash="old", latest_server_cursor=5, deleted=True,
    )
    res = mat.apply(_env(domain), store=_FakeStore(state))
    assert res.status == "conflict"
    assert res.message == f"{label} upsert cannot resurrect a tombstoned {noun}"


@pytest.mark.parametrize("mat,domain,prefix,label,noun", CASES)
def test_reused_object_id_with_different_hash_conflicts(mat, domain, prefix, label, noun) -> None:
    state = SyncObjectState(
        dataset_id="ds-1", domain=domain, object_id="obj-1",
        object_revision=3, object_hash="different", latest_server_cursor=5, deleted=False,
    )
    res = mat.apply(_env(domain), store=_FakeStore(state))
    assert res.status == "conflict"
    assert res.message == f"{label} stable object ID was reused with a different payload hash"


@pytest.mark.parametrize("mat,domain,prefix,label,noun", CASES)
def test_conflict_results_carry_the_domain_scoped_metadata_key(mat, domain, prefix, label, noun) -> None:
    """The metadata dict key is client-visible and domain-scoped: {prefix}_object_id."""
    state = SyncObjectState(
        dataset_id="ds-1", domain=domain, object_id="obj-1",
        object_revision=3, object_hash="different", latest_server_cursor=5, deleted=False,
    )
    res = mat.apply(_env(domain), store=_FakeStore(state))
    assert res.conflict_type == f"{prefix}_hash_mismatch"
    assert res.metadata[f"{prefix}_object_id"] == "obj-1"
    assert res.metadata["incoming_payload_hash"] == "h1"
    assert res.metadata["server_object_hash"] == "different"
    assert res.metadata["server_object_revision"] == 3
    assert res.metadata["server_cursor"] == 5
    assert res.metadata["server_deleted"] is False


@pytest.mark.parametrize("mat,domain,prefix,label,noun", CASES)
def test_tombstone_conflict_metadata(mat, domain, prefix, label, noun) -> None:
    state = SyncObjectState(
        dataset_id="ds-1", domain=domain, object_id="obj-1",
        object_revision=3, object_hash="old", latest_server_cursor=5, deleted=True,
    )
    res = mat.apply(_env(domain), store=_FakeStore(state))
    assert res.conflict_type == f"{prefix}_tombstoned"
    assert res.metadata[f"{prefix}_object_id"] == "obj-1"


@pytest.mark.parametrize("mat,domain,prefix,label,noun", CASES)
def test_tombstone_operation_records_state(mat, domain, prefix, label, noun) -> None:
    store = _FakeStore()
    res = mat.apply(_env(domain, operation="tombstone"), store=store)
    assert res.status == "applied"
    assert store.upserts and store.upserts[0].deleted is True


@pytest.mark.parametrize("mat,domain,prefix,label,noun", CASES)
def test_unsupported_operation(mat, domain, prefix, label, noun) -> None:
    lower = "media metadata" if prefix == "media_metadata" else "source_cache.entry"
    res = mat.apply(_env(domain, operation="patch"), store=_FakeStore())
    assert res.status == "failed"
    assert res.error_code == f"{prefix}_projection_failed"
    assert res.message == f"Unsupported {lower} operation: patch"


@pytest.mark.parametrize("mat,domain,prefix,label,noun", CASES)
def test_idempotent_upsert_same_hash_applies(mat, domain, prefix, label, noun) -> None:
    state = SyncObjectState(
        dataset_id="ds-1", domain=domain, object_id="obj-1",
        object_revision=3, object_hash="h1", latest_server_cursor=5, deleted=False,
    )
    res = mat.apply(_env(domain), store=_FakeStore(state))
    assert res.status == "applied"


@pytest.mark.parametrize("mat,domain,prefix,label,noun", CASES)
def test_foreign_domain_is_skipped(mat, domain, prefix, label, noun) -> None:
    assert mat.apply(_env("not.my.domain"), store=_FakeStore()).status == "skipped"


def test_media_materializer_serves_all_three_registered_domains() -> None:
    """factory.py registers it for media.item, media.keyword and media.keyword_link."""
    for d in ("media.item", "media.keyword", "media.keyword_link"):
        mat = MediaMetadataMaterializer(domain=d)
        assert mat.apply(_env(d, payload_hash=""), store=_FakeStore()).error_code == (
            "media_metadata_projection_failed"
        )

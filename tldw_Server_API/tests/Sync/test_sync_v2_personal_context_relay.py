"""Regression coverage for ordered Personal Context publication relay."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import pytest

from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    PublicationSourceBatch,
    PublicationSourceRow,
)


def test_relay_exports_a_result_type() -> None:
    """The relay boundary is available independently of HTTP transport."""

    from tldw_Server_API.app.core.Sync.v2.personal_context_relay import (
        PersonalContextRelayResult,
    )

    assert PersonalContextRelayResult.__name__ == "PersonalContextRelayResult"


class _Publications:
    def __init__(self) -> None:
        self.rows = [
            _row(0, "semantic"),
            _row(1, "manifest"),
        ]
        self.acknowledged: list[int] = []
        self.attention = False
        self.failed_after: int | None = None

    @contextmanager
    def profile_lease(self, _profile_id: str):
        yield type("Lease", (), {"owner_token": "lease-token"})()

    def renew_lease(self, _lease: object) -> bool:
        return True

    def row_is_current(self, _row: PublicationSourceRow, _lease: object) -> bool:
        return True

    def earliest_nonterminal_batch(self, _profile_id: str) -> PublicationSourceBatch:
        return PublicationSourceBatch("profile-a", 1, "batch-a", tuple(self.rows))

    def acknowledge_row(self, row: PublicationSourceRow, *, server_cursor: int, lease: object) -> None:
        del lease
        if self.failed_after is not None and row.batch_ordinal == self.failed_after:
            raise RuntimeError("injected interruption")
        self.acknowledged.append(server_cursor)
        self.rows[row.batch_ordinal] = replace(row, row_state="acknowledged", sync_server_cursor=server_cursor)

    def complete_if_acknowledged(self, _batch: PublicationSourceBatch) -> bool:
        return all(row.row_state == "acknowledged" for row in self.rows)

    def mark_attention(self, _batch: PublicationSourceBatch) -> None:
        self.attention = True


def _row(ordinal: int, role: str) -> PublicationSourceRow:
    return PublicationSourceRow(
        profile_id="profile-a", profile_publication_sequence=1, publication_batch_id="batch-a",
        batch_ordinal=ordinal, batch_size=2, purge_generation=0, role=role,  # type: ignore[arg-type]
        object_id=f"object-{ordinal}", version_id=f"version-{ordinal}", operation="upsert",
        deterministic_envelope_id=f"envelope-{ordinal}", domain="personal_context.record",
        canonical=b'{}', integrity_tag="hmac-sha256-v1:" + "a" * 64,
        sync_server_cursor=None, row_state="pending",
    )


def test_relay_never_stages_manifest_before_semantic_siblings() -> None:
    from tldw_Server_API.app.core.Sync.v2.personal_context_relay import PersonalContextRelay

    publications = _Publications()
    staged: list[str] = []
    relay = PersonalContextRelay(
        publications=publications,
        stage_authority=lambda row, _dataset, _user: staged.append(row.role) or row.batch_ordinal + 1,
    )

    publications.failed_after = 0
    with pytest.raises(RuntimeError, match="injected interruption"):
        relay.relay_profile(user_id="user-a", profile_id="profile-a", dataset_id="dataset-a", after_server_cursor=None)
    assert staged == ["semantic"]
    publications.failed_after = None
    relay.relay_profile(user_id="user-a", profile_id="profile-a", dataset_id="dataset-a", after_server_cursor=None)
    assert staged == ["semantic", "semantic", "manifest"]


def test_relay_poison_blocks_the_earliest_batch_without_error_body() -> None:
    from tldw_Server_API.app.core.Sync.v2.personal_context_relay import PersonalContextRelay

    publications = _Publications()
    relay = PersonalContextRelay(
        publications=publications,
        stage_authority=lambda *_args: (_ for _ in ()).throw(RuntimeError("secret")),
    )

    result = relay.relay_profile(
        user_id="user-a", profile_id="profile-a", dataset_id="dataset-a", after_server_cursor=None
    )

    assert result.continuation == "relay_poisoned"
    assert publications.attention is True

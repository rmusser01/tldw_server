from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import pytest
from tldw_profile_core import (
    AgentVisibility,
    ProfileControls,
    ProfileManifest,
    ProfileProposal,
    ProfileRecord,
    ProfileScope,
    ProposalState,
    ScopeKind,
    SyncMode,
    canonical_bytes,
)
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Personalization.personal_context_crypto import EnvelopeCipher
from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    CanonicalApplyReceipt,
    IngressIdentity,
    PersonalContextPublicationJournal,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ProfileKeyMaterial,
)
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    PersonalContextService,
)
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    encoded_master_key,
    proposal,
)

REPLAY_ERROR = "ingress identity reused with a different payload"


def _digest(canonical: bytes) -> str:
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _record(service: PersonalContextService, record_id: str) -> ProfileRecord:
    scope = service.list_scopes()[0]
    return service.build_manual_record(
        scope_id=scope.scope_id,
        payload={
            "kind": "preference",
            "polarity": "like",
            "subject": record_id,
            "value": "concise",
        },
        semantic_key={"namespace": "preference", "subject": record_id},
        controls=ProfileControls(
            sync_mode=SyncMode.SYNCABLE,
            agent_visibility=AgentVisibility.AGENT_VISIBLE,
        ),
    ).model_copy(update={"record_id": record_id})


@dataclass
class LegacyReceiptHarness:
    database: PersonalizationDB
    service: PersonalContextService
    keys: ProfileKeyMaterial
    profile_id: str
    identity: IngressIdentity
    receipt: CanonicalApplyReceipt
    case: str
    domain: str
    canonical: bytes
    source_role: str
    expected_object_id: str

    @property
    def journal(self) -> PersonalContextPublicationJournal:
        return PersonalContextPublicationJournal(self.keys)

    def replay(
        self,
        *,
        journal: PersonalContextPublicationJournal | None = None,
        identity: IngressIdentity | None = None,
    ) -> CanonicalApplyReceipt | None:
        with self.database.transaction(immediate=True) as connection:
            return (journal or self.journal).read_ingress_receipt(
                connection,
                identity or self.identity,
            )

    def stored_wire_version(self) -> str:
        return str(self.stored_receipt()["wire_entity_version"])

    def stored_receipt(self) -> dict[str, object]:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM personal_context_ingress_receipts "
                "WHERE dataset_id = ? AND device_id = ? AND client_envelope_id = ?",
                (
                    self.identity.dataset_id,
                    self.identity.device_id,
                    self.identity.client_envelope_id,
                ),
            ).fetchone()
        assert row is not None
        return dict(row)

    def execute(self, sql: str, parameters: tuple[object, ...]) -> None:
        with self.database.transaction(immediate=True) as connection:
            connection.execute(sql, parameters)

    def row(self, *, role: str) -> dict[str, object]:
        with self.database.transaction() as connection:
            result = connection.execute(
                "SELECT * FROM personal_context_publication_rows "
                "WHERE publication_batch_id = ? AND role = ?",
                (self.receipt.publication_batch_id, role),
            ).fetchone()
        assert result is not None
        return dict(result)

    def rewrite_row(
        self,
        *,
        role: str,
        domain: str | None = None,
        canonical: bytes | None = None,
        fields: dict[str, object] | None = None,
    ) -> bytes:
        fields = fields or {}
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM personal_context_publication_rows "
                "WHERE publication_batch_id = ? AND role = ?",
                (self.receipt.publication_batch_id, role),
            ).fetchone()
            assert row is not None
            old_domain, old_canonical = self.journal.decrypt_row(row)
            final_domain = domain or old_domain
            final_canonical = canonical or old_canonical
            values = dict(row)
            values.update(fields)
            payload = canonical_json_bytes(
                {"canonical": final_canonical.decode("utf-8"), "domain": final_domain}
            )
            integrity_tag = "hmac-sha256-v1:" + hmac.new(
                self.keys.integrity_key,
                payload,
                hashlib.sha256,
            ).hexdigest()
            values["integrity_tag"] = integrity_tag
            envelope = EnvelopeCipher(
                self.keys.encryption_key,
                key_version=self.keys.key_version,
            ).encrypt(
                payload,
                self.journal._aad(
                    profile_id=str(values["profile_id"]),
                    batch_id=str(values["publication_batch_id"]),
                    sequence=int(values["profile_publication_sequence"]),
                    ordinal=int(values["batch_ordinal"]),
                    batch_size=int(values["batch_size"]),
                    role=str(values["role"]),
                    purge_generation=int(values["purge_generation"]),
                    object_id=str(values["opaque_object_id"]),
                    version_id=str(values["opaque_version_id"]),
                    operation=str(values["operation"]),
                    deterministic_envelope_id=str(values["deterministic_envelope_id"]),
                    integrity_tag=integrity_tag,
                    sync_server_cursor=(
                        None
                        if values["sync_server_cursor"] is None
                        else int(values["sync_server_cursor"])
                    ),
                    row_state=str(values["row_state"]),
                ),
            )
            connection.execute(
                """
                UPDATE personal_context_publication_rows
                SET profile_id = ?, profile_publication_sequence = ?,
                    publication_batch_id = ?, batch_ordinal = ?, batch_size = ?,
                    purge_generation = ?, role = ?, opaque_object_id = ?,
                    opaque_version_id = ?, operation = ?, algorithm = ?,
                    key_version = ?, nonce = ?, wrapped_dek = ?,
                    wrapped_dek_nonce = ?, ciphertext = ?, integrity_tag = ?,
                    payload_size_bytes = ?, deterministic_envelope_id = ?,
                    sync_server_cursor = ?, row_state = ?
                WHERE profile_id = ? AND profile_publication_sequence = ?
                  AND batch_ordinal = ?
                """,
                (
                    values["profile_id"],
                    values["profile_publication_sequence"],
                    values["publication_batch_id"],
                    values["batch_ordinal"],
                    values["batch_size"],
                    values["purge_generation"],
                    values["role"],
                    values["opaque_object_id"],
                    values["opaque_version_id"],
                    values["operation"],
                    envelope.algorithm,
                    envelope.key_version,
                    envelope.nonce,
                    envelope.wrapped_dek,
                    envelope.wrapped_dek_nonce,
                    envelope.ciphertext,
                    integrity_tag,
                    len(payload),
                    values["deterministic_envelope_id"],
                    values["sync_server_cursor"],
                    values["row_state"],
                    row["profile_id"],
                    row["profile_publication_sequence"],
                    row["batch_ordinal"],
                ),
            )
        return final_canonical

    def tamper(self, fact: str) -> None:
        receipt_table = "personal_context_ingress_receipts"
        receipt_where = " WHERE receipt_id = ?"
        if fact == "receipt_id":
            self.execute(
                f"UPDATE {receipt_table} SET receipt_id = ?{receipt_where}",
                ("receipt-tampered", self.receipt.receipt_id),
            )
        elif fact == "profile":
            source = self.row(role="semantic")
            _, canonical = self.journal.decrypt_row(source)  # type: ignore[arg-type]
            body = json.loads(canonical)
            body["profile_id"] = "profile-tampered"
            changed = canonical_json_bytes(body)
            digest = _digest(changed)
            self.rewrite_row(role="semantic", canonical=changed)
            self.execute(
                f"UPDATE {receipt_table} SET canonical_payload_digest = ?{receipt_where}",
                (digest, self.receipt.receipt_id),
            )
            self.identity = replace(self.identity, canonical_payload_digest=digest)
        elif fact == "generation":
            self.execute(
                f"UPDATE {receipt_table} SET purge_generation = ?{receipt_where}",
                (1, self.receipt.receipt_id),
            )
        elif fact == "batch_sequence":
            self.execute(
                f"UPDATE {receipt_table} SET profile_publication_sequence = ?{receipt_where}",
                (999, self.receipt.receipt_id),
            )
        elif fact == "batch_size":
            self.execute(
                "UPDATE personal_context_publication_batches SET batch_size = 3 "
                "WHERE publication_batch_id = ?",
                (self.receipt.publication_batch_id,),
            )
        elif fact == "result_object":
            self.rewrite_row(
                role="semantic", fields={"opaque_object_id": "record-tampered"}
            )
            self.execute(
                f"UPDATE {receipt_table} SET resulting_object_id = ?{receipt_where}",
                ("record-tampered", self.receipt.receipt_id),
            )
        elif fact == "result_version":
            self.rewrite_row(
                role=self.source_role,
                fields={"opaque_version_id": "version-tampered"},
            )
            self.execute(
                f"UPDATE {receipt_table} SET resulting_version_id = ?{receipt_where}",
                ("version-tampered", self.receipt.receipt_id),
            )
        elif fact == "receipt_result_version":
            self.execute(
                f"UPDATE {receipt_table} SET resulting_version_id = ?{receipt_where}",
                ("version-tampered", self.receipt.receipt_id),
            )
        elif fact == "source_result_version":
            self.rewrite_row(
                role=self.source_role,
                fields={"opaque_version_id": "version-tampered"},
            )
        elif fact == "manifest_revision":
            self.execute(
                f"UPDATE {receipt_table} SET resulting_manifest_revision = 99{receipt_where}",
                (self.receipt.receipt_id,),
            )
        elif fact == "manifest_revision_type":
            self.execute(
                f"UPDATE {receipt_table} SET resulting_manifest_revision = ?{receipt_where}",
                ("protected-canary", self.receipt.receipt_id),
            )
        elif fact == "manifest_revision_fractional":
            self.execute(
                f"UPDATE {receipt_table} SET resulting_manifest_revision = 1.5{receipt_where}",
                (self.receipt.receipt_id,),
            )
        elif fact == "manifest_version":
            self.rewrite_row(
                role="manifest", fields={"opaque_version_id": "manifest-tampered"}
            )
            self.execute(
                f"UPDATE {receipt_table} SET resulting_manifest_version_id = ?{receipt_where}",
                ("manifest-tampered", self.receipt.receipt_id),
            )
        elif fact == "source_role":
            self.rewrite_row(role="semantic", fields={"role": "purge_barrier"})
        elif fact == "source_domain":
            self.rewrite_row(role="semantic", domain="personal_context.scope")
        elif fact == "source_domain_record":
            self.rewrite_row(role=self.source_role, domain="personal_context.record")
        elif fact == "source_operation":
            self.rewrite_row(role="semantic", fields={"operation": "tombstone"})
        elif fact == "manifest_sibling":
            row = self.row(role="manifest")
            manifest = ProfileManifest.model_validate_json(
                self.journal.decrypt_row(row)[1]  # type: ignore[arg-type]
            )
            changed = manifest.model_copy(update={"revision": manifest.revision + 10})
            self.rewrite_row(role="manifest", canonical=canonical_bytes(changed))
        elif fact == "manifest_lineage":
            self.execute(
                "UPDATE personal_context_object_versions SET parent_version_id = ? "
                "WHERE profile_id = ? AND object_type = 'manifest' "
                "AND object_id = ? AND version_id = ?",
                (
                    "manifest-parent-tampered",
                    self.profile_id,
                    self.profile_id,
                    self.receipt.manifest_version_id,
                ),
            )
        elif fact == "current_manifest":
            profile_id = str(self.row(role="manifest")["profile_id"])
            with self.database.transaction(immediate=True) as connection:
                historical = connection.execute(
                    "SELECT parent_version_id FROM personal_context_object_versions "
                    "WHERE profile_id = ? AND object_type = 'manifest' "
                    "AND object_id = ? AND version_id = ?",
                    (profile_id, profile_id, self.receipt.manifest_version_id),
                ).fetchone()
                assert historical is not None and historical["parent_version_id"] is not None
                connection.execute(
                    "UPDATE personal_context_object_heads SET current_version_id = ? "
                    "WHERE profile_id = ? AND object_type = 'manifest' AND object_id = ?",
                    (historical["parent_version_id"], profile_id, profile_id),
                )
        elif fact == "digest":
            self.execute(
                f"UPDATE {receipt_table} SET canonical_payload_digest = ?{receipt_where}",
                ("sha256:" + "0" * 64, self.receipt.receipt_id),
            )
        elif fact == "ciphertext":
            self.execute(
                "UPDATE personal_context_publication_rows SET ciphertext = ? "
                "WHERE publication_batch_id = ? AND role = 'semantic'",
                (b"corrupt", self.receipt.publication_batch_id),
            )
        elif fact == "pending_state":
            self.execute(
                "UPDATE personal_context_publication_batches SET status = 'complete' "
                "WHERE publication_batch_id = ?",
                (self.receipt.publication_batch_id,),
            )
        elif fact == "terminal_state":
            self.rewrite_row(role="semantic", fields={"row_state": "acknowledged"})
        else:
            raise AssertionError(f"unknown test fact: {fact}")


def _proposal(
    service: PersonalContextService,
    *,
    proposal_id: str,
) -> ProfileProposal:
    profile_id = service.get_manifest().profile_id
    return proposal(profile_id=profile_id, proposal_id=proposal_id).model_copy(
        update={
            "scope_id": service.list_scopes()[0].scope_id,
            "proposed_record": _record(service, f"{proposal_id}-record"),
        }
    )


def _ingress_case(
    service: PersonalContextService,
    case: str,
) -> tuple[str, ProfileManifest | ProfileProposal | ProfileRecord | ProfileScope, str | None]:
    if case == "record":
        return "personal_context.record", _record(service, "record-ingress"), None
    if case == "scope":
        global_scope = service.list_scopes()[0]
        return (
            "personal_context.scope",
            global_scope.model_copy(
                update={
                    "scope_id": "scope-ingress",
                    "kind": ScopeKind.WORKSPACE,
                    "version_id": "scope-ingress-v1",
                }
            ),
            None,
        )
    if case == "manifest":
        current = service.get_manifest()
        return (
            "personal_context.manifest",
            current.model_copy(
                update={
                    "revision": current.revision + 1,
                    "updated_at": current.updated_at + timedelta(seconds=1),
                    "current_version_id": "manifest-ingress-v1",
                }
            ),
            _digest(canonical_bytes(current)),
        )
    if case == "proposal-pending":
        return (
            "personal_context.proposal",
            _proposal(service, proposal_id="proposal-pending-ingress"),
            None,
        )
    if case == "proposal-terminal":
        pending = _proposal(service, proposal_id="proposal-terminal-ingress")
        service.create_proposal(pending)
        terminal = pending.model_copy(
            update={
                "state": ProposalState.REJECTED,
                "proposed_record": None,
                "confidence": None,
            }
        )
        return "personal_context.proposal", terminal, _digest(canonical_bytes(pending))
    raise AssertionError(f"unknown legacy ingress case: {case}")

@pytest.fixture()
def harness(request, tmp_path, monkeypatch) -> LegacyReceiptHarness:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    database = PersonalizationDB.for_path(tmp_path / "personalization.db")
    counter: dict[str, int] = {}

    def identifiers(label: str) -> str:
        counter[label] = counter.get(label, 0) + 1
        return f"{label}-{counter[label]}"

    repository = PersonalContextRepository(database)
    service = PersonalContextService(
        repository,
        clock=lambda: datetime.now(UTC),
        id_factory=identifiers,
    )
    manifest = service.create_profile()
    case = getattr(request, "param", "record")
    domain, value, base_object_hash = _ingress_case(service, case)
    canonical = canonical_bytes(value)
    wire_version = (
        "sync-proposal-sha256:" + hashlib.sha256(canonical).hexdigest()
        if isinstance(value, ProfileProposal)
        else value.current_version_id
        if isinstance(value, ProfileManifest)
        else value.version_id
    )
    identity = IngressIdentity(
        dataset_id="dataset-a",
        device_id="device-a",
        client_envelope_id=f"client-envelope-legacy-{case}",
        canonical_payload_digest=_digest(canonical),
        purge_generation=0,
        wire_entity_version=wire_version,
    )
    receipt = service.apply_sync_ingress(
        identity=identity,
        domain=domain,
        value=value,
        base_object_hash=base_object_hash,
    )
    with database.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE personal_context_ingress_receipts SET wire_entity_version = '' "
            "WHERE receipt_id = ?",
            (receipt.receipt_id,),
        )
    return LegacyReceiptHarness(
        database=database,
        service=service,
        keys=repository.key_material_for_test(manifest.profile_id),
        profile_id=manifest.profile_id,
        identity=identity,
        receipt=receipt,
        case=case,
        domain=domain,
        canonical=canonical,
        source_role="manifest" if domain == "personal_context.manifest" else "semantic",
        expected_object_id=(
            value.profile_id
            if isinstance(value, ProfileManifest)
            else value.proposal_id
            if isinstance(value, ProfileProposal)
            else value.record_id
            if isinstance(value, ProfileRecord)
            else value.scope_id
        ),
    )


def test_matching_legacy_receipt_backfills_only_wire_identity(
    harness: LegacyReceiptHarness,
) -> None:
    before = harness.stored_receipt()
    replay = harness.replay()
    after = harness.stored_receipt()

    assert replay == harness.receipt
    assert harness.stored_wire_version() == harness.identity.wire_entity_version
    before.pop("wire_entity_version")
    after.pop("wire_entity_version")
    assert after == before


def test_modern_receipt_replay_does_not_enter_legacy_validation(
    harness: LegacyReceiptHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness.execute(
        "UPDATE personal_context_ingress_receipts SET wire_entity_version = ? "
        "WHERE receipt_id = ?",
        (harness.identity.wire_entity_version, harness.receipt.receipt_id),
    )

    def reject_legacy_path(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("modern replay entered legacy validation")

    monkeypatch.setattr(
        PersonalContextPublicationJournal,
        "_validate_legacy_receipt_source",
        reject_legacy_path,
    )

    assert harness.replay() == harness.receipt


@pytest.mark.parametrize(
    "harness",
    ["record", "scope", "manifest", "proposal-pending", "proposal-terminal"],
    indirect=True,
)
def test_supported_legacy_ingress_backfills_domain_specific_identity(
    harness: LegacyReceiptHarness,
) -> None:
    replay = harness.replay()
    source = harness.row(role=harness.source_role)
    source_domain, source_canonical = harness.journal.decrypt_row(source)  # type: ignore[arg-type]

    assert replay == harness.receipt
    assert source_domain == harness.domain
    assert source_canonical == harness.canonical
    assert str(source["opaque_object_id"]) == harness.expected_object_id
    assert replay.resulting_object_id == harness.expected_object_id
    assert str(source["opaque_version_id"]) == replay.resulting_version_id
    assert replay.wire_entity_version == harness.identity.wire_entity_version
    if harness.domain == "personal_context.proposal":
        assert replay.resulting_version_id != replay.wire_entity_version
        stored_proposal = ProfileProposal.model_validate_json(source_canonical)
        if harness.case == "proposal-pending":
            assert stored_proposal.state is ProposalState.PENDING
            assert stored_proposal.proposed_record is not None
        else:
            assert stored_proposal.state is ProposalState.REJECTED
            assert stored_proposal.proposed_record is None
            assert stored_proposal.confidence is None
    else:
        assert replay.resulting_version_id == replay.wire_entity_version


@pytest.mark.parametrize(
    ("harness", "fact"),
    [
        pytest.param("record", "source_operation", id="record-operation"),
        pytest.param("scope", "source_domain_record", id="scope-domain"),
        pytest.param("manifest", "manifest_sibling", id="manifest-revision"),
        pytest.param(
            "proposal-pending",
            "receipt_result_version",
            id="proposal-pending-receipt-version",
        ),
        pytest.param(
            "proposal-terminal",
            "source_result_version",
            id="proposal-terminal-source-version",
        ),
    ],
    indirect=["harness"],
)
def test_legacy_domain_binding_mismatch_is_content_free_and_mutation_free(
    harness: LegacyReceiptHarness,
    fact: str,
) -> None:
    harness.tamper(fact)

    with pytest.raises(ValueError, match="identity reused") as error:
        harness.replay()

    assert str(error.value) == REPLAY_ERROR
    assert harness.stored_wire_version() == ""


@pytest.mark.parametrize(
    "fact",
    [
        "receipt_id",
        "profile",
        "generation",
        "batch_sequence",
        "batch_size",
        "result_object",
        "result_version",
        "manifest_revision",
        "manifest_revision_type",
        "manifest_revision_fractional",
        "manifest_version",
        "source_role",
        "source_domain",
        "source_operation",
        "manifest_sibling",
        "manifest_lineage",
        "current_manifest",
        "digest",
        "ciphertext",
        "pending_state",
        "terminal_state",
    ],
)
def test_legacy_backfill_rejects_mismatched_fact_without_mutating_receipt(
    harness: LegacyReceiptHarness,
    fact: str,
) -> None:
    harness.tamper(fact)

    with pytest.raises(ValueError, match="identity reused") as error:
        harness.replay()

    assert str(error.value) == REPLAY_ERROR
    assert harness.stored_wire_version() == ""


def test_legacy_backfill_rejects_changed_encryption_key_without_mutation(
    harness: LegacyReceiptHarness,
) -> None:
    wrong_keys = replace(harness.keys, encryption_key=b"x" * 32)

    with pytest.raises(ValueError, match="identity reused") as error:
        harness.replay(journal=PersonalContextPublicationJournal(wrong_keys))

    assert str(error.value) == REPLAY_ERROR
    assert harness.stored_wire_version() == ""


def test_later_valid_manifest_head_keeps_exact_legacy_receipt_replayable(
    harness: LegacyReceiptHarness,
) -> None:
    harness.service.create_record(_record(harness.service, "record-later"))

    replay = harness.replay()

    assert replay == harness.receipt
    assert harness.stored_wire_version() == harness.identity.wire_entity_version


@pytest.mark.parametrize("harness", ["proposal-pending"], indirect=True)
def test_later_terminal_proposal_keeps_pending_legacy_receipt_replayable(
    harness: LegacyReceiptHarness,
) -> None:
    pending = ProfileProposal.model_validate_json(harness.canonical)
    terminal = pending.model_copy(
        update={
            "state": ProposalState.REJECTED,
            "proposed_record": None,
            "confidence": None,
        }
    )
    canonical = canonical_bytes(terminal)
    terminal_identity = IngressIdentity(
        dataset_id="dataset-a",
        device_id="device-a",
        client_envelope_id="client-envelope-terminal-after-pending",
        canonical_payload_digest=_digest(canonical),
        purge_generation=0,
        wire_entity_version=(
            "sync-proposal-sha256:" + hashlib.sha256(canonical).hexdigest()
        ),
    )
    harness.service.apply_sync_ingress(
        identity=terminal_identity,
        domain="personal_context.proposal",
        value=terminal,
        base_object_hash=_digest(harness.canonical),
    )

    assert harness.replay() == harness.receipt
    assert harness.stored_wire_version() == harness.identity.wire_entity_version


def test_complete_acknowledged_batch_keeps_exact_legacy_receipt_replayable(
    harness: LegacyReceiptHarness,
) -> None:
    with harness.database.transaction(immediate=True) as connection:
        rows = connection.execute(
            "SELECT * FROM personal_context_publication_rows "
            "WHERE publication_batch_id = ? ORDER BY batch_ordinal",
            (harness.receipt.publication_batch_id,),
        ).fetchall()
        for cursor, row in enumerate(rows, start=100):
            harness.journal.transition_row_state(
                connection,
                row,
                row_state="acknowledged",
                sync_server_cursor=cursor,
            )
        connection.execute(
            "UPDATE personal_context_publication_batches SET status = 'complete' "
            "WHERE publication_batch_id = ?",
            (harness.receipt.publication_batch_id,),
        )

    assert harness.replay() == harness.receipt
    assert harness.stored_wire_version() == harness.identity.wire_entity_version


def test_backfilled_receipt_uses_strict_wire_identity_on_subsequent_replay(
    harness: LegacyReceiptHarness,
) -> None:
    assert harness.replay() == harness.receipt
    changed = replace(harness.identity, wire_entity_version="wire-tampered")

    with pytest.raises(ValueError, match="identity reused") as error:
        harness.replay(identity=changed)

    assert str(error.value) == REPLAY_ERROR
    assert harness.stored_wire_version() == harness.identity.wire_entity_version

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Sync.v2.personal_context_ongoing_contract import (
    PERSONAL_CONTEXT_ONGOING_ENDPOINTS,
    PersonalContextActivationReceipt,
    PersonalContextAuthorityMetadata,
    PersonalContextCleanupAck,
    PersonalContextExchangeProof,
    PersonalContextPurgeReceipt,
    PersonalContextRelayContinuation,
    export_personal_context_ongoing_contract,
    validate_client_personal_context_metadata,
)


def test_exchange_proof_requires_exact_version_epoch_and_token() -> None:
    proof = PersonalContextExchangeProof.model_validate(
        {
            "ongoing_sync_version": 1,
            "activation_epoch": "epoch_0123456789abcdef",
            "continuity_token": "continuity_0123456789abcdef",
        }
    )

    assert proof.ongoing_sync_version == 1


def test_client_envelope_cannot_claim_home_authority() -> None:
    with pytest.raises(ValueError, match="home authority"):
        validate_client_personal_context_metadata(
            PersonalContextAuthorityMetadata(
                role="home_authority",
                publication_batch_id="batch_0123456789abcdef",
                profile_publication_sequence=1,
                batch_ordinal=0,
                batch_size=2,
            )
        )


def test_pull_relay_continuation_distinguishes_pending_from_poisoned() -> None:
    continuation = PersonalContextRelayContinuation.model_validate(
        {
            "state": "relay_poisoned",
            "scan_watermark": "cursor_0123456789abcdef",
        }
    )

    assert continuation.state == "relay_poisoned"


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "role": "home_authority",
            "publication_batch_id": "batch_0123456789abcdef",
            "profile_publication_sequence": 1,
            "batch_ordinal": 0,
        },
        {
            "role": "client_ingress",
            "publication_batch_id": "batch_0123456789abcdef",
            "profile_publication_sequence": 1,
            "batch_ordinal": 0,
            "batch_size": 2,
        },
    ],
)
def test_authority_metadata_requires_role_specific_publication_fields(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        PersonalContextAuthorityMetadata.model_validate(metadata)


def test_contract_identifiers_are_strict_bounded_and_content_free() -> None:
    with pytest.raises(ValueError):
        PersonalContextExchangeProof.model_validate(
            {
                "ongoing_sync_version": 1,
                "activation_epoch": "short",
                "continuity_token": "continuity_0123456789abcdef",
            }
        )
    with pytest.raises(ValueError):
        PersonalContextExchangeProof.model_validate(
            {
                "ongoing_sync_version": 1,
                "activation_epoch": "!" + "a" * 16,
                "continuity_token": "continuity_0123456789abcdef",
            }
        )
    with pytest.raises(ValueError):
        PersonalContextExchangeProof.model_validate(
            {
                "ongoing_sync_version": 1,
                "activation_epoch": "a" * 257,
                "continuity_token": "continuity_0123456789abcdef",
            }
        )
    with pytest.raises(ValueError):
        PersonalContextRelayContinuation.model_validate(
            {
                "state": "complete",
                "scan_watermark": "x" * 513,
            }
        )
    with pytest.raises(ValueError):
        PersonalContextCleanupAck.model_validate(
            {
                "object_id": "object_0123456789abcdef",
                "version_id": "version_0123456789abcdef",
                "purge_generation": 0,
                "server_cleanup_complete": "true",
            }
        )


def test_activation_and_purge_receipts_require_strict_safe_values() -> None:
    receipt = PersonalContextActivationReceipt.model_validate(
        {
            "activation_id": "activation_0123456789abcdef",
            "baseline_digest": "a" * 64,
            "purge_generation": 0,
            "publication_watermark": 0,
            "home_server_cursor": 0,
            "home_manifest_revision": 0,
            "home_manifest_version_id": "version_0123456789abcdef",
            "state": "prepared",
        }
    )
    assert receipt.state == "prepared"

    with pytest.raises(ValueError):
        PersonalContextActivationReceipt.model_validate(
            {
                **receipt.model_dump(),
                "baseline_digest": "!" + "a" * 64,
            }
        )

    with pytest.raises(ValueError):
        PersonalContextPurgeReceipt.model_validate(
            {
                "request_id": "request_0123456789abcdef",
                "profile_id": "profile_0123456789abcdef",
                "purge_generation": 0,
                "barrier_envelope_id": "barrier_0123456789abcdef",
                "state": "accepted",
            }
        )


def test_contract_endpoint_map_is_complete_and_versioned() -> None:
    assert PERSONAL_CONTEXT_ONGOING_ENDPOINTS == {
        "capabilities": ("GET", "/api/v1/sync/capabilities"),
        "activation_prepare": ("POST", "/api/v1/sync/personal-context/bootstrap"),
        "activation_acknowledge": (
            "POST",
            "/api/v1/sync/personal-context/activation/acknowledge",
        ),
        "push": ("POST", "/api/v1/sync/push"),
        "pull": ("GET", "/api/v1/sync/pull"),
        "conflict_list": ("GET", "/api/v1/sync/conflicts"),
        "conflict_resolve": ("POST", "/api/v1/sync/conflicts/resolve"),
        "purge": ("POST", "/api/v1/sync/personal-context/purge"),
    }


def test_contract_export_includes_cleanup_acknowledgement_root() -> None:
    contract = export_personal_context_ongoing_contract()

    assert "PersonalContextCleanupAck" in contract["$defs"]


def test_contract_generator_writes_reproducible_schema_and_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "contract"
    source_commit = "0" * 40
    generator = Path(__file__).parents[3] / "Helper_Scripts" / "generate_personal_context_ongoing_contract.py"

    subprocess.run(
        [
            sys.executable,
            str(generator),
            "--source-commit",
            source_commit,
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    schema_path = output_dir / "personal-context-ongoing-v1.schema.json"
    manifest_path = output_dir / "personal-context-ongoing-v1.manifest.json"
    schema_bytes = schema_path.read_bytes()
    manifest = json.loads(manifest_path.read_text())

    assert schema_bytes.endswith(b"\n")
    assert manifest == {
        "contract": "personal-context-ongoing-v1",
        "schema_version": 1,
        "server_source_commit": source_commit,
        "sha256": f"sha256:{hashlib.sha256(schema_bytes).hexdigest()}",
    }

    first_schema = schema_bytes
    subprocess.run(
        [
            sys.executable,
            str(generator),
            "--source-commit",
            source_commit,
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert schema_path.read_bytes() == first_schema

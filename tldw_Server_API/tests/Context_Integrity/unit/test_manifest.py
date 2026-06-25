from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _entry(asset_id: str, digest: str = "sha256:a") -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "source_type": "skill_file",
        "digest": digest,
        "display_name": asset_id,
        "executable": True,
        "required": False,
        "owner_scope": "user:1",
    }


def test_manifest_roundtrip_verifies_signature() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=1, entries=[_entry("skill:user:1/demo")], signer=signer)

    verified = verify_signed_manifest(signed, signer=signer)

    assert verified.sequence == 1
    assert verified.entries[0]["asset_id"] == "skill:user:1/demo"


def test_manifest_tamper_is_rejected() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        ManifestSignatureError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=1, entries=[_entry("skill:user:1/demo")], signer=signer)
    signed["manifest"]["entries"][0]["digest"] = "sha256:evil"

    with pytest.raises(ManifestSignatureError):
        verify_signed_manifest(signed, signer=signer)


def test_anti_rollback_anchor_rejects_older_valid_manifest() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        AntiRollbackAnchor,
        HmacManifestSigner,
        ManifestRollbackError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=2, entries=[_entry("skill:user:1/demo")], signer=signer)
    anchor = AntiRollbackAnchor(sequence=3, manifest_digest="sha256:newer")

    with pytest.raises(ManifestRollbackError):
        verify_signed_manifest(signed, signer=signer, anti_rollback_anchor=anchor)


def test_anti_rollback_anchor_rejects_same_sequence_with_different_digest() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        AntiRollbackAnchor,
        HmacManifestSigner,
        ManifestRollbackError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=3, entries=[_entry("skill:user:1/demo")], signer=signer)
    anchor = AntiRollbackAnchor(sequence=3, manifest_digest="sha256:different")

    with pytest.raises(ManifestRollbackError):
        verify_signed_manifest(signed, signer=signer, anti_rollback_anchor=anchor)

from __future__ import annotations

import base64
import hashlib
import hmac
from types import MappingProxyType
from typing import Any

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


def _resign(signed: dict[str, Any], signer: Any) -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import _manifest_digest, _stable_json

    manifest = signed["manifest"]
    assert isinstance(manifest, dict)
    payload = _stable_json(manifest)
    signed["manifest_digest"] = _manifest_digest(payload)
    signature = signed["signature"]
    assert isinstance(signature, dict)
    signature["value"] = signer.sign(payload)


def test_hmac_manifest_signer_requires_key_id() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import HmacManifestSigner

    with pytest.raises(ValueError):
        HmacManifestSigner(key_id="", secret=b"secret")


def test_hmac_manifest_signer_requires_secret() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import HmacManifestSigner

    with pytest.raises(ValueError):
        HmacManifestSigner(key_id="test-key", secret=b"")


def test_hmac_manifest_signer_returns_base64url_sha256_signature() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import HmacManifestSigner

    payload = b"payload"
    secret = b"secret"
    signer = HmacManifestSigner(key_id="test-key", secret=secret)
    expected = base64.urlsafe_b64encode(hmac.new(secret, payload, hashlib.sha256).digest()).decode("ascii").rstrip("=")

    signature = signer.sign(payload)

    assert signature == expected
    assert "=" not in signature
    assert set(signature) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    assert signer.verify(payload, signature) is True


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
    assert verified.manifest_digest == signed["manifest_digest"]
    assert verified.key_id == signer.key_id
    assert isinstance(verified.entries, tuple)
    assert verified.entries[0]["asset_id"] == "skill:user:1/demo"


def test_manifest_verification_accepts_mapping_payloads() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=1, entries=[_entry("skill:user:1/demo")], signer=signer)
    signed_mapping = MappingProxyType(
        {
            "manifest": MappingProxyType(signed["manifest"]),
            "signature": MappingProxyType(signed["signature"]),
            "manifest_digest": signed["manifest_digest"],
        }
    )

    verified = verify_signed_manifest(signed_mapping, signer=signer)

    assert verified.entries[0]["asset_id"] == "skill:user:1/demo"


def test_verified_manifest_entries_are_immutable_snapshots() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    entry = _entry("skill:user:1/demo")
    entry["metadata"] = {"nested": {"flag": True}, "tags": ["demo"]}
    signed = create_signed_manifest(sequence=1, entries=[entry], signer=signer)

    verified = verify_signed_manifest(signed, signer=signer)

    with pytest.raises(TypeError):
        verified.entries[0]["digest"] = "sha256:mutated"
    signed["manifest"]["entries"][0]["digest"] = "sha256:mutated"
    signed["manifest"]["entries"][0]["metadata"]["nested"]["flag"] = False
    assert verified.entries[0]["digest"] == "sha256:a"
    assert verified.entries[0]["metadata"]["nested"]["flag"] is True


@pytest.mark.parametrize(
    ("normalized_display_name", "raw_display_name"),
    [
        ("line\nname", "line\r\nname"),
        ("caf\u00e9", "cafe\u0301"),
    ],
)
def test_verified_manifest_entries_use_signed_canonical_string_values(
    normalized_display_name: str,
    raw_display_name: str,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    entry = _entry("skill:user:1/demo")
    entry["display_name"] = normalized_display_name
    signed = create_signed_manifest(sequence=1, entries=[entry], signer=signer)
    signed["manifest"]["entries"][0]["display_name"] = raw_display_name

    verified = verify_signed_manifest(signed, signer=signer)

    assert verified.manifest_digest == signed["manifest_digest"]
    assert verified.entries[0]["display_name"] == normalized_display_name


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


def test_manifest_rejects_non_mapping_signed_manifest() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        ManifestSignatureError,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")

    with pytest.raises(ManifestSignatureError):
        verify_signed_manifest(None, signer=signer)


def test_manifest_rejects_unsupported_signature_algorithm() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        ManifestSignatureError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=1, entries=[_entry("skill:user:1/demo")], signer=signer)
    signed["signature"]["alg"] = "none"

    with pytest.raises(ManifestSignatureError):
        verify_signed_manifest(signed, signer=signer)


@pytest.mark.parametrize(
    "case",
    [
        "key_id_mismatch",
        "manifest_digest_mismatch",
        "signature_value_mismatch",
        "malformed_manifest_shape",
        "malformed_signature_shape",
    ],
)
def test_manifest_rejects_explicit_malformed_verification_inputs(case: str) -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        ManifestSignatureError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=1, entries=[_entry("skill:user:1/demo")], signer=signer)

    if case == "key_id_mismatch":
        signed["signature"]["key_id"] = "other-key"
    elif case == "manifest_digest_mismatch":
        signed["manifest_digest"] = "sha256:different"
    elif case == "signature_value_mismatch":
        signed["signature"]["value"] = "invalid"
    elif case == "malformed_manifest_shape":
        signed["manifest"] = []
    elif case == "malformed_signature_shape":
        signed["signature"] = []
    else:
        raise AssertionError(f"unhandled case: {case}")

    with pytest.raises(ManifestSignatureError):
        verify_signed_manifest(signed, signer=signer)


def test_signer_verify_returns_false_for_non_string_signature() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import HmacManifestSigner

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")

    assert signer.verify(b"payload", None) is False


def test_manifest_rejects_non_ascii_signature_value() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        ManifestSignatureError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=1, entries=[_entry("skill:user:1/demo")], signer=signer)
    signed["signature"]["value"] = "not-ascii-\u00e9"

    with pytest.raises(ManifestSignatureError):
        verify_signed_manifest(signed, signer=signer)


def test_manifest_converts_uncanonicalizable_payload_to_signature_error() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        ManifestSignatureError,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = {
        "manifest": {
            "schema_version": 1,
            "sequence": 1,
            "entries": [{1: "non-string key"}],
        },
        "signature": {
            "alg": "hmac-sha256",
            "key_id": signer.key_id,
            "value": "invalid",
        },
        "manifest_digest": "sha256:invalid",
    }

    with pytest.raises(ManifestSignatureError):
        verify_signed_manifest(signed, signer=signer)


@pytest.mark.parametrize(
    "case",
    [
        "missing_schema_version",
        "non_int_schema_version",
        "bool_schema_version",
        "unsupported_schema_version",
        "missing_sequence",
        "non_int_sequence",
        "bool_sequence",
        "missing_entries",
        "none_entries",
    ],
)
def test_manifest_rejects_malformed_payload_that_is_correctly_signed(case: str) -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        ManifestSignatureError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=1, entries=[_entry("skill:user:1/demo")], signer=signer)
    manifest = signed["manifest"]
    assert isinstance(manifest, dict)

    if case == "missing_schema_version":
        del manifest["schema_version"]
    elif case == "non_int_schema_version":
        manifest["schema_version"] = "1"
    elif case == "bool_schema_version":
        manifest["schema_version"] = True
    elif case == "unsupported_schema_version":
        manifest["schema_version"] = 999
    elif case == "missing_sequence":
        del manifest["sequence"]
    elif case == "non_int_sequence":
        manifest["sequence"] = "1"
    elif case == "bool_sequence":
        manifest["sequence"] = True
    elif case == "missing_entries":
        del manifest["entries"]
    elif case == "none_entries":
        manifest["entries"] = None
    else:
        raise AssertionError(f"unhandled case: {case}")
    _resign(signed, signer)

    with pytest.raises(ManifestSignatureError):
        verify_signed_manifest(signed, signer=signer)


@pytest.mark.parametrize(
    "entry",
    [
        1,
        [["asset_id", "skill:user:1/demo"], ["digest", "sha256:a"]],
    ],
)
def test_manifest_rejects_malformed_entry_that_is_correctly_signed(entry: object) -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        ManifestSignatureError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=1, entries=[_entry("skill:user:1/demo")], signer=signer)
    manifest = signed["manifest"]
    assert isinstance(manifest, dict)
    manifest["entries"] = [entry]
    _resign(signed, signer)

    with pytest.raises(ManifestSignatureError):
        verify_signed_manifest(signed, signer=signer)


@pytest.mark.parametrize(
    "case",
    [
        "missing_digest",
        "missing_source_type",
        "missing_display_name",
        "display_name_none",
        "empty_display_name",
        "missing_owner_scope",
        "owner_scope_none",
        "asset_id_none",
        "executable_string",
        "required_string",
        "empty_asset_id",
        "path_int",
        "metadata_scalar",
    ],
)
def test_manifest_rejects_malformed_entry_schema_that_is_correctly_signed(case: str) -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        ManifestSignatureError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=1, entries=[_entry("skill:user:1/demo")], signer=signer)
    manifest = signed["manifest"]
    assert isinstance(manifest, dict)
    entries = manifest["entries"]
    assert isinstance(entries, list)
    entry = entries[0]
    assert isinstance(entry, dict)

    if case == "missing_digest":
        del entry["digest"]
    elif case == "missing_source_type":
        del entry["source_type"]
    elif case == "missing_display_name":
        del entry["display_name"]
    elif case == "display_name_none":
        entry["display_name"] = None
    elif case == "empty_display_name":
        entry["display_name"] = ""
    elif case == "missing_owner_scope":
        del entry["owner_scope"]
    elif case == "owner_scope_none":
        entry["owner_scope"] = None
    elif case == "asset_id_none":
        entry["asset_id"] = None
    elif case == "executable_string":
        entry["executable"] = "true"
    elif case == "required_string":
        entry["required"] = "false"
    elif case == "empty_asset_id":
        entry["asset_id"] = ""
    elif case == "path_int":
        entry["path"] = 1
    elif case == "metadata_scalar":
        entry["metadata"] = "not-a-mapping"
    else:
        raise AssertionError(f"unhandled case: {case}")
    _resign(signed, signer)

    with pytest.raises(ManifestSignatureError):
        verify_signed_manifest(signed, signer=signer)


def test_manifest_entry_validation_runs_before_anti_rollback_check() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        AntiRollbackAnchor,
        HmacManifestSigner,
        ManifestSignatureError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=1, entries=[_entry("skill:user:1/demo")], signer=signer)
    manifest = signed["manifest"]
    assert isinstance(manifest, dict)
    entries = manifest["entries"]
    assert isinstance(entries, list)
    entry = entries[0]
    assert isinstance(entry, dict)
    del entry["digest"]
    _resign(signed, signer)
    anchor = AntiRollbackAnchor(sequence=2, manifest_digest="sha256:newer")

    with pytest.raises(ManifestSignatureError):
        verify_signed_manifest(signed, signer=signer, anti_rollback_anchor=anchor)


def test_manifest_rejects_unsorted_entries_that_are_correctly_signed() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        ManifestSignatureError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(
        sequence=1,
        entries=[
            _entry("skill:user:1/a"),
            _entry("skill:user:1/b"),
        ],
        signer=signer,
    )
    manifest = signed["manifest"]
    assert isinstance(manifest, dict)
    entries = manifest["entries"]
    assert isinstance(entries, list)
    entries.reverse()
    _resign(signed, signer)

    with pytest.raises(ManifestSignatureError):
        verify_signed_manifest(signed, signer=signer)


def test_manifest_rejects_duplicate_asset_ids_that_are_correctly_signed() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        ManifestSignatureError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(
        sequence=1,
        entries=[
            _entry("skill:user:1/demo", digest="sha256:a"),
            _entry("skill:user:1/demo", digest="sha256:b"),
        ],
        signer=signer,
    )

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

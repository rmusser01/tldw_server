"""Fail-closed preparation of the two Expat candidate source copies."""

from __future__ import annotations

import hashlib
import importlib.util
import json

# Test double for the fixed GnuPG process boundary.
import subprocess  # nosec B404
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Helper_Scripts/Supply_Chain/expat_candidate.py"


def candidate_module():
    """Load the standalone candidate tool without changing import search paths."""
    assert SCRIPT.is_file(), "candidate source preparation is not implemented"
    spec = importlib.util.spec_from_file_location("expat_candidate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create the actual metadata fields consumed by CPython's refresh tool."""
    refresh = tmp_path / "Modules/expat/refresh.sh"
    refresh.parent.mkdir(parents=True)
    refresh.write_text(
        '#!/bin/bash\nexpected_libexpat_tag="R_2_8_3"\n'
        'expected_libexpat_version="2.8.3"\n'
        'expected_libexpat_sha256="22920a86c83f32300b11463635b71f11137a917975af297725e55525027d4e50"\n'
        "# Preserve the existing refresh procedure.\n",
        encoding="utf-8",
    )
    sbom = tmp_path / "Misc/sbom.spdx.json"
    sbom.parent.mkdir()
    sbom.write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "name": "expat",
                        "SPDXID": "SPDXRef-PACKAGE-expat",
                        "versionInfo": "2.8.3",
                        "downloadLocation": "https://github.com/libexpat/libexpat/releases/download/R_2_8_3/expat-2.8.3.tar.gz",
                        "checksums": [
                            {
                                "algorithm": "SHA256",
                                "checksumValue": "22920a86c83f32300b11463635b71f11137a917975af297725e55525027d4e50",
                            }
                        ],
                        "externalRefs": [
                            {
                                "referenceType": "cpe23Type",
                                "referenceLocator": "cpe:2.3:a:libexpat_project:libexpat:2.8.3:*:*:*:*:*:*:*",
                            }
                        ],
                    },
                    {"name": "unrelated", "versionInfo": "2.8.3"},
                ],
                "files": [{"fileName": "retained-until-upstream-regeneration"}],
            }
        ),
        encoding="utf-8",
    )
    return refresh, sbom


def test_refresh_and_sbom_use_the_same_complete_release(tmp_path):
    refresh, sbom = metadata_fixture(tmp_path)
    original = json.loads(sbom.read_text())
    candidate_module().update_python_metadata(tmp_path)
    updated = json.loads(sbom.read_text())
    package = updated["packages"][0]
    assert package["versionInfo"] == "2.8.4"
    assert package["checksums"][0]["checksumValue"] == (
        "b8ece2437692dad44d851c4532723390a5a330990007706be9c8d2b90d294f36"
    )
    assert 'expected_libexpat_tag="R_2_8_4"' in refresh.read_text()
    assert 'expected_libexpat_version="2.8.4"' in refresh.read_text()
    assert package["checksums"][0]["checksumValue"] in refresh.read_text()
    assert package["downloadLocation"].endswith("R_2_8_4/expat-2.8.4.tar.gz")
    assert ":2.8.4:" in package["externalRefs"][0]["referenceLocator"]
    assert updated["packages"][1:] == original["packages"][1:]
    assert updated["files"] == original["files"]
    assert refresh.read_text().endswith("# Preserve the existing refresh procedure.\n")


@pytest.mark.parametrize("mutation", ["version", "checksum", "duplicate", "url", "cpe"])
def test_inconsistent_sbom_is_rejected_before_any_write(tmp_path, mutation):
    refresh, sbom = metadata_fixture(tmp_path)
    data = json.loads(sbom.read_text())
    package = data["packages"][0]
    if mutation == "version":
        package["versionInfo"] = "2.8.2"
    elif mutation == "checksum":
        package["checksums"][0]["checksumValue"] = "0" * 64
    elif mutation == "duplicate":
        data["packages"].append(package.copy())
    elif mutation == "url":
        package["downloadLocation"] = "https://invalid.example/expat-2.8.3.tar.gz"
    else:
        package["externalRefs"][0]["referenceLocator"] = "unexpected"
    sbom.write_text(json.dumps(data))
    before = (refresh.read_bytes(), sbom.read_bytes())
    with pytest.raises(ValueError, match="baseline"):
        candidate_module().update_python_metadata(tmp_path)
    assert (refresh.read_bytes(), sbom.read_bytes()) == before


@pytest.mark.parametrize("mutation", ["changed", "duplicate", "missing"])
def test_unexpected_refresh_assignment_is_rejected_before_any_write(tmp_path, mutation):
    refresh, sbom = metadata_fixture(tmp_path)
    text = refresh.read_text()
    assignment = 'expected_libexpat_version="2.8.3"'
    replacement = {
        "changed": 'expected_libexpat_version="2.8.2"',
        "duplicate": assignment + "\n" + assignment,
        "missing": "# version omitted",
    }[mutation]
    refresh.write_text(text.replace(assignment, replacement))
    before = (refresh.read_bytes(), sbom.read_bytes())
    with pytest.raises(ValueError, match="baseline"):
        candidate_module().update_python_metadata(tmp_path)
    assert (refresh.read_bytes(), sbom.read_bytes()) == before


@pytest.mark.parametrize("kind", ["missing", "corrupt", "symlink"])
def test_source_bundle_rejects_untrusted_archive(tmp_path, kind):
    module = candidate_module()
    for filename in module.SOURCE_SHA256:
        path = tmp_path / filename
        if kind == "corrupt":
            path.write_bytes(b"not the pinned source archive")
        elif kind == "symlink":
            target = tmp_path / (filename + ".target")
            target.write_bytes(b"not a regular source input")
            path.symlink_to(target)
    with pytest.raises(ValueError, match="source"):
        module.verify_sources(tmp_path)


def test_source_contract_covers_system_and_bundled_archives():
    assert set(candidate_module().SOURCE_SHA256) == {
        "Python-3.12.14.tar.xz",
        "expat-2.8.4.tar.gz",
        "expat_2.8.4-1.dsc",
        "expat_2.8.4.orig.tar.gz",
        "expat_2.8.4-1.debian.tar.xz",
    }


SIGNERS = {
    "expat-2.8.4.tar.gz": (
        "CB8DE70A90CFBF6C3BF5CC5696262ACFFBD3AEC6",
        "3176EF7DB2367F1FCA4F306B1F9B0E909AF37285",
        "00",
    ),
    "Python-3.12.14.tar.xz": (
        "7169605F62C751356D054A26A821E680E5FA6305",
        "7169605F62C751356D054A26A821E680E5FA6305",
        "00",
    ),
    "expat_2.8.4-1.dsc": (
        "7D887DC8BA7BBBA7B835E3BADCE310E7864CC8BF",
        "A0DF7E0D3851E0EE45C00BC8ACE1F33CB933BBBB",
        "01",
    ),
}


def signature_status(filename: str) -> str:
    """Model the actual GnuPG status protocol, not human diagnostic output."""
    signer, primary, signature_class = SIGNERS[filename]
    return (
        "[GNUPG:] NEWSIG\n"
        f"[GNUPG:] GOODSIG {signer[-16:]} Release signer\n"
        f"[GNUPG:] VALIDSIG {signer} 2026-08-31 1788182984 0 4 0 1 10 {signature_class} {primary}\n"
        "[GNUPG:] TRUST_UNDEFINED 0 pgp\n"
    )


@pytest.mark.parametrize("filename", SIGNERS)
def test_signature_accepts_exact_approved_signer_without_owner_trust(filename):
    result = candidate_module().validate_signature_status(filename, signature_status(filename), 0)
    assert result == {"signer": SIGNERS[filename][0], "primary": SIGNERS[filename][1]}


@pytest.mark.parametrize(
    "adverse_status",
    [
        "BADSIG",
        "ERRSIG",
        "NO_PUBKEY",
        "EXPSIG",
        "EXPKEYSIG",
        "REVKEYSIG",
        "KEYEXPIRED",
        "SIGEXPIRED",
        "KEYREVOKED",
        "NODATA",
        "FAILURE",
        "ERROR",
        "TRUST_NEVER",
    ],
)
def test_signature_rejects_adverse_status_even_with_matching_validsig(adverse_status):
    filename = "expat-2.8.4.tar.gz"
    status = signature_status(filename) + f"[GNUPG:] {adverse_status} details\n"
    with pytest.raises(ValueError, match="signature"):
        candidate_module().validate_signature_status(filename, status, 0)


@pytest.mark.parametrize(
    "mutation", ["signer", "primary", "class", "missing", "duplicate", "extra-newsig", "truncated", "weak-hash"]
)
def test_signature_rejects_wrong_identity_or_incomplete_authentication(mutation):
    filename = "expat-2.8.4.tar.gz"
    status = signature_status(filename)
    if mutation in {"signer", "primary"}:
        status = status.replace(SIGNERS[filename][mutation == "primary"], "A" * 40)
    elif mutation == "class":
        status = status.replace(" 10 00 ", " 10 01 ")
    elif mutation == "missing":
        status = "[GNUPG:] GOODSIG 96262ACFFBD3AEC6 Release signer\n"
    elif mutation == "duplicate":
        status += signature_status(filename)
    elif mutation == "extra-newsig":
        status += "[GNUPG:] NEWSIG\n"
    elif mutation == "truncated":
        status = "[GNUPG:] VALIDSIG " + SIGNERS[filename][0] + "\n"
    else:
        status = status.replace(" 1 10 00 ", " 1 2 00 ")
    with pytest.raises(ValueError, match="signature"):
        candidate_module().validate_signature_status(filename, status, 0)


def test_signature_rejects_nonzero_process_status_despite_matching_validsig():
    filename = "expat-2.8.4.tar.gz"
    with pytest.raises(ValueError, match="signature"):
        candidate_module().validate_signature_status(filename, signature_status(filename), 2)


def authentication_fixture(tmp_path, monkeypatch, *, failure=None):
    """Replace unavailable GPG only; exercise real hash, file and gate behavior."""
    module = candidate_module()
    source = tmp_path / "sources"
    source.mkdir()
    keys = tmp_path / "keys"
    keys.mkdir()
    for filename in module.SOURCE_SHA256:
        content = filename.encode()
        (source / filename).write_bytes(content)
        monkeypatch.setitem(module.SOURCE_SHA256, filename, hashlib.sha256(content).hexdigest())
    for filename in ("expat-2.8.4.tar.gz.asc", "Python-3.12.14.tar.xz.asc"):
        (source / filename).write_text("signature fixture")
    for filename in ("expat-key.asc", "python-key.asc", "debian-maintainer-full-key.asc"):
        (keys / filename).write_text("public key fixture")

    def gpg_process(command, **kwargs):
        # Reject the wrong command boundary rather than simulating a success
        # regardless of which source, signature, or home the controller uses.
        assert command[0] == "/usr/bin/gpg"
        assert "--no-options" in command and "--no-auto-key-retrieve" in command
        assert "--no-auto-key-import" in command and "--no-autostart" in command
        home = Path(command[command.index("--homedir") + 1])
        assert home.is_dir() and home.stat().st_mode & 0o777 == 0o700
        assert kwargs.get("shell", False) is False
        assert kwargs["timeout"] <= 60
        status = ""
        code = 0
        if "--verify" in command:
            filename = Path(command[-1]).name
            if filename.endswith(".dsc"):
                assert command[command.index("--verify") + 1 :] == [str(source / filename)]
            else:
                assert command[-2:] == [str(source / (filename + ".asc")), str(source / filename)]
            status = signature_status(filename)
            if failure == filename:
                code = 2
        elif "--import" in command:
            assert set(command[command.index("--import") + 1 :]) == {
                str(keys / name) for name in ("expat-key.asc", "python-key.asc", "debian-maintainer-full-key.asc")
            }
            if failure == "import":
                code = 2
        else:
            assert "--version" in command
            status = "gpg (GnuPG) 2.4.7\n"
        return subprocess.CompletedProcess(command, code, status, "retained GPG diagnostic\n")

    monkeypatch.setattr(subprocess, "run", gpg_process)
    return module, source, keys, tmp_path / "evidence"


def test_authentication_records_all_three_fresh_signatures_and_source_hashes(tmp_path, monkeypatch):
    module, source, keys, evidence = authentication_fixture(tmp_path, monkeypatch)
    result = module.authenticate_sources(source, keys, evidence)
    assert set(result["signatures"]) == set(SIGNERS)
    assert result["sources"] == module.verify_sources(source)
    assert json.loads((evidence / "authentication.json").read_text()) == result
    assert not list(evidence.glob("gnupg-*"))
    for filename in SIGNERS:
        assert (evidence / (filename + ".status")).read_text() == signature_status(filename)


@pytest.mark.parametrize("failure", ["import", *SIGNERS])
def test_authentication_failure_retains_diagnostics_without_success_record(tmp_path, monkeypatch, failure):
    module, source, keys, evidence = authentication_fixture(tmp_path, monkeypatch, failure=failure)
    with pytest.raises(ValueError):
        module.authenticate_sources(source, keys, evidence)
    assert not (evidence / "authentication.json").exists()
    assert any(path.read_text() == "2\n" for path in evidence.glob("*.exit"))
    assert any("retained GPG diagnostic" in path.read_text() for path in evidence.glob("*.stderr"))


@pytest.mark.parametrize("kind", ["source", "signature", "key", "symlink-signature", "symlink-key"])
def test_authentication_rejects_untrusted_inputs_before_external_process(tmp_path, monkeypatch, kind):
    module, source, keys, evidence = authentication_fixture(tmp_path, monkeypatch)
    path = source / "Python-3.12.14.tar.xz"
    if "signature" in kind:
        path = source / "Python-3.12.14.tar.xz.asc"
    elif "key" in kind:
        path = keys / "python-key.asc"
    path.unlink()
    if kind.startswith("symlink"):
        path.symlink_to(source / "expat-2.8.4.tar.gz")

    def unexpected_process(*args, **kwargs):
        pytest.fail("untrusted input reached external GPG process")

    monkeypatch.setattr(subprocess, "run", unexpected_process)
    with pytest.raises(ValueError):
        module.authenticate_sources(source, keys, evidence)


def test_authentication_refuses_stale_evidence_directory(tmp_path, monkeypatch):
    module, source, keys, evidence = authentication_fixture(tmp_path, monkeypatch)
    evidence.mkdir()
    sentinel = evidence / "authentication.json"
    sentinel.write_text("old success must not be reused")
    with pytest.raises(FileExistsError):
        module.authenticate_sources(source, keys, evidence)
    assert sentinel.read_text() == "old success must not be reused"


@pytest.mark.parametrize("failure", ["timeout", "missing-executable"])
def test_authentication_preserves_process_failure_evidence(tmp_path, monkeypatch, failure):
    module, source, keys, evidence = authentication_fixture(tmp_path, monkeypatch)

    def failed_process(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 60, output=b"partial status\n", stderr=b"partial diagnostic\n")
        raise FileNotFoundError("GnuPG is unavailable")

    monkeypatch.setattr(subprocess, "run", failed_process)
    with pytest.raises(ValueError, match="GnuPG"):
        module.authenticate_sources(source, keys, evidence)
    assert not (evidence / "authentication.json").exists()
    expected_outcome = "timeout\n" if failure == "timeout" else "os-error\n"
    assert (evidence / "gpg-version.exit").read_text() == expected_outcome
    expected = "partial diagnostic" if failure == "timeout" else "GnuPG is unavailable"
    assert expected in (evidence / "gpg-version.stderr").read_text()

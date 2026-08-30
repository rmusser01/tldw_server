from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import tomllib
from packaging.specifiers import SpecifierSet
from packaging.version import Version

EXPECTED_CONTRACT_DIGEST = "a1e0868dcd873a0c94eb0405934983466ceed68fced4b749489226d9932a5e9b"


def _contract_files(root: Path) -> list[Path]:
    return [
        root / "pyproject.toml",
        *sorted((root / "src/tldw_profile_core").glob("*.py")),
        *sorted((root / "schemas").glob("*.json")),
        *sorted((root / "fixtures/v1").glob("*.json")),
    ]


def _contract_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _contract_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_server_pins_exact_chatbook_profile_core_contract() -> None:
    root = Path(__file__).parents[3] / "packages/tldw_profile_core"

    assert root.is_dir(), "the pinned Shared Profile Core snapshot is missing"
    assert _contract_digest(root) == EXPECTED_CONTRACT_DIGEST


def test_server_python_floor_can_import_the_shared_contract() -> None:
    repository_root = Path(__file__).parents[3]
    server_project = tomllib.loads((repository_root / "pyproject.toml").read_text(encoding="utf-8"))
    core_project = tomllib.loads(
        (repository_root / "packages/tldw_profile_core/pyproject.toml").read_text(encoding="utf-8")
    )
    server_requires = SpecifierSet(server_project["project"]["requires-python"])
    core_requires = SpecifierSet(core_project["project"]["requires-python"])

    assert Version("3.10") not in server_requires
    assert Version("3.11") in server_requires
    assert Version("3.11") in core_requires


def test_server_matches_cross_runtime_canonical_fixture() -> None:
    root = Path(__file__).parents[3] / "packages/tldw_profile_core"
    assert root.is_dir(), "the pinned Shared Profile Core snapshot is missing"
    sys.path.insert(0, str(root / "src"))
    from tldw_profile_core import ProfileProposal, canonical_bytes, integrity_tag

    fixture = json.loads((root / "fixtures/v1/19-jcs-conformance.json").read_text(encoding="utf-8"))
    value = ProfileProposal.model_validate(fixture["data"])

    assert canonical_bytes(value) == fixture["canonical_utf8"].encode("utf-8")
    assert integrity_tag(value, bytes.fromhex(fixture["canonical_key_hex"])) == fixture["integrity_tag"]

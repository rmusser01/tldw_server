from __future__ import annotations

import importlib.util
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "Helper_Scripts/ci/check_frontend_license_gate.py"

SPEC = importlib.util.spec_from_file_location("check_frontend_license_gate", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
license_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(license_gate)


EXPECTED_PREFIXES = (
    "admin-ui/",
    "apps/tldw-frontend/",
    "apps/extension/",
    "apps/packages/ui/",
    "LICENSES/",
    "tldw_Server_API/app/api/v1/",
)

EXPECTED_EXACT = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.txt",
    "Helper_Scripts/ci/check_frontend_license_gate.py",
    ".github/workflows/frontend-license-gate.yml",
    ".github/workflows/frontend-required.yml",
    "tldw_Server_API/app/main.py",
)


def run_cli(*args: str, input_data: bytes = b"") -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        input=input_data,
        capture_output=True,
        check=False,
    )


def test_policy_constants_match_the_trusted_boundaries() -> None:
    assert license_gate.PROTECTED_PREFIXES == EXPECTED_PREFIXES
    assert frozenset(EXPECTED_EXACT) == license_gate.PROTECTED_EXACT


def test_owner_is_allowed_to_change_protected_paths() -> None:
    paths = [*EXPECTED_EXACT, *(f"{prefix}child" for prefix in EXPECTED_PREFIXES)]

    assert license_gate.evaluate(author="RMUSSER01", owner="rmusser01", paths=paths) == []


def test_external_contributor_is_blocked_from_every_protected_boundary() -> None:
    paths = [*EXPECTED_EXACT, *(f"{prefix}child" for prefix in EXPECTED_PREFIXES)]

    assert license_gate.evaluate(author="contributor", owner="rmusser01", paths=paths) == paths


def test_near_prefixes_are_allowed() -> None:
    paths = [
        "admin-ui-copy/x",
        "LICENSE.md",
        "LICENSES-copy/x",
        "tldw_Server_API/app/main.py.bak",
        "tldw_Server_API/app/api/v10/x.py",
    ]

    assert license_gate.evaluate(author="contributor", owner="rmusser01", paths=paths) == []


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["admin-ui/old.ts", "unrelated/new.ts"], ["admin-ui/old.ts"]),
        (["unrelated/old.ts", "apps/tldw-frontend/new.ts"], ["apps/tldw-frontend/new.ts"]),
    ],
)
def test_rename_old_and_new_paths_are_both_examined(paths: list[str], expected: list[str]) -> None:
    assert license_gate.evaluate(author="contributor", owner="rmusser01", paths=paths) == expected


def test_read_nul_paths_preserves_path_bytes() -> None:
    data = b" leading\0trailing \0\ttabs\t\0line\nbreak\0caf\xc3\xa9\0" b"undecodable-\xff\0"

    assert license_gate.read_nul_paths(BytesIO(data)) == [
        " leading",
        "trailing ",
        "\ttabs\t",
        "line\nbreak",
        "caf\N{LATIN SMALL LETTER E WITH ACUTE}",
        "undecodable-\udcff",
    ]


def test_read_nul_paths_continues_after_short_reads() -> None:
    class ShortReadStream(BytesIO):
        def read(self, size: int = -1) -> bytes:
            return super().read(min(size, 2))

    assert license_gate.read_nul_paths(ShortReadStream(b"first\0second\0third"), max_bytes=18) == [
        "first",
        "second",
        "third",
    ]


def test_read_nul_paths_ignores_empty_fields_without_stripping() -> None:
    assert license_gate.read_nul_paths(BytesIO(b"\0 x \0\0\t\0")) == [" x ", "\t"]


def test_read_nul_paths_rejects_oversized_input() -> None:
    with pytest.raises(ValueError, match="changed-path input exceeds 3 bytes"):
        license_gate.read_nul_paths(BytesIO(b"abcd"), max_bytes=3)


def test_cli_blocks_external_protected_path() -> None:
    result = run_cli(
        "--author",
        "contributor",
        "--owner",
        "rmusser01",
        "--null",
        input_data=b"apps/tldw-frontend/page.tsx\0",
    )

    assert result.returncode == 1


@pytest.mark.parametrize(
    ("author", "input_data"),
    [
        ("contributor", b"unrelated/file.py\0"),
        ("RMUSSER01", b"apps/tldw-frontend/page.tsx\0"),
    ],
)
def test_cli_allows_unrelated_or_owner_paths(author: str, input_data: bytes) -> None:
    result = run_cli(
        "--author",
        author,
        "--owner",
        "rmusser01",
        "--null",
        input_data=input_data,
    )

    assert result.returncode == 0


def test_cli_rejects_oversized_input() -> None:
    result = run_cli(
        "--author",
        "contributor",
        "--owner",
        "rmusser01",
        "--null",
        input_data=b"x" * (license_gate.MAX_INPUT_BYTES + 1),
    )

    assert result.returncode == 2


def test_cli_rejects_missing_null_mode() -> None:
    result = run_cli("--author", "contributor", "--owner", "rmusser01")

    assert result.returncode == 2


def test_cli_diagnostics_escape_control_and_surrogate_bytes() -> None:
    result = run_cli(
        "--author",
        "contributor",
        "--owner",
        "rmusser01",
        "--null",
        input_data=b"admin-ui/name\npart\tundecodable-\xff\0",
    )

    assert result.returncode == 1
    assert result.stderr.splitlines()[-1] == (b"- 'admin-ui/name\\npart\\tundecodable-\\udcff'")

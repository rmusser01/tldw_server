"""Ratchet: every env name the app reads is in the curated ``Docs/Operations/Env_Vars.md``, or frozen here.

TASK-13950.21. The baseline lists the names the doc does not mention yet; it only shrinks. The doc counts
as documenting a name when the name appears whole, or matches a placeholder form such as
``CUSTOM_OPENAI_API_{N}_IP`` or ``RG_TEST_*``. The root ``Env_Vars.md`` is generated from code and never counts.

    python -m tldw_Server_API.tests.lint.test_env_vars_docs_ratchet --write-baseline
    python -m tldw_Server_API.tests.lint.test_env_vars_docs_ratchet --tighten
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import pytest

from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, REPO_ROOT
from tldw_Server_API.tests.lint.test_env_read_ratchet import _seed, _trees, app_reads, measure_reads, settings_env_names

DOC: Path = REPO_ROOT / "Docs" / "Operations" / "Env_Vars.md"
BASELINE: Path = Path(__file__).resolve().with_name("env_vars_undocumented_baseline.txt")
ALLOWLIST: dict[str, str] = {
    "APPDATA": "Set by Windows for every process; the app only locates the user config dir (TASK-13950.21)",
    "LOCALAPPDATA": "Set by Windows for every process; the app only locates the local data dir (TASK-13950.21)",
    "XDG_CONFIG_HOME": "Set by the desktop session per the XDG spec, not by operators of this app (TASK-13950.21)",
    "PATH": "Set by the OS shell; the app only searches it for binaries (TASK-13950.21)",
    "CUDA_VISIBLE_DEVICES": "Set by the CUDA runtime or the launcher; documented by NVIDIA (TASK-13950.21)",
    "TORCH_HOME": "Set by PyTorch's convention; documented by PyTorch (TASK-13950.21)",
    "FORCE_COLOR": "Set by the terminal or CI convention (force-color.org), not an app setting (TASK-13950.21)",
    "GITHUB_ACTIONS": "Set by GitHub Actions runners; the app only detects CI (TASK-13950.21)",
    "PYTEST_XDIST_WORKER": "Set by pytest-xdist in each worker; the app only detects test workers (TASK-13950.21)",
}

_PLACEHOLDER = re.compile(r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9_]*(?:(?:\{[^}]+\}|<[^>]+>|\*)[A-Z0-9_]*)+")
_SPLIT = re.compile(r"\{[^}]+\}|<[^>]+>|\*")


class DocIndex(NamedTuple):
    tokens: frozenset[str]
    patterns: tuple[re.Pattern[str], ...]

    def covers(self, name: str) -> bool:
        return name in self.tokens or any(p.fullmatch(name) for p in self.patterns)


def doc_index(text: str) -> DocIndex:
    patterns = tuple(
        re.compile("[A-Z0-9_]+".join(re.escape(part) for part in _SPLIT.split(m.group())))
        for m in _PLACEHOLDER.finditer(text)
    )
    return DocIndex(frozenset(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)), patterns)


def env_names(root: Path = APP_ROOT) -> frozenset[str]:
    reads = app_reads() if root == APP_ROOT else measure_reads(root)
    names = {n for module in reads.values() for read in module for n in read.names}
    for tree in _trees(root).values():
        names |= settings_env_names(tree)
    return frozenset(names)


def undocumented(names: Iterable[str], index: DocIndex, allowlist: Mapping[str, str]) -> list[str]:
    return sorted(n for n in set(names) if n not in allowlist and not index.covers(n))


def problems(current: Sequence[str], baseline: Sequence[str]) -> list[str]:
    now, frozen = set(current), set(baseline)
    found = {
        n: f"{n}: read in app/ but missing from Docs/Operations/Env_Vars.md. Document it there and run "
        f"Helper_Scripts/refresh_docs_published.sh, or add it to ALLOWLIST in {Path(__file__).name} with a reason (TASK-13950.21)"
        for n in now - frozen
    }
    found.update(
        {
            n: f"{n}: stale baseline entry (now documented, allowlisted or no longer read). "
            "Run: python -m tldw_Server_API.tests.lint.test_env_vars_docs_ratchet --tighten"
            for n in frozen - now
        }
    )
    return [found[n] for n in sorted(found)]


def allowlist_problems(allowlist: Mapping[str, str]) -> list[str]:
    return sorted(
        f"{name}: reason must cite TASK-<n> or #<n> and fill every <...>"
        for name, reason in allowlist.items()
        if not re.search(r"TASK-\d+|#\d+", reason) or re.search(r"<[^>]*>", reason)
    )


# --- Tests ---------------------------------------------------------------------------------------


def test_doc_index_matches_whole_names_and_placeholder_patterns() -> None:
    index = doc_index(
        "- `AUTH_MODE`: mode.\n- `CUSTOM_OPENAI_API_{N}_IP` and `RG_TEST_*`; also AUTH_MODE_X.\n"
        "- fields use `custom_openai_api_N_*`.\n"
    )
    assert [n for n in ("AUTH_MODE", "AUTH_MODE_X", "CUSTOM_OPENAI_API_7_IP", "RG_TEST_FOO") if not index.covers(n)] == []
    assert [n for n in ("AUTH", "CUSTOM_OPENAI_API_IP", "RG_TEST", "N_FOO") if index.covers(n)] == []


def test_a_new_undocumented_name_and_a_stale_entry_fail(tmp_path: Path) -> None:
    root = _seed(tmp_path / "app", {"core/X/a.py": 'import os\nA = os.getenv("NEW_ONE")\nB = os.getenv("OLD_ONE")\n'})
    found = problems(undocumented(env_names(root), doc_index("`OLD_ONE`"), {}), ["GONE", "OLD_ONE"])
    assert [p.split(":")[0] for p in found] == ["GONE", "NEW_ONE", "OLD_ONE"]
    assert "stale" in found[0] and "missing from Docs/Operations/Env_Vars.md" in found[1] and "stale" in found[2]


def test_settings_classes_contribute_their_env_names(tmp_path: Path) -> None:
    root = _seed(
        tmp_path / "app",
        {
            "core/X/s.py": (
                "from pydantic import Field\nfrom pydantic_settings import BaseSettings\n\n\nclass S(BaseSettings):\n"
                '    FOO: int = 1\n    bar: str = Field(default="", validation_alias="MCP_BAR")\n    baz: int = 0\n'
            )
        },
    )
    assert env_names(root) == {"FOO", "MCP_BAR"}


def test_allowlist_reasons_cite_a_task() -> None:
    assert allowlist_problems(ALLOWLIST) == []
    assert allowlist_problems({"X": "because"}) != []


@pytest.mark.unit
def test_every_env_name_is_documented_allowlisted_or_baselined() -> None:
    current = undocumented(env_names(), doc_index(DOC.read_text(encoding="utf-8")), ALLOWLIST)
    found = problems(current, BASELINE.read_text(encoding="utf-8").split())
    assert found == [], "\n".join(found)


def _current() -> list[str]:
    return undocumented(env_names(), doc_index(DOC.read_text(encoding="utf-8")), ALLOWLIST)


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-baseline"]:
        names = _current()
    elif sys.argv[1:] == ["--tighten"]:
        keep = set(_current())
        names = [n for n in BASELINE.read_text(encoding="utf-8").split() if n in keep]
    else:
        raise SystemExit("usage: --write-baseline | --tighten")
    BASELINE.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")
    print(f"{len(names)} undocumented names")

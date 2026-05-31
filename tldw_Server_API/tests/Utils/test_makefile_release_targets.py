from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MAKEFILE_PATH = REPO_ROOT / "Makefile"


def _read_makefile() -> str:
    return MAKEFILE_PATH.read_text(encoding="utf-8")


def _extract_target_block(makefile_text: str, target_name: str) -> str:
    lines = makefile_text.splitlines()
    for index, line in enumerate(lines):
        if line == f"{target_name}:":
            block_lines: list[str] = []
            for candidate in lines[index + 1 :]:
                if candidate.startswith("\t"):
                    block_lines.append(candidate)
                    continue
                if candidate == "":
                    if block_lines:
                        break
                    continue
                break
            if not block_lines:
                raise AssertionError(f"Missing recipe for Make target: {target_name}")
            return "\n".join(block_lines) + "\n"
    raise AssertionError(f"Missing Make target: {target_name}")


def test_release_patch_target_exists_and_delegates_to_release_helper() -> None:
    block = _extract_target_block(_read_makefile(), "release-patch")
    assert "Helper_Scripts/release.py" in block
    assert "--bump patch" in block
    assert "RELEASE_DRY_RUN" in block
    assert "--dry-run" in block


def test_release_minor_target_exists_and_delegates_to_release_helper() -> None:
    block = _extract_target_block(_read_makefile(), "release-minor")
    assert "Helper_Scripts/release.py" in block
    assert "--bump minor" in block
    assert "RELEASE_DRY_RUN" in block
    assert "--dry-run" in block


def test_release_target_exists_and_delegates_to_release_helper() -> None:
    makefile_text = _read_makefile()
    block = _extract_target_block(makefile_text, "release")
    assert "Helper_Scripts/release.py" in block or "release-patch" in block
    assert "RELEASE_DRY_RUN=$(RELEASE_DRY_RUN)" in block

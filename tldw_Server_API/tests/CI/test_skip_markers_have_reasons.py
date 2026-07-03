"""Every unconditional skip must say why (audit F9)."""
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
_SELF = Path(__file__).resolve()


@pytest.mark.unit
def test_all_unconditional_skips_have_reasons():
    offenders: list[str] = []
    paths = list(TESTS_ROOT.rglob("test_*.py")) + list(TESTS_ROOT.rglob("conftest.py"))
    for path in paths:
        if path == _SELF:
            continue  # don't scan our own assert-message strings
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            # Detect only on the code before any '#' so commented-out markers
            # don't false-positive.
            clean = line.split("#", 1)[0]
            if "pytest.mark.skip" not in clean or "skipif" in clean:
                continue
            # OK forms: skip(reason=...) on this line, or continuation with
            # reason nearby. Use the raw window (not comment-stripped) so a
            # reason= string containing '#' isn't mangled.
            window = "\n".join(lines[i - 1 : i + 2])
            if "reason" not in window:
                offenders.append(f"{path.relative_to(TESTS_ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        "Unconditional pytest.mark.skip without reason= (add one, "
        "e.g. reason='needs X, see #issue'):\n" + "\n".join(offenders)
    )

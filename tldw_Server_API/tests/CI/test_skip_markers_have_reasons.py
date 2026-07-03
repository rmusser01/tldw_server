"""Every unconditional skip must say why (audit F9)."""
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_all_unconditional_skips_have_reasons():
    offenders: list[str] = []
    for path in TESTS_ROOT.rglob("test_*.py"):
        if path == Path(__file__).resolve():
            continue  # don't scan our own assert-message strings
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            if "pytest.mark.skip" not in line or "skipif" in line:
                continue
            # OK forms: skip(reason=...) on this line, or continuation with reason nearby
            window = "\n".join(text.splitlines()[i - 1 : i + 2])
            if "reason" not in window:
                offenders.append(f"{path.relative_to(TESTS_ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        "Unconditional pytest.mark.skip without reason= (add one, "
        "e.g. reason='needs X, see #issue'):\n" + "\n".join(offenders)
    )

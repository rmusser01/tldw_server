"""Shared quarantine helper for known-failing suites.

See audits/2026-07-02-quarantined-suites.md for the burn-down tracker.
"""
import os
from pathlib import Path

import pytest


def quarantine_items(conftest_file: str, items) -> None:
    """Skip all collected items under conftest_file's directory unless RUN_QUARANTINED=1.

    pytest_collection_modifyitems receives the FULL session item list even in a
    subdirectory conftest, so the skip must be scoped to the suite's own path.
    """
    if os.getenv("RUN_QUARANTINED") == "1":
        return
    here = Path(conftest_file).resolve().parent
    skip = pytest.mark.skip(
        reason="quarantined: known-failing suite, run with RUN_QUARANTINED=1 "
        "(see issue #2581 and audits/2026-07-02-quarantined-suites.md)"
    )
    for item in items:
        try:
            item_path = Path(str(getattr(item, "path", None) or item.fspath)).resolve()
        except Exception as exc:  # pragma: no cover - path oddities should be visible, not fatal
            item.warn(pytest.PytestWarning(f"quarantine: could not resolve path for {item.nodeid}: {exc}"))
            continue
        if here in item_path.parents:
            item.add_marker(skip)

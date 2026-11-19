"""Ratchet: the list of tests allowed to reach the network only shrinks (TASK-15486).

``tldw_Server_API/tests/live_network_baseline.txt`` names, one pytest node id per line, the tests the
session-wide offline blocker (``tests/conftest.py::_no_live_network``) still lets reach a non-loopback
host. A PR that stubs one deletes its line and lowers CEILING to the new count; a PR may not add a line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.tests.helpers.offline_network import LIVE_NETWORK_BASELINE, baseline_node_ids

REPO_ROOT = Path(__file__).resolve().parents[3]
# The count recorded from the first CI run with the blocker on; lower it as the baseline shrinks.
CEILING = 387


def _entries() -> list[str]:
    lines = (line.strip() for line in LIVE_NETWORK_BASELINE.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


@pytest.mark.unit
def test_baseline_does_not_grow():
    entries = _entries()
    assert len(entries) <= CEILING, f"{len(entries)} node ids in {LIVE_NETWORK_BASELINE.name}, ceiling {CEILING}: stub the test's seam instead"


@pytest.mark.unit
def test_baseline_entries_are_sorted_unique_node_ids_of_existing_files():
    entries = _entries()
    assert entries == sorted(set(entries)), "sort the baseline and drop duplicates"
    for entry in entries:
        path, sep, _rest = entry.partition("::")
        assert sep and path.startswith("tldw_Server_API/"), f"not a repo-relative node id: {entry}"
        assert (REPO_ROOT / path).is_file(), f"baseline names a test file that no longer exists: {path}"
    assert set(entries) == baseline_node_ids()

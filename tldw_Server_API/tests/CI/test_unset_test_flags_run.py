"""Sentinel for security-required's TEST_MODE-unset step (TASK-13950.19).

Runs last in that step, so it also catches a fixture that wrote a flag into os.environ.
The step sets TLDW_UNSET_TEST_FLAGS=1 and the workflow contract pins it, so the skip below
cannot hide this test in the required run.
"""

from __future__ import annotations

import os

import pytest

from tldw_Server_API.app.core import testing

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        os.getenv("TLDW_UNSET_TEST_FLAGS") != "1",
        reason="env gate: runs only in security-required's TEST_MODE-unset step (TASK-13950.19)",
    ),
]


def test_unset_run_has_no_test_mode_flags() -> None:
    assert [flag for flag in ("TEST_MODE", "TESTING", "TLDW_TEST_MODE") if flag in os.environ] == []
    assert testing.is_test_mode() is False

"""Safe error summaries retain useful type diagnostics without exception data."""

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing import logging_safety


@pytest.mark.parametrize("error", [ValueError("sensitive body"), RuntimeError("credential=secret")])
def test_exception_summary_omits_message(error):
    assert logging_safety.exception_type_for_log(error) in {"ValueError", "RuntimeError"}


def test_exception_summary_does_not_render_exception():
    class UnrenderableError(Exception):
        def __str__(self):
            raise AssertionError("Logging must not render exception data")

    assert logging_safety.exception_type_for_log(UnrenderableError()) == "UnrenderableError"


def test_exception_summary_is_bounded():
    error_type = type("X" * 1000, (Exception,), {})
    assert len(logging_safety.exception_type_for_log(error_type())) <= 80

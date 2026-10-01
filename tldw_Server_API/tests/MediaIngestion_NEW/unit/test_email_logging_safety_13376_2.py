"""Safe error summaries retain useful type diagnostics without exception data."""

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing import logging_safety

pytestmark = pytest.mark.unit


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


def test_exception_frames_preserve_locations_without_private_exception_data():
    def fail():
        raise ValueError("private-body-and-credential")

    fail.__code__ = fail.__code__.replace(co_filename="private-upload-filename.eml")
    try:
        fail()
    except ValueError as error:
        frames = logging_safety.exception_frames_for_log(error)
    assert any(frame.startswith("fail:") for frame in frames)
    assert "private-upload-filename.eml" not in " ".join(frames)
    assert "private-body-and-credential" not in " ".join(frames)


def test_exception_frames_are_bounded_for_recursive_errors():
    def fail(depth):
        if depth:
            fail(depth - 1)
        else:
            raise ValueError("sensitive body")

    try:
        fail(40)
    except ValueError as error:
        frames = logging_safety.exception_frames_for_log(error)
    assert len(frames) == 16
    assert all(frame.startswith("fail:") for frame in frames)

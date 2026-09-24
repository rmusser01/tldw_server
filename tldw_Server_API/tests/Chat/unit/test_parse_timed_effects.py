"""parse_timed_effects keeps a None fallback for bad stored values, but logs them."""

from __future__ import annotations

import pytest
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.chat_dictionary_schemas import TimedEffects, parse_timed_effects

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "value",
    [{"sticky": 5, "cooldown": 1}, '{"sticky": 5, "cooldown": 1}', TimedEffects(sticky=5, cooldown=1)],
)
def test_valid_representations_parse(value) -> None:
    assert parse_timed_effects(value) == TimedEffects(sticky=5, cooldown=1)


@pytest.mark.parametrize("value", [{"sticky": -1}, '{"sticky": -1}', "{not json"])
def test_invalid_values_fall_back_to_none_and_are_logged(value) -> None:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        assert parse_timed_effects(value) is None
    finally:
        logger.remove(sink)
    assert any("timed_effects" in m for m in messages), "failure was swallowed silently"


@pytest.mark.parametrize("value", [None, "", "   ", "[1, 2]", 7])
def test_absent_or_non_object_values_are_none(value) -> None:
    assert parse_timed_effects(value) is None

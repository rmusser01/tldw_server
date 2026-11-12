import pytest

from tldw_Server_API.app.core import testing
from tldw_Server_API.app.core.MCP_unified import environment as mcp_environment
from tldw_Server_API.app.core.Utils.coercion import FALSY, TRUTHY, env_bool, parse_bool


def _spellings(tokens):
    """Each token as written, upper-cased, and padded the way compose/.env files leave it."""
    for token in sorted(tokens):
        yield token
        yield token.upper()
        yield f"  {token.title()} \t"


@pytest.mark.unit
@pytest.mark.parametrize("value", list(_spellings(TRUTHY)))
@pytest.mark.parametrize("default", [True, False])
def test_every_truthy_spelling_is_true_regardless_of_default(value, default):
    assert parse_bool(value, default=default) is True


@pytest.mark.unit
@pytest.mark.parametrize("value", list(_spellings(FALSY)))
@pytest.mark.parametrize("default", [True, False])
def test_every_falsy_spelling_is_false_regardless_of_default(value, default):
    assert parse_bool(value, default=default) is False


@pytest.mark.unit
@pytest.mark.parametrize("value", ["maybe", "nope", "2", "tru", None, object()])
@pytest.mark.parametrize("default", [True, False])
def test_unrecognised_input_returns_default_never_true(value, default):
    assert parse_bool(value, default=default) is default


@pytest.mark.unit
@pytest.mark.parametrize(("value", "expected"), [(True, True), (False, False), (0, False), (1, True), (2, True), (0.0, False), (0.5, True)])
def test_bool_and_numeric_inputs(value, expected):
    assert parse_bool(value, default=not expected) is expected


@pytest.mark.unit
def test_vocabularies_are_disjoint():
    assert not TRUTHY & FALSY


@pytest.mark.unit
def test_env_bool_unset_uses_default_and_set_values_parse(monkeypatch):
    monkeypatch.delenv("TLDW_COERCION_TEST_FLAG", raising=False)
    assert env_bool("TLDW_COERCION_TEST_FLAG", default=True) is True
    monkeypatch.setenv("TLDW_COERCION_TEST_FLAG", " n ")
    assert env_bool("TLDW_COERCION_TEST_FLAG", default=True) is False
    monkeypatch.setenv("TLDW_COERCION_TEST_FLAG", "enabled ")
    assert env_bool("TLDW_COERCION_TEST_FLAG", default=False) is True
    monkeypatch.setenv("TLDW_COERCION_TEST_FLAG", "garbage")
    assert env_bool("TLDW_COERCION_TEST_FLAG", default=False) is False


@pytest.mark.unit
def test_testing_reexports_and_is_truthy_is_two_way():
    assert testing.parse_bool is parse_bool
    assert testing.env_bool is env_bool
    assert testing.is_truthy(" Enabled ") is True
    assert testing.is_truthy("maybe") is False
    assert testing.is_truthy(None) is False
    # String semantics are preserved for non-string inputs: 2 was never truthy here.
    assert testing.is_truthy(2) is False
    assert testing.is_truthy(True) is True


@pytest.mark.unit
def test_mcp_standalone_copy_keeps_the_same_truthy_set():
    assert mcp_environment._TRUTHY == TRUTHY

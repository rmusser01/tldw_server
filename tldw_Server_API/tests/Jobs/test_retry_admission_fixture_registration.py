"""Bounded selection controls for the retry-admission fixture bridge."""

from __future__ import annotations

from inspect import unwrap
from typing import Any

import pytest
from _pytest.fixtures import FixtureDef, getfixturemarker

from tldw_Server_API.tests._plugins import authnz_isolated_fixtures
from tldw_Server_API.tests.AuthNZ import conftest as authnz_fixtures
from tldw_Server_API.tests.Jobs import test_shared_retry_admission_pg_fixtures as shared_guards

pytest_plugins = shared_guards.pytest_plugins
pytestmark = pytest.mark.unit


def _selected_fixture(request: pytest.FixtureRequest, name: str) -> FixtureDef[Any]:
    """Inspect pytest's actual applicable definition without resolving its body."""
    definitions = request._fixturemanager.getfixturedefs(name, request.node)
    assert definitions, f"No applicable fixture: {name}"
    return definitions[-1]


def test_jobs_selection_excludes_authnz_autouse(request: pytest.FixtureRequest) -> None:
    """Importing the bridge must not add AuthNZ resets to a Jobs test's closure."""
    assert {"reset_singletons", "clear_app_overrides"}.isdisjoint(request.fixturenames)


def test_jobs_event_loop_is_not_authnz(request: pytest.FixtureRequest) -> None:
    """Jobs must retain its normal loop, not AuthNZ's session-loop override."""
    selected = _selected_fixture(request, "event_loop")
    assert unwrap(selected.func).__module__ != authnz_fixtures.__name__


def test_shared_fixture_selection_preserves_original(request: pytest.FixtureRequest) -> None:
    """The bridge must select the existing lifecycle with only native dependencies."""
    selected = _selected_fixture(request, "isolated_test_environment")
    assert unwrap(selected.func) is unwrap(authnz_fixtures.isolated_test_environment)
    assert selected.argnames == ("monkeypatch", "tmp_path")
    assert not selected._autouse
    assert "isolated_test_environment" not in request.fixturenames
    assert "pg_temp_db" not in request.fixturenames


def test_narrow_plugin_exports_only_original_lifecycle() -> None:
    """No autouse reset or custom loop may escape through the narrow namespace."""
    assert authnz_isolated_fixtures.isolated_test_environment is authnz_fixtures.isolated_test_environment
    assert {name for name, value in vars(authnz_isolated_fixtures).items() if getfixturemarker(value) is not None} == {
        "isolated_test_environment",
    }

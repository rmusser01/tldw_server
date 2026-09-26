"""Bounded selection controls for the retry-admission fixture bridge."""

from __future__ import annotations

import pytest

from tldw_Server_API.tests._plugins import authnz_isolated_fixtures
from tldw_Server_API.tests.AuthNZ import conftest as authnz_fixtures
from tldw_Server_API.tests.Jobs import test_shared_retry_admission_pg_fixtures as shared_guards

pytest_plugins = shared_guards.pytest_plugins
pytestmark = pytest.mark.unit


def test_jobs_selection_excludes_authnz_autouse(request: pytest.FixtureRequest) -> None:
    """Importing the bridge must not add AuthNZ resets to a Jobs test's closure."""
    assert {"reset_singletons", "clear_app_overrides"}.isdisjoint(request.fixturenames)


def test_bridge_resolution_preserves_jobs_loop_and_isolation(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe reset selection and function-loop teardown, including a bad bridge."""
    pytester.makepyfile(baseline_plugin="from tldw_Server_API.tests.conftest import event_loop")
    pytester.makepyfile(
        test_probe="""
        import pytest
        loops = []

        @pytest.mark.unit
        def test_no_resets(request: pytest.FixtureRequest) -> None:
            assert {"reset_singletons", "clear_app_overrides"}.isdisjoint(
                request.fixturenames
            ), "AuthNZ reset fixtures leaked"
            assert "isolated_test_environment" not in request.fixturenames
            assert "pg_temp_db" not in request.fixturenames

        @pytest.mark.unit
        def test_first_loop(request: pytest.FixtureRequest) -> None:
            loops.append(request.getfixturevalue("event_loop"))
            assert not loops[0].is_closed()

        @pytest.mark.unit
        def test_loop_lifetime(request: pytest.FixtureRequest) -> None:
            assert loops[0].is_closed(), "AuthNZ session event loop leaked"
            assert request.getfixturevalue("event_loop") is not loops[0]
    """
    )
    for bridge, failed in [("authnz_full_fixtures", 2), ("authnz_isolated_fixtures", 0)]:
        pytester.makeconftest(f"""
            import asyncpg
            import psycopg
            import pytest
            from typing import Any, NoReturn

            pytest_plugins = ["baseline_plugin", "tldw_Server_API.tests._plugins.{bridge}"]

            @pytest.fixture(autouse=True)
            def _forbid_pg_io(monkeypatch: pytest.MonkeyPatch) -> None:
                def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
                    pytest.fail("Registration probe attempted PostgreSQL I/O")
                monkeypatch.setattr(asyncpg, "connect", forbidden)
                monkeypatch.setattr(psycopg, "connect", forbidden)
        """)
        result = shared_guards._run_fixture_probe(
            pytester,
            monkeypatch,
            f"registration-{bridge}",
            "-p", "no:randomly",
        )
        result.assert_outcomes(passed=3 - failed, failed=failed)
        if failed:
            assert "AuthNZ reset fixtures leaked" in result.stdout.str()
            assert "AuthNZ session event loop leaked" in result.stdout.str()


def test_shared_fixture_selection_preserves_original(request: pytest.FixtureRequest) -> None:
    """The bridge must select the existing lifecycle with only native dependencies."""
    assert authnz_isolated_fixtures.isolated_test_environment is authnz_fixtures.isolated_test_environment
    assert "isolated_test_environment" not in request.fixturenames
    assert "pg_temp_db" not in request.fixturenames


def test_narrow_plugin_exports_only_original_lifecycle() -> None:
    """No autouse reset or custom loop may escape through the narrow namespace."""
    assert authnz_isolated_fixtures.isolated_test_environment is authnz_fixtures.isolated_test_environment
    assert authnz_isolated_fixtures.__all__ == ["isolated_test_environment"]
    assert {name for name in vars(authnz_isolated_fixtures) if name.isidentifier() and not name.startswith("_")} == {
        "isolated_test_environment"
    }

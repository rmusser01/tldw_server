"""Request leases must release PostgreSQL read locks before another owner opens."""

from contextlib import nullcontext

import pytest

from tldw_Server_API.app.api.v1.API_Deps import prompt_studio_deps as deps

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest.mark.parametrize("handler_fails", [False, True], ids=["normal", "handler-error"])
def test_request_lease_releases_read_before_next_tenant(pg_restricted_backend, tmp_path, monkeypatch, handler_fails):
    """Real repository reads cannot pin schema locks past their request lease."""
    deps.shutdown_prompt_studio_deps()
    monkeypatch.setattr(
        deps,
        "_get_prompt_studio_db_path_for_user",
        lambda user_id: tmp_path / user_id / "prompt-studio.sqlite",
    )
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: pg_restricted_backend)
    outcome = pytest.raises(RuntimeError, match="handler failed after read") if handler_fails else nullcontext()
    try:
        with outcome:
            with deps.managed_prompt_studio_db({"user_id": "99", "client_id": "web"}) as first:
                project = first.create_project("C01 Alice", user_id="99")
                prompt = first.create_prompt(project["id"], "C01 Pirate", system_prompt="Always say ARRR.")
                assert first.get_prompt_with_project(prompt["id"])["system_prompt"] == "Always say ARRR."  # nosec B101 - Causal request-scope ownership assertion.
                borrowed = first.get_connection().raw_connection
                assert borrowed.info.transaction_status.name == "INTRANS"  # nosec B101 - Causal request-scope ownership assertion.
                if handler_fails:
                    raise RuntimeError("handler failed after read")

        # Fail before the second constructor on the old code: it would block on
        # this reader's schema locks. Finally still releases the red connection.
        assert borrowed.info.transaction_status.name == "IDLE"  # nosec B101 - Causal request-scope ownership assertion.

        with deps.managed_prompt_studio_db({"user_id": "100", "client_id": "web"}) as second:
            project = second.create_project("C01 Bob", user_id="100")
            assert second.get_project(project["id"])["user_id"] == "100"  # nosec B101 - Causal request-scope ownership assertion.
            second_borrowed = second.get_connection().raw_connection
        assert second_borrowed.info.transaction_status.name == "IDLE"  # nosec B101 - Causal request-scope ownership assertion.
    finally:
        deps.shutdown_prompt_studio_deps()

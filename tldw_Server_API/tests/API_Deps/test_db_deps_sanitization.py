import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.api.v1.API_Deps import DB_Deps as deps


class _User:
    id = "user-42"
    id_int = 42


def test_content_backend_initialization_failure_log_is_sanitized(monkeypatch):
    secret = "postgresql://user:super-secret-token@localhost/content"
    messages: list[str] = []
    sink_id = deps.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        format="{message}",
    )

    def _raise_backend_failure():
        raise RuntimeError(f"could not connect to {secret}")

    monkeypatch.setattr(deps, "get_content_backend_instance", _raise_backend_failure)

    try:
        with pytest.raises(HTTPException) as exc_info:
            deps._get_or_create_media_db_factory(_User())
    finally:
        deps.logger.remove(sink_id)

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert (
        exc_info.value.detail
        == "PostgreSQL content backend required but unavailable. Check server logs."
    )

    rendered_logs = "\n".join(messages)
    assert "Content backend initialization failed" in rendered_logs
    assert "could not connect" not in rendered_logs
    assert secret not in rendered_logs
    assert "super-secret-token" not in rendered_logs

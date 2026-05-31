import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import setup_deps
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


def _make_request() -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/setup/status",
        "headers": [],
        "client": ("127.0.0.1", 1234),
    }
    return Request(scope)


def _make_remote_post_request(path="/api/v1/setup/first-run/state") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(b"host", b"example.test")],
        "client": ("203.0.113.10", 4444),
    }
    return Request(scope)


def _make_local_proxied_post_request(
    forwarded_for: str,
    *,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/setup/first-run/state",
        "headers": [
            (b"host", b"localhost"),
            (b"x-forwarded-for", forwarded_for.encode("ascii")),
        ]
        + (extra_headers or []),
        "client": ("127.0.0.1", 4444),
    }
    return Request(scope)


def _capture_setup_deps_records() -> tuple[list[dict], int]:
    records: list[dict] = []
    sink_id = setup_deps.logger.add(
        lambda message: records.append(
            {
                "message": str(message.record.get("message") or ""),
                "extra": {
                    key: value
                    for key, value in dict(message.record.get("extra") or {}).items()
                    if value not in ("", None)
                },
                "exception": message.record.get("exception"),
            }
        ),
        level="DEBUG",
        format="{message}",
    )
    return records, sink_id


@pytest.mark.asyncio
async def test_require_admin_for_remote_rejects_non_admin(monkeypatch):
    async def fake_get_auth_principal(_request):
        return AuthPrincipal(kind="user", user_id=999, roles=[], permissions=[], is_admin=False)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_auth_principal",
        fake_get_auth_principal,
    )

    with pytest.raises(HTTPException) as excinfo:
        await setup_deps._require_admin_for_remote(_make_request())

    assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_require_admin_for_remote_allows_admin_role(monkeypatch):
    async def fake_get_auth_principal(_request):
        return AuthPrincipal(
            kind="user",
            user_id=999,
            roles=["admin"],
            permissions=["system.configure"],
            is_admin=False,
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_auth_principal",
        fake_get_auth_principal,
    )

    await setup_deps._require_admin_for_remote(_make_request())


@pytest.mark.asyncio
async def test_require_admin_for_remote_rejects_boolean_admin_without_claims(monkeypatch):
    async def fake_get_auth_principal(_request):
        return AuthPrincipal(kind="user", user_id=999, roles=["user"], permissions=[], is_admin=True)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_auth_principal",
        fake_get_auth_principal,
    )

    with pytest.raises(HTTPException) as excinfo:
        await setup_deps._require_admin_for_remote(_make_request())

    assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_require_admin_for_remote_allows_system_configure_permission(monkeypatch):
    async def fake_get_auth_principal(_request):
        return AuthPrincipal(
            kind="user",
            user_id=999,
            roles=["user"],
            permissions=["system.configure"],
            is_admin=False,
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_auth_principal",
        fake_get_auth_principal,
    )

    await setup_deps._require_admin_for_remote(_make_request())


@pytest.mark.asyncio
async def test_require_local_setup_access_calls_admin_guard(monkeypatch):
    called = {"value": False}

    async def fake_guard(_request):
        called["value"] = True

    monkeypatch.setenv("TLDW_SETUP_ALLOW_REMOTE", "1")
    monkeypatch.setattr(setup_deps, "_config_allows_remote", lambda: False)
    monkeypatch.setattr(setup_deps, "_require_admin_for_remote", fake_guard)

    await setup_deps.require_local_setup_access(_make_request())

    assert called["value"] is True


@pytest.mark.asyncio
async def test_remote_setup_write_rejected_when_remote_setup_disabled(monkeypatch):
    monkeypatch.delenv("TLDW_SETUP_ALLOW_REMOTE", raising=False)
    monkeypatch.setattr(setup_deps, "_config_allows_remote", lambda: False)

    with pytest.raises(HTTPException) as excinfo:
        await setup_deps.require_local_setup_access(_make_remote_post_request())

    assert excinfo.value.status_code == 403
    assert "localhost" in str(excinfo.value.detail).lower()


@pytest.mark.asyncio
async def test_remote_setup_write_requires_admin_guard_when_remote_override_enabled(monkeypatch):
    called = {"value": False}

    async def fake_guard(_request):
        called["value"] = True

    monkeypatch.setenv("TLDW_SETUP_ALLOW_REMOTE", "1")
    monkeypatch.setattr(setup_deps, "_config_allows_remote", lambda: False)
    monkeypatch.setattr(setup_deps, "_require_admin_for_remote", fake_guard)

    await setup_deps.require_local_setup_access(_make_remote_post_request())

    assert called["value"] is True


@pytest.mark.asyncio
async def test_local_setup_write_rejects_mixed_forwarded_for_chain(monkeypatch):
    monkeypatch.delenv("TLDW_SETUP_ALLOW_REMOTE", raising=False)
    monkeypatch.setattr(setup_deps, "_config_allows_remote", lambda: False)

    with pytest.raises(HTTPException) as excinfo:
        await setup_deps.require_local_setup_access(
            _make_local_proxied_post_request("127.0.0.1, 203.0.113.10")
        )

    rendered_detail = str(excinfo.value.detail)
    assert excinfo.value.status_code == 403
    assert "203.0.113.10" not in rendered_detail
    assert "127.0.0.1" not in rendered_detail


@pytest.mark.asyncio
async def test_local_setup_write_rejects_conflicting_forwarded_client_headers(monkeypatch):
    monkeypatch.delenv("TLDW_SETUP_ALLOW_REMOTE", raising=False)
    monkeypatch.setattr(setup_deps, "_config_allows_remote", lambda: False)

    with pytest.raises(HTTPException) as excinfo:
        await setup_deps.require_local_setup_access(
            _make_local_proxied_post_request(
                "127.0.0.1",
                extra_headers=[(b"forwarded", b"for=203.0.113.10")],
            )
        )

    rendered_detail = str(excinfo.value.detail)
    assert excinfo.value.status_code == 403
    assert "203.0.113.10" not in rendered_detail
    assert "127.0.0.1" not in rendered_detail


@pytest.mark.asyncio
async def test_local_setup_write_allows_loopback_forwarded_for_chain(monkeypatch):
    monkeypatch.delenv("TLDW_SETUP_ALLOW_REMOTE", raising=False)
    monkeypatch.setattr(setup_deps, "_config_allows_remote", lambda: False)

    await setup_deps.require_local_setup_access(
        _make_local_proxied_post_request("127.0.0.1, ::1")
    )


@pytest.mark.asyncio
async def test_require_shared_audio_installer_access_rejects_non_admin(monkeypatch):
    async def fake_get_auth_principal(_request):
        return AuthPrincipal(kind="user", user_id=999, roles=["user"], permissions=[], is_admin=False)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_auth_principal",
        fake_get_auth_principal,
    )

    with pytest.raises(HTTPException) as excinfo:
        await setup_deps.require_shared_audio_installer_access(_make_request())

    assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_require_shared_audio_installer_access_allows_admin_claims(monkeypatch):
    async def fake_get_auth_principal(_request):
        return AuthPrincipal(
            kind="user",
            user_id=999,
            roles=["admin"],
            permissions=["system.configure"],
            is_admin=False,
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_auth_principal",
        fake_get_auth_principal,
    )

    await setup_deps.require_shared_audio_installer_access(_make_request())


def test_config_allows_remote_sanitizes_config_read_fallback_log(monkeypatch):
    sensitive_marker = "RAW_SETUP_CONFIG_READ_MARKER"
    sensitive_path = "/private/tmp/setup/token-config.txt"
    sensitive_token = "setup-token-abc123"
    sensitive_detail = f"{sensitive_marker} failed for {sensitive_path} using {sensitive_token}"
    records, sink_id = _capture_setup_deps_records()

    def _raise_config_path_failure():
        raise RuntimeError(sensitive_detail)

    setup_deps.reset_remote_access_cache()
    monkeypatch.setattr(setup_deps.setup_manager, "get_config_file_path", _raise_config_path_failure)

    try:
        assert setup_deps._config_allows_remote() is False
    finally:
        setup_deps.logger.remove(sink_id)
        setup_deps.reset_remote_access_cache()

    assert records == [
        {
            "message": "Unable to read setup remote access configuration; using localhost-only default",
            "extra": {"error_type": "RuntimeError"},
            "exception": None,
        }
    ]

    rendered_record = "\n".join(
        f"{record['message']} {record['extra']} {record['exception']}" for record in records
    )
    assert "exc_info" not in rendered_record
    assert sensitive_marker not in rendered_record
    assert sensitive_path not in rendered_record
    assert sensitive_token not in rendered_record
    assert sensitive_detail not in rendered_record

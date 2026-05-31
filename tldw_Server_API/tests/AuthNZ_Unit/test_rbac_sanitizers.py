import pytest

from tldw_Server_API.app.core.AuthNZ import rbac as rbac_module


class _FailingRbacRepo:
    def get_effective_permissions(self, user_id: int) -> list[str]:
        raise RuntimeError("effective permission DB failed at /private/rbac.db")

    def has_permission(self, user_id: int, permission: str) -> bool:
        raise RuntimeError("permission check DB failed at /private/rbac-perm.db")


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str, *args, **kwargs) -> None:
        self.errors.append(message)


@pytest.mark.parametrize("_redact", [False, True])
def test_get_effective_permissions_failure_is_sanitized(monkeypatch, _redact):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(rbac_module, "_get_rbac_repo", lambda: _FailingRbacRepo())
    monkeypatch.setattr(rbac_module, "logger", logger_stub)

    with pytest.raises(rbac_module.RBACError) as exc_info:
        rbac_module.get_effective_permissions(123)

    assert str(exc_info.value) == "Failed to compute effective permissions"
    assert logger_stub.errors == ["RBAC effective permissions check failed"]
    combined = f"{logger_stub.errors}\n{exc_info.value}"
    assert "123" not in combined
    assert "effective permission DB failed" not in combined
    assert "/private/rbac.db" not in combined


@pytest.mark.parametrize("_redact", [False, True])
def test_user_has_permission_failure_is_sanitized(monkeypatch, _redact):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(rbac_module, "_get_rbac_repo", lambda: _FailingRbacRepo())
    monkeypatch.setattr(rbac_module, "logger", logger_stub)

    with pytest.raises(rbac_module.RBACError) as exc_info:
        rbac_module.user_has_permission(456, "secret.permission")

    assert str(exc_info.value) == "Failed to check permission"
    assert logger_stub.errors == ["RBAC permission check failed"]
    combined = f"{logger_stub.errors}\n{exc_info.value}"
    assert "456" not in combined
    assert "secret.permission" not in combined
    assert "permission check DB failed" not in combined
    assert "/private/rbac-perm.db" not in combined

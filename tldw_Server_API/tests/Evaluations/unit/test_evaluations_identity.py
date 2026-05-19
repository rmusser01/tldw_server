from __future__ import annotations

import importlib

import pytest

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


@pytest.mark.unit
def test_evaluations_identity_from_user_preserves_tenant_style_string():
    from tldw_Server_API.app.core.Evaluations.identity import evaluations_identity_from_user

    identity = evaluations_identity_from_user(
        User(id="tenant-user", username="tenant", email=None, is_active=True)
    )

    assert identity.user_scope == "tenant-user"
    assert identity.created_by == "tenant-user"
    assert identity.rate_limit_subject == "tenant-user"
    assert identity.webhook_user_id == "user_tenant-user"


@pytest.mark.unit
def test_evaluations_identity_from_user_falls_back_to_single_user_id():
    from tldw_Server_API.app.core.Evaluations.identity import evaluations_identity_from_user

    identity = evaluations_identity_from_user(
        User(id="", username="fallback", email=None, is_active=True),
        fallback=DatabasePaths.get_single_user_id(),
    )

    expected = str(DatabasePaths.get_single_user_id())
    assert identity.user_scope == expected
    assert identity.created_by == expected
    assert identity.rate_limit_subject == expected
    assert identity.webhook_user_id == f"user_{expected}"


@pytest.mark.unit
def test_canonical_evaluations_user_scope_requires_explicit_fallback_when_missing():
    from tldw_Server_API.app.core.Evaluations.identity import canonical_evaluations_user_scope

    with pytest.raises(ValueError, match="Evaluations user scope is required"):
        canonical_evaluations_user_scope(
            User(id="", username="missing", email=None, is_active=True)
        )


@pytest.mark.unit
def test_canonical_evaluations_user_scope_retries_id_when_id_str_is_blank():
    from tldw_Server_API.app.core.Evaluations.identity import canonical_evaluations_user_scope

    class _UserLike:
        id_str = ""
        id = "tenant-user"

    assert canonical_evaluations_user_scope(_UserLike()) == "tenant-user"


@pytest.mark.unit
def test_unified_service_cache_uses_canonical_string_scope(monkeypatch: pytest.MonkeyPatch):
    import tldw_Server_API.app.core.Evaluations.unified_evaluation_service as service_module

    service_module = importlib.reload(service_module)
    created: dict[str, str] = {}

    class _DummyService:
        def __init__(self, db_path: str, **_kwargs):
            created["db_path"] = db_path

    monkeypatch.setattr(service_module, "UnifiedEvaluationService", _DummyService)
    service_module._service_instances_by_user.clear()

    service = service_module.get_unified_evaluation_service_for_user("tenant-user")

    assert isinstance(service, _DummyService)
    assert created["db_path"] == str(DatabasePaths.get_evaluations_db_path("tenant-user"))
    assert "tenant-user" in service_module._service_instances_by_user


@pytest.mark.unit
def test_rate_limiter_cache_uses_canonical_string_scope(monkeypatch: pytest.MonkeyPatch):
    import tldw_Server_API.app.core.Evaluations.user_rate_limiter as limiter_module

    limiter_module = importlib.reload(limiter_module)
    created: dict[str, str] = {}

    class _DummyLimiter:
        def __init__(self, db_path: str):
            created["db_path"] = db_path

    monkeypatch.setattr(limiter_module, "UserRateLimiter", _DummyLimiter)
    monkeypatch.setattr(limiter_module, "is_test_mode", lambda: False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    limiter_module._user_rate_limiter_instances.clear()

    limiter = limiter_module.get_user_rate_limiter_for_user("tenant-user")

    assert isinstance(limiter, _DummyLimiter)
    assert created["db_path"] == str(DatabasePaths.get_evaluations_db_path("tenant-user"))
    assert "tenant-user" in limiter_module._user_rate_limiter_instances


@pytest.mark.unit
def test_evaluation_manager_preserves_string_user_scope(monkeypatch: pytest.MonkeyPatch):
    import tldw_Server_API.app.core.Evaluations.evaluation_manager as manager_module

    monkeypatch.setattr(manager_module.EvaluationManager, "_init_database", lambda self: None, raising=False)

    manager = manager_module.EvaluationManager(user_id="tenant-user")

    assert manager.db_path == DatabasePaths.get_evaluations_db_path("tenant-user")

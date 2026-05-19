from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import moderation as moderation_mod
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Moderation.moderation_service import ModerationPolicy, ModerationService, PatternRule
from tldw_Server_API.app.core.Moderation import supervised_policy as supervised_policy_mod


def _make_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[SYSTEM_CONFIGURE],
        is_admin=True,
        org_ids=[1],
        team_ids=[],
    )


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(moderation_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = _make_principal()
        request.state.auth = AuthContext(
            principal=principal,
            ip=None,
            user_agent=None,
            request_id=None,
        )
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    return app


@pytest.mark.unit
def test_moderation_test_sample_matches_selected_rule(monkeypatch):
    svc = ModerationService()
    rule_warn = PatternRule(regex=re.compile(r"alpha", re.IGNORECASE), action="warn")
    rule_block = PatternRule(regex=re.compile(r"beta", re.IGNORECASE), action="block")
    svc._global_policy = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        output_enabled=True,
        input_action="warn",
        output_action="warn",
        redact_replacement="[REDACTED]",
        per_user_overrides=False,
        block_patterns=[rule_warn, rule_block],
        categories_enabled=None,
    )

    monkeypatch.setattr(moderation_mod, "get_moderation_service", lambda: svc)

    app = _build_app()
    text = "alpha " + ("x" * 60) + " beta"

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/moderation/test",
            json={"text": text, "phase": "input"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("action") == "block"
    sample = body.get("sample")
    assert sample
    assert "[REDACTED]" in sample
    assert "alpha" not in sample.lower()
    assert "beta" not in sample.lower()


@pytest.mark.unit
def test_moderation_test_guardian_overlay_requires_user_id(monkeypatch):
    svc = ModerationService()
    monkeypatch.setattr(moderation_mod, "get_moderation_service", lambda: svc)

    app = _build_app()

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/moderation/test",
            json={
                "text": "guardian check",
                "phase": "input",
                "apply_guardian_overlay": True,
            },
        )

    assert resp.status_code == 400
    assert resp.json().get("detail") == "user_id is required when apply_guardian_overlay=true"


@pytest.mark.unit
def test_moderation_test_rejects_mismatched_guardian_simulation_ids(monkeypatch):
    svc = ModerationService()
    monkeypatch.setattr(moderation_mod, "get_moderation_service", lambda: svc)

    app = _build_app()

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/moderation/test",
            json={
                "text": "guardian check",
                "phase": "input",
                "user_id": "guardian-user",
                "dependent_user_id": "child-user",
                "apply_guardian_overlay": True,
            },
        )

    assert resp.status_code == 400
    assert resp.json().get("detail") == (
        "dependent_user_id must match user_id for live-chat guardian simulation"
    )


@pytest.mark.unit
def test_moderation_test_guardian_overlay_returns_overlaid_effective_policy(monkeypatch):
    svc = ModerationService()
    svc._global_policy = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        output_enabled=True,
        input_action="warn",
        output_action="warn",
        redact_replacement="[REDACTED]",
        per_user_overrides=True,
        block_patterns=[],
        categories_enabled=None,
    )
    captured: dict[str, tuple[str, ...] | tuple[str, str]] = {}

    class _FakeEngine:
        def build_moderation_policy_overlay(self, dependent_user_id, base_policy, chat_type=None):
            captured["overlay"] = (str(dependent_user_id), str(chat_type))
            return ModerationPolicy(
                enabled=True,
                input_enabled=base_policy.input_enabled,
                output_enabled=base_policy.output_enabled,
                input_action=base_policy.input_action,
                output_action=base_policy.output_action,
                redact_replacement=base_policy.redact_replacement,
                per_user_overrides=base_policy.per_user_overrides,
                block_patterns=[
                    PatternRule(
                        regex=re.compile(r"guardian topic", re.IGNORECASE),
                        action="block",
                    )
                ],
                categories_enabled=base_policy.categories_enabled,
            )

        def check_text(self, text, dependent_user_id, phase="input", chat_type=None):
            from tldw_Server_API.app.core.Moderation.supervised_policy import SupervisedCheckResult
            return SupervisedCheckResult()

    def _fake_bootstrap_guardian_runtime(*, user_id, dependent_user_id, chat_type):
        captured["bootstrap"] = (str(user_id), str(dependent_user_id), str(chat_type))
        return SimpleNamespace(
            supervised_engine=_FakeEngine(),
            dependent_user_id=str(dependent_user_id),
            chat_type=str(chat_type),
        )

    monkeypatch.setattr(moderation_mod, "get_moderation_service", lambda: svc)
    monkeypatch.setattr(
        moderation_mod,
        "bootstrap_guardian_moderation_runtime",
        _fake_bootstrap_guardian_runtime,
    )

    app = _build_app()

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/moderation/test",
            json={
                "text": "guardian topic appears here",
                "phase": "input",
                "user_id": "alice",
                "apply_guardian_overlay": True,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "block"
    assert captured["bootstrap"] == ("alice", "alice", "regular")
    assert captured["overlay"] == ("alice", "regular")
    assert any(rule["pattern"] == "guardian topic" for rule in body["effective"]["rules"])


@pytest.mark.unit
def test_moderation_test_uses_canonical_evaluate_text_contract(monkeypatch):
    class _Policy:
        def to_dict(self):
            return {"enabled": True, "rules": ["canonical"]}

    class _StubService:
        def get_effective_policy(self, user_id):
            assert user_id == "alice"
            return _Policy()

        def evaluate_text(self, text, policy, phase):
            assert text == "canonical trigger"
            assert phase == "input"
            assert isinstance(policy, _Policy)
            return SimpleNamespace(
                action="warn",
                sample="[REDACTED] trigger",
                redacted_text=None,
                category="policy",
            )

        def evaluate_action(self, *_args, **_kwargs):
            raise AssertionError("legacy evaluate_action() should not be used")

        def evaluate_action_with_match(self, *_args, **_kwargs):
            raise AssertionError("legacy evaluate_action_with_match() should not be used")

    monkeypatch.setattr(moderation_mod, "get_moderation_service", lambda: _StubService())

    app = _build_app()

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/moderation/test",
            json={
                "text": "canonical trigger",
                "phase": "input",
                "user_id": "alice",
            },
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "flagged": True,
        "action": "warn",
        "sample": "[REDACTED] trigger",
        "redacted_text": None,
        "effective": {"enabled": True, "rules": ["canonical"]},
        "category": "policy",
    }


@pytest.mark.unit
def test_bootstrap_guardian_moderation_runtime_uses_core_guardian_db_resolver(monkeypatch):
    sentinel_db = object()
    captured: dict[str, object] = {}

    def _fake_resolve_guardian_db_for_user_id(user_id):
        captured["user_id"] = user_id
        return sentinel_db

    def _fake_get_supervised_policy_engine(guardian_db):
        captured["guardian_db"] = guardian_db
        return "engine"

    monkeypatch.setattr(
        supervised_policy_mod,
        "resolve_guardian_db_for_user_id",
        _fake_resolve_guardian_db_for_user_id,
    )
    monkeypatch.setattr(
        supervised_policy_mod,
        "get_supervised_policy_engine",
        _fake_get_supervised_policy_engine,
    )

    runtime = supervised_policy_mod.bootstrap_guardian_moderation_runtime(
        user_id="parent",
        dependent_user_id="child",
        chat_type="character",
    )

    assert captured["user_id"] == "parent"
    assert captured["guardian_db"] is sentinel_db
    assert runtime.guardian_db is sentinel_db
    assert runtime.supervised_engine == "engine"
    assert runtime.dependent_user_id == "child"
    assert runtime.chat_type == "character"


@pytest.mark.unit
def test_effective_policy_merges_user_rules_and_returns_warn_action():
    svc = ModerationService()
    svc._global_policy = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        output_enabled=True,
        input_action="warn",
        output_action="warn",
        redact_replacement="[REDACTED]",
        per_user_overrides=True,
        block_patterns=[],
        categories_enabled=None,
    )
    svc._user_overrides = {
        "alice": {
            "rules": [
                {
                    "id": "r1",
                    "pattern": "heads up",
                    "is_regex": False,
                    "action": "warn",
                    "phase": "both",
                }
            ]
        }
    }

    policy = svc.get_effective_policy("alice")
    action, _, sample, _ = svc.evaluate_action("please heads up now", policy, "output")
    assert action == "warn"
    assert sample


@pytest.mark.unit
def test_effective_policy_user_rules_apply_with_category_filter_enabled():
    svc = ModerationService()
    svc._global_policy = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        output_enabled=True,
        input_action="warn",
        output_action="warn",
        redact_replacement="[REDACTED]",
        per_user_overrides=True,
        block_patterns=[],
        categories_enabled={"pii"},
    )
    svc._user_overrides = {
        "alice": {
            "rules": [
                {
                    "id": "r1",
                    "pattern": "heads up",
                    "is_regex": False,
                    "action": "block",
                    "phase": "both",
                }
            ]
        }
    }

    policy = svc.get_effective_policy("alice")
    action, category, _, _ = svc.evaluate_action("please heads up now", policy, "input")
    flagged, _ = svc.check_text("please heads up now", policy, phase="input")

    assert action == "block"
    assert flagged is True
    assert category is None

from __future__ import annotations

import json

import pytest


class FakeUser:
    def __init__(
        self,
        user_id: int,
        *,
        active_org_id: int | None = None,
        org_ids: list[int] | None = None,
    ) -> None:
        self.id = user_id
        self.active_org_id = active_org_id
        self.org_ids = org_ids or []


def _flag(**overrides):
    payload = {
        "key": "ingestion_sources.local_directory",
        "scope": "global",
        "enabled": True,
        "org_id": None,
        "user_id": None,
        "target_user_ids": [],
        "rollout_percent": 100,
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def access_policy(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Sources import access_policy as module

    monkeypatch.setattr(module, "list_feature_flags", lambda: [])
    monkeypatch.setattr(module, "is_single_user_mode", lambda: False)
    return module


def test_single_user_mode_allows_local_directory_creation(access_policy, monkeypatch):
    monkeypatch.setattr(access_policy, "is_single_user_mode", lambda: True)

    assert access_policy.can_create_local_directory_ingestion_source(FakeUser(1)) is True


def test_multi_user_mode_denies_without_feature_flag(access_policy):
    assert access_policy.can_create_local_directory_ingestion_source(FakeUser(1)) is False


def test_user_scoped_flag_allows_exact_user(access_policy, monkeypatch):
    monkeypatch.setattr(access_policy, "list_feature_flags", lambda: [_flag(scope="user", user_id=7)])

    assert access_policy.can_create_local_directory_ingestion_source(FakeUser(7)) is True
    assert access_policy.can_create_local_directory_ingestion_source(FakeUser(8)) is False


def test_org_scoped_flag_allows_active_org_member(access_policy, monkeypatch):
    monkeypatch.setattr(access_policy, "list_feature_flags", lambda: [_flag(scope="org", org_id=42)])

    assert access_policy.can_create_local_directory_ingestion_source(
        FakeUser(7, active_org_id=42)
    ) is True


def test_org_scoped_flag_allows_secondary_org_member(access_policy, monkeypatch):
    monkeypatch.setattr(access_policy, "list_feature_flags", lambda: [_flag(scope="org", org_id=42)])

    assert access_policy.can_create_local_directory_ingestion_source(
        FakeUser(7, org_ids=[41, 42])
    ) is True


def test_org_scoped_flag_uses_explicit_scope_when_user_has_no_org_attrs(access_policy, monkeypatch):
    monkeypatch.setattr(access_policy, "list_feature_flags", lambda: [_flag(scope="org", org_id=42)])

    user = type("RealisticUser", (), {"id": 7})()

    assert access_policy.can_create_local_directory_ingestion_source(
        user,
        active_org_id=42,
        org_ids=[41],
    ) is True


def test_disabled_flags_do_not_allow(access_policy, monkeypatch):
    monkeypatch.setattr(
        access_policy,
        "list_feature_flags",
        lambda: [
            _flag(scope="global", enabled=False),
            _flag(scope="org", org_id=42, enabled=False),
            _flag(scope="user", user_id=7, enabled=False),
        ],
    )

    assert access_policy.can_create_local_directory_ingestion_source(
        FakeUser(7, active_org_id=42)
    ) is False


def test_global_flags_do_not_allow_local_directory_creation(access_policy, monkeypatch):
    monkeypatch.setattr(
        access_policy,
        "list_feature_flags",
        lambda: [
            _flag(scope="global"),
            _flag(scope="global", target_user_ids=[7]),
        ],
    )

    assert access_policy.can_create_local_directory_ingestion_source(FakeUser(7)) is False
    assert access_policy.can_create_local_directory_ingestion_source(FakeUser(8)) is False


def test_target_user_ids_narrows_org_flags(access_policy, monkeypatch):
    monkeypatch.setattr(
        access_policy,
        "list_feature_flags",
        lambda: [_flag(scope="org", org_id=42, target_user_ids=[7])],
    )

    assert access_policy.can_create_local_directory_ingestion_source(
        FakeUser(7, active_org_id=42)
    ) is True
    assert access_policy.can_create_local_directory_ingestion_source(
        FakeUser(8, active_org_id=42)
    ) is False


def test_non_matching_flag_key_is_ignored(access_policy, monkeypatch):
    monkeypatch.setattr(
        access_policy,
        "list_feature_flags",
        lambda: [_flag(key="some.other.flag", scope="global")],
    )

    assert access_policy.can_create_local_directory_ingestion_source(FakeUser(7)) is False


def test_rollout_percent_is_deterministic_per_user(access_policy, monkeypatch):
    monkeypatch.setattr(
        access_policy,
        "list_feature_flags",
        lambda: [_flag(scope="user", user_id=7, rollout_percent=50)],
    )

    first = access_policy.can_create_local_directory_ingestion_source(FakeUser(7))
    second = access_policy.can_create_local_directory_ingestion_source(FakeUser(7))

    assert first is second


def test_missing_rollout_percent_defaults_to_full_rollout(access_policy, monkeypatch):
    monkeypatch.setattr(
        access_policy,
        "list_feature_flags",
        lambda: [_flag(scope="user", user_id=7, rollout_percent=None)],
    )

    assert access_policy.can_create_local_directory_ingestion_source(FakeUser(7)) is True


def test_malformed_rollout_percent_fails_closed(access_policy, monkeypatch):
    monkeypatch.setattr(
        access_policy,
        "list_feature_flags",
        lambda: [_flag(scope="user", user_id=7, rollout_percent="not-a-number")],
    )

    assert access_policy.can_create_local_directory_ingestion_source(FakeUser(7)) is False


def test_malformed_persisted_rollout_percent_from_feature_flag_service_fails_closed(
    access_policy,
    monkeypatch,
    tmp_path,
):
    from tldw_Server_API.app.services import admin_system_ops_service

    store_path = tmp_path / "system_ops.json"
    store_path.write_text(
        json.dumps(
            {
                "feature_flags": [
                    _flag(scope="user", user_id=7, rollout_percent="not-a-number")
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(admin_system_ops_service, "_STORE_PATH", store_path)
    monkeypatch.setattr(access_policy, "list_feature_flags", admin_system_ops_service.list_feature_flags)

    assert access_policy.can_create_local_directory_ingestion_source(FakeUser(7)) is False

from __future__ import annotations

import pytest


def test_provider_identity_contract_is_canonical_first_and_deterministic():
    from tldw_Server_API.app.core.LLM_Calls.provider_identity import (
        canonical_provider_name,
        provider_lookup_names,
    )

    assert canonical_provider_name("OAI") == "openai"
    assert provider_lookup_names("oai") == ("openai", "oai")
    assert provider_lookup_names("openai-compatible") == (
        "custom-openai-api",
        "custom_openai_api",
        "custom-openai",
        "openai-compatible",
        "customopenai",
    )


def test_legacy_storage_normalizer_remains_noncanonical():
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import normalize_provider_name

    assert normalize_provider_name(" OAI ") == "oai"


def test_default_allowlist_accepts_registered_alias(monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    monkeypatch.setattr(
        byok_helpers,
        "get_settings",
        lambda: SimpleNamespace(BYOK_ALLOWED_PROVIDERS=[]),
    )

    assert byok_helpers.is_provider_allowlisted("oai") is True


def test_validate_credential_fields_default_allowlist():
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import validate_credential_fields

    fields = {"org_id": "org-123", "project_id": "proj-456"}
    cleaned = validate_credential_fields("unknown-provider", fields)
    assert cleaned == fields

    with pytest.raises(ValueError):
        validate_credential_fields("unknown-provider", {"api_key": "nope"})


def test_validate_credential_fields_required_policy(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import validate_credential_fields
    from tldw_Server_API.app.core.LLM_Calls import provider_metadata

    monkeypatch.setitem(
        provider_metadata.BYOK_CREDENTIAL_FIELDS,
        "test-provider",
        {"allowed": {"org_id"}, "required": {"org_id"}},
    )

    cleaned = validate_credential_fields("test-provider", {"org_id": "org-789"})
    assert cleaned == {"org_id": "org-789"}

    with pytest.raises(ValueError):
        validate_credential_fields("test-provider", {})


def test_trusted_base_url_request_rejects_legacy_boolean_only_user_shape():
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import is_trusted_base_url_request

    assert (
        is_trusted_base_url_request(
            None,
            principal=None,
            user={
                "role": "user",
                "roles": ["user"],
                "permissions": [],
                "is_admin": True,
                "is_superuser": True,
            },
        )
        is False
    )


def test_trusted_base_url_request_accepts_permission_claims_from_legacy_user_shape():
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import is_trusted_base_url_request

    assert (
        is_trusted_base_url_request(
            None,
            principal=None,
            user={
                "role": "user",
                "roles": ["user"],
                "permissions": ["system.configure"],
                "is_admin": False,
                "is_superuser": False,
            },
        )
        is True
    )

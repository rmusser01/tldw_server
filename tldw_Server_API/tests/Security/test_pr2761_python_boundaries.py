"""Behavioral evidence for individually traced PR 2761 CodeQL boundaries."""

from __future__ import annotations

import secrets
from types import SimpleNamespace

import pytest
from lxml.html import fromstring

from tldw_Server_API.app.core.AuthNZ import byok_runtime
from tldw_Server_API.app.core.exceptions import NoteAttachmentPolicyError
from tldw_Server_API.app.core.Notes import attachment_policy
from tldw_Server_API.app.core.Notes_Graph.suggestion_api import (
    build_notes_graph_suggestions_api,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.mcts_optimizer import (
    MCTSOptimizer,
)
from tldw_Server_API.app.core.TTS.gateway_config import normalize_gateway_specs
from tldw_Server_API.app.core.Web_Scraping.selectors.engine import (
    _select_nodes_with_status,
    _selector_validation_error,
    compile_selector,
)

pytestmark = pytest.mark.unit


def test_gateway_generation_tracks_permission_boolean_and_excludes_api_key() -> None:
    """2666's SARIF source is allow_user_api_key, a permission boolean."""
    config = {
        "enabled": True,
        "base_url": "https://speech.example.com/v1",
        "speech_path": "audio/speech",
        "default_model": "model-one",
        "default_voice": "voice-one",
        "api_key": "first-provider-credential",
        "allow_user_api_key": False,
    }
    first = normalize_gateway_specs({}, {"company": config})["gateway:company"]
    rotated = normalize_gateway_specs(
        {}, {"company": {**config, "api_key": "rotated-provider-credential"}}
    )["gateway:company"]
    permission = normalize_gateway_specs(
        {}, {"company": {**config, "allow_user_api_key": True}}
    )["gateway:company"]
    assert rotated.config_generation == first.config_generation
    assert permission.config_generation != first.config_generation


@pytest.mark.parametrize("field", ["api_key", "password", "access_token"])
def test_mcts_fingerprint_drops_credentials_at_both_mapping_levels(field: str) -> None:
    """2615's api_key cannot survive the recursive durable-data projection."""
    config = {"provider": "openai", "model": "model-one", "temperature": 0.4}
    with_credentials = {
        **config,
        field: "request-credential",
        "app_config": {"temperature": 0.4, field: "nested-request-credential"},
    }
    without_credentials = {**config, "app_config": {"temperature": 0.4}}
    assert MCTSOptimizer._model_behavior_fingerprint(
        with_credentials
    ) == MCTSOptimizer._model_behavior_fingerprint(without_credentials)


@pytest.mark.parametrize(
    "generation",
    [
        byok_runtime.openai_oauth_credential_generation,
        byok_runtime.openai_oauth_refresh_state_generation,
    ],
)
def test_oauth_generations_track_access_tokens_not_refresh_secrets(generation) -> None:
    """2613/2614 identify access-token versions for refresh coalescing."""
    access_token, rotated_access, refresh_token, rotated_refresh = (
        secrets.token_urlsafe(32) for _ in range(4)
    )
    oauth = {"access_token": access_token, "refresh_token": refresh_token}
    first = generation({"credential_version": 2, "credentials": {"oauth": oauth}})
    refresh_rotated = generation(
        {"credential_version": 2, "credentials": {"oauth": {**oauth, "refresh_token": rotated_refresh}}}
    )
    access_rotated = generation(
        {"credential_version": 2, "credentials": {"oauth": {**oauth, "access_token": rotated_access}}}
    )
    assert first == refresh_rotated
    assert first != access_rotated
    assert first is not None and len(first) == 64


def test_admin_gateway_revision_changes_when_provider_key_rotates() -> None:
    """2663's revision invalidates a provider cache; it authenticates no user."""
    first = byok_runtime._gateway_admin_credential_revision("provider-key-one")
    repeated = byok_runtime._gateway_admin_credential_revision("provider-key-one")
    rotated = byok_runtime._gateway_admin_credential_revision("provider-key-two")
    assert first == repeated
    assert first != rotated
    assert len(first) == 64


def test_notes_cursor_factory_binds_signature_to_configured_secret(monkeypatch) -> None:
    """2662 normalizes a configured MAC key; cursors cannot cross key rotation."""
    from tldw_Server_API.app.core.AuthNZ import settings
    from tldw_Server_API.app.core.Notes_Graph import suggestion_service

    secret_config = SimpleNamespace(JWT_SECRET_KEY="a" * 32, SINGLE_USER_API_KEY=None)
    monkeypatch.setattr(settings, "get_settings", lambda: secret_config)
    monkeypatch.setattr(
        suggestion_service, "build_suggestion_decision_service", lambda **_kwargs: None
    )
    kwargs = {
        "note_db": SimpleNamespace(note_graph_suggestion_store=object()),
        "owner_user_id": "owner-one",
        "dataset_id": "dataset-one",
        "jobs": None,
    }
    first = build_notes_graph_suggestions_api(**kwargs)
    binding = {"owner": "owner-one", "dataset": "dataset-one"}
    position = ("2026-09-10", "row-one")
    cursor = first._cursor.encode(binding=binding, position=position)
    assert first._cursor.decode(cursor, binding=binding) == position
    with pytest.raises(ValueError, match="notes_graph_cursor_invalid"):
        first._cursor.decode(cursor, binding={**binding, "owner": "owner-two"})
    secret_config.JWT_SECRET_KEY = "b" * 32
    rotated = build_notes_graph_suggestions_api(**kwargs)
    with pytest.raises(ValueError, match="notes_graph_cursor_invalid"):
        rotated._cursor.decode(cursor, binding=binding)


@pytest.mark.parametrize("cache", [False, True])
def test_xpath_compilation_only_queries_the_supplied_document(cache: bool) -> None:
    """2600/2601 compile user-authored extraction expressions by design."""
    first = fromstring("<main><p>first-document</p></main>")
    second = fromstring("<main><p>second-document</p></main>")
    selector = compile_selector("//p/text()", cache=cache)
    assert selector(first) == ["first-document"]
    assert selector(second) == ["second-document"]


@pytest.mark.parametrize("max_results", [None, 1])
def test_xpath_document_function_cannot_read_external_content(max_results, tmp_path) -> None:
    """Both normal and 2602's bounded evaluator lack XSLT document() I/O."""
    secret_file = tmp_path / "secret.xml"
    secret_file.write_text("<secret>private-document</secret>")
    expression = f'document("{secret_file.as_uri()}")/secret/text()'
    result, failed = _select_nodes_with_status(
        fromstring("<main><p>provided</p></main>"),
        expression,
        max_results=max_results,
    )
    assert result == []
    assert failed is True


@pytest.mark.parametrize("expression", ["$secret", "//p | //secret", "//p/parent::main"])
def test_xpath_runtime_and_validation_reject_disallowed_grammar(expression: str) -> None:
    assert _selector_validation_error(expression).startswith("selector_too_complex:")
    assert _select_nodes_with_status(fromstring("<main><p>value</p></main>"), expression) == (
        [],
        False,
    )


def test_mime_exact_limit_and_one_over_are_distinct() -> None:
    """2598's fullmatch receives at most 255 characters."""
    exact = "a/" + "!" * 253
    assert attachment_policy.validate_note_attachment_content_type(exact) == exact
    with pytest.raises(NoteAttachmentPolicyError):
        attachment_policy.validate_note_attachment_content_type(exact + "!")


def test_oversized_mime_is_rejected_before_regex_evaluation(monkeypatch) -> None:
    class MustNotEvaluate:
        def fullmatch(self, _value):
            pytest.fail("oversized input reached the regex")

    monkeypatch.setattr(attachment_policy, "_MEDIA_TYPE_RE", MustNotEvaluate())
    with pytest.raises(NoteAttachmentPolicyError):
        attachment_policy.validate_note_attachment_content_type("!" * 1_000_000)


@pytest.mark.asyncio
async def test_csrf_api_key_lookup_binds_user_id_instead_of_presented_credential(
    monkeypatch,
) -> None:
    """2264's tainted lookup result selects user_id before the HMAC message."""
    from starlette.requests import Request

    from tldw_Server_API.app.core.AuthNZ import api_key_manager, csrf_protection

    config = csrf_protection.get_settings().model_copy(update={"CSRF_BIND_TO_USER": True})
    monkeypatch.setattr(csrf_protection, "get_settings", lambda: config)
    lookup_result = {"user_id": 42}

    async def validate(**_kwargs):
        return lookup_result

    async def get_manager():
        return SimpleNamespace(validate_api_key=validate)

    monkeypatch.setattr(api_key_manager, "get_api_key_manager", get_manager)
    middleware = csrf_protection.CSRFProtectionMiddleware(app=None)
    suffixes = []
    for _ in range(2):
        credential = secrets.token_urlsafe(32)
        request = Request(
            {
                "type": "http",
                "headers": [(b"x-api-key", credential.encode())],
                "client": ("127.0.0.1", 1234),
            }
        )
        user_id = await middleware._resolve_user_id(request)
        assert user_id == 42
        suffixes.append(middleware.token_manager._bind_suffix(user_id))
    assert suffixes[0] == suffixes[1]
    assert suffixes[0] != middleware.token_manager._bind_suffix(43)


@pytest.mark.parametrize("algorithm", ["RS256", "ES256"])
def test_notes_cursor_asymmetric_jwt_rejects_public_fallback_forgery(
    monkeypatch, algorithm: str
) -> None:
    """Private-key JWT configurations must never fall back to a public MAC key."""
    import hashlib

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec, rsa

    from tldw_Server_API.app.core.AuthNZ import settings
    from tldw_Server_API.app.core.Notes_Graph import suggestion_service
    from tldw_Server_API.app.core.Notes_Graph.suggestion_api import (
        OpaqueSuggestionCursorCodec,
    )

    private_key = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        if algorithm == "RS256"
        else ec.generate_private_key(ec.SECP256R1())
    )
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    config = settings.Settings(
        _env_file=None,
        AUTH_MODE="multi_user",
        JWT_ALGORITHM=algorithm,
        JWT_PRIVATE_KEY=private_pem,
        JWT_SECRET_KEY=None,
        SINGLE_USER_API_KEY=None,
        API_KEY_PEPPER=None,
    )
    assert config.JWT_SECRET_KEY is None
    monkeypatch.setattr(settings, "get_settings", lambda: config)
    monkeypatch.setattr(
        suggestion_service, "build_suggestion_decision_service", lambda **_kwargs: None
    )
    facade = build_notes_graph_suggestions_api(
        note_db=SimpleNamespace(note_graph_suggestion_store=object()),
        owner_user_id="owner-one",
        dataset_id="dataset-one",
        jobs=None,
    )
    binding = {"owner": "owner-one", "dataset": "dataset-one"}
    position = ("2026-09-10", "row-one")
    forged = OpaqueSuggestionCursorCodec(
        hashlib.sha256(b"notes-graph-cursor-local").digest()
    ).encode(binding=binding, position=position)
    with pytest.raises(ValueError, match="notes_graph_cursor_invalid"):
        facade._cursor.decode(forged, binding=binding)
    authentic = facade._cursor.encode(binding=binding, position=position)
    assert facade._cursor.decode(authentic, binding=binding) == position

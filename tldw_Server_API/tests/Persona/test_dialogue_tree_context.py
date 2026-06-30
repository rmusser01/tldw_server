import pytest


pytestmark = pytest.mark.unit


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_runtime_context_redacts_secret_like_values_before_provider_view() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_context import (
        build_runtime_tree_context,
    )

    context = build_runtime_tree_context(
        persona_id="p1",
        session_id="s1",
        user_message="hello",
        policy_snapshot={"allow": ["chat"], "authorization": "Bearer secret-token"},
        memory_entries=[{"id": "m1", "content": "safe note", "api_key": "sk-test"}],
        state_docs=[{"id": "doc1", "content": "state text"}],
        exemplar_sections=[("persona_exemplars", "style anchor", 12)],
        tool_results=[{"tool": "web", "raw": "private external response"}],
    )

    provider_payload = context.for_generator()
    serialized = repr(provider_payload)

    _check("secret-token" not in serialized, "authorization token leaked")
    _check("sk-test" not in serialized, "api_key leaked")
    _check("private external response" not in serialized, "raw tool response leaked")
    _check(
        provider_payload["tool_results"][0]["raw_omitted"] is True,
        "runtime tool omission marker was not preserved",
    )
    _check(
        "omitted_context_categories" in provider_payload["metadata"],
        "metadata did not include omission categories",
    )


def test_redact_sensitive_payload_handles_nested_case_insensitive_keys() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_context import (
        redact_sensitive_payload,
    )

    password_value = "p" * 6
    client_secret_value = "c" * 10
    credential_blob_value = "d" * 8
    payload = {
        "Authorization": "Bearer abc",
        "safe": "visible",
        "nested": [
            {"api_Key": "sk-value", "content": "ok"},
            {"inner": {"PASSWORD": password_value, "CLIENT_SECRET": client_secret_value}},
            ("leave-me", {"CredentialBlob": credential_blob_value}),
        ],
        "auth_header": {"value": "Bearer xyz"},
    }

    redacted = redact_sensitive_payload(payload)

    _check(redacted["Authorization"] == "[REDACTED]", "authorization not redacted")
    _check(redacted["nested"][0]["api_Key"] == "[REDACTED]", "api key not redacted")
    _check(redacted["nested"][1]["inner"]["PASSWORD"] == "[REDACTED]", "password not redacted")
    _check(
        redacted["nested"][1]["inner"]["CLIENT_SECRET"] == "[REDACTED]",
        "client secret not redacted",
    )
    _check(
        redacted["nested"][2][1]["CredentialBlob"] == "[REDACTED]",
        "credential blob not redacted",
    )
    _check(redacted["auth_header"] == {"redacted": True}, "auth_header not redacted")
    _check(redacted["safe"] == "visible", "safe field changed unexpectedly")
    _check(redacted["nested"][2][0] == "leave-me", "tuple element changed unexpectedly")


def test_redact_sensitive_payload_preserves_container_types_for_sensitive_values() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_context import (
        redact_sensitive_payload,
    )

    payload = {
        "api_key": {"value": "sk-secret", "scope": "all"},
        "tokens": ["one", "two"],
        "nested": {"client_secret": ("tuple-secret",)},
    }

    redacted = redact_sensitive_payload(payload)

    _check(isinstance(redacted["api_key"], dict), "sensitive dict placeholder changed type")
    _check(redacted["api_key"]["redacted"] is True, "sensitive dict placeholder missing marker")
    _check(isinstance(redacted["tokens"], list), "sensitive list placeholder changed type")
    _check(redacted["tokens"] == ["[REDACTED]"], "sensitive list placeholder mismatch")
    _check(isinstance(redacted["nested"]["client_secret"], tuple), "sensitive tuple placeholder changed type")
    _check(redacted["nested"]["client_secret"] == ("[REDACTED]",), "sensitive tuple placeholder mismatch")


def test_runtime_context_preserves_safe_summaries_and_ids() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_context import (
        build_runtime_tree_context,
    )

    context = build_runtime_tree_context(
        persona_id="p1",
        session_id="s1",
        user_message="hello",
        memory_entries=[
            {
                "id": "m1",
                "summary": "safe memory summary",
                "content": "safe memory content",
                "private_note": "memory-private-value",
            }
        ],
        state_docs=[
            {
                "id": "state-1",
                "summary": "safe state summary",
                "content": "safe state content",
            }
        ],
        exemplar_sections=[],
        tool_results=[
            {
                "tool": "search",
                "id": "tool-result-1",
                "summary": "safe tool summary",
                "raw": "private raw output",
            }
        ],
    )

    payload = context.for_generator()
    serialized = repr(payload)

    _check(
        payload["memory_entries"][0]["summary"] == "safe memory summary",
        "memory summary was dropped",
    )
    _check(
        payload["state_docs"][0]["summary"] == "safe state summary",
        "state summary was dropped",
    )
    _check(payload["tool_results"][0]["id"] == "tool-result-1", "tool result id was dropped")
    _check(payload["tool_results"][0]["summary"] == "safe tool summary", "tool summary was dropped")
    _check(payload["tool_results"][0]["raw_omitted"] is True, "tool raw omission marker changed")
    _check("memory-private-value" not in serialized, "private memory detail leaked")
    _check("private raw output" not in serialized, "raw tool output leaked")


def test_runtime_context_redacts_sensitive_literals_inside_allowed_text() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_context import (
        build_runtime_tree_context,
    )

    bearer_text = "Bearer " + "inline-token"
    provider_marker = "sk-" + "inline"
    phrase_marker = "pass" + "word=" + "inline-pass"
    whitespace_marker = "whitespace-token"
    password_marker = "hunter-" + "inline"
    multiword_api_marker = "nonprefixed-" + "inline"
    multiword_client_marker = "client-" + "inline"
    context = build_runtime_tree_context(
        persona_id="p1",
        session_id="s1",
        user_message=(
            f"please use Authorization: {bearer_text}; "
            f"api key {multiword_api_marker}; client secret {multiword_client_marker}"
        ),
        memory_entries=[
            {
                "id": "m1",
                "summary": f"summary contains token={bearer_text}",
                "content": f"content has api_key={provider_marker}",
            }
        ],
        state_docs=[{"id": "s1", "content": f"safe state token {whitespace_marker}"}],
        exemplar_sections=[("style", f"example says {phrase_marker}", 4)],
        tool_results=[
            {
                "tool": "search",
                "summary": f"summary has {provider_marker} and password {password_marker}",
            }
        ],
    )

    payload = context.for_generator()
    serialized = repr(payload)

    _check("inline-token" not in serialized, "inline bearer value leaked")
    _check("sk-inline" not in serialized, "inline provider marker leaked")
    _check("inline-pass" not in serialized, "inline phrase marker leaked")
    _check("whitespace-token" not in serialized, "whitespace token marker leaked")
    _check("hunter-inline" not in serialized, "whitespace password marker leaked")
    _check("nonprefixed-inline" not in serialized, "multi-word api key marker leaked")
    _check("client-inline" not in serialized, "multi-word client marker leaked")
    _check("[REDACTED]" in serialized, "inline sensitive text was not redacted")
    _check(
        payload["metadata"]["redacted_field_count"] >= 4,
        "inline redactions were not tracked in metadata",
    )


def test_tool_result_private_fields_are_omitted_with_metadata() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_context import (
        build_runtime_tree_context,
    )

    bearer_text = "Bearer " + "tool-token"
    context = build_runtime_tree_context(
        persona_id="p1",
        session_id="s1",
        user_message="hello",
        tool_results=[
            {
                "tool": "web",
                "summary": "safe summary",
                "response": "private external response",
                "raw_response": "private raw response",
                "body": "private response body",
                "headers": {"Authorization": bearer_text},
            }
        ],
    )

    payload = context.for_generator()
    serialized = repr(payload)
    categories = payload["metadata"]["omitted_context_categories"]

    _check("private external response" not in serialized, "tool response leaked")
    _check("private raw response" not in serialized, "tool raw_response leaked")
    _check("private response body" not in serialized, "tool body leaked")
    _check("tool-token" not in serialized, "tool headers leaked")
    _check(
        payload["tool_results"][0]["raw_omitted"] is True,
        "private tool omission marker missing",
    )
    _check(
        "tool_results.private_response_fields" in categories,
        "private tool omission category missing",
    )


def test_non_mapping_tool_results_are_sanitized_as_omitted_placeholders() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_context import (
        build_runtime_tree_context,
    )

    context = build_runtime_tree_context(
        persona_id="p1",
        session_id="s1",
        user_message="hello",
        tool_results=[None, "raw secret token=tool-secret"],
    )

    payload = context.for_generator()
    serialized = repr(payload)

    _check(payload["tool_results"][0]["tool"] == "tool_0", "first placeholder tool name mismatch")
    _check(payload["tool_results"][0]["raw_omitted"] is True, "first placeholder missing omission marker")
    _check(payload["tool_results"][1]["tool"] == "tool_1", "second placeholder tool name mismatch")
    _check(payload["tool_results"][1]["raw_omitted"] is True, "second placeholder missing omission marker")
    _check("tool-secret" not in serialized, "non-mapping tool result leaked raw value")
    _check(
        "tool_results.invalid_entries" in payload["metadata"]["omitted_context_categories"],
        "invalid tool result omission category missing",
    )


def test_truncate_text_fields_caps_oversized_text_and_tracks_metadata() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_context import truncate_text_fields

    payload = {
        "short": "abc",
        "long": "x" * 40,
        "nested": {"content": "y" * 50},
    }

    truncated, metadata = truncate_text_fields(payload, max_length=20)

    _check(truncated["short"] == "abc", "short field should not truncate")
    _check(truncated["long"] == ("x" * 17) + "...", "long field truncate mismatch")
    _check(
        truncated["nested"]["content"] == ("y" * 17) + "...",
        "nested content truncate mismatch",
    )
    _check(metadata["truncated_field_count"] == 2, "unexpected truncate count")
    _check("long" in metadata["truncated_paths"], "missing truncate path: long")
    _check(
        "nested.content" in metadata["truncated_paths"],
        "missing truncate path: nested.content",
    )


def test_offline_context_redacts_secrets_and_keeps_safe_tool_diagnostics() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_context import (
        build_offline_tree_context,
    )

    token_value = "-".join(("raw", "token"))
    memory_password_value = "".join(("1", "2", "3"))
    context = build_offline_tree_context(
        persona_id="p2",
        session_id="s2",
        user_message="diagnose",
        policy_snapshot={"token": token_value, "mode": "safe"},
        memory_entries=[{"id": "m1", "content": "memo", "password": memory_password_value}],
        state_docs=[],
        exemplar_sections=[],
        tool_results=[
            {
                "tool": "search",
                "raw": "private bytes",
                "status": "ok",
                "latency_ms": 25,
                "error": None,
                "id": "tr-1",
            }
        ],
        max_text_length=30,
    )

    payload = context.for_generator()
    serialized = repr(payload)

    _check("raw-token" not in serialized, "offline payload leaked token")
    _check("private bytes" not in serialized, "offline payload leaked tool raw output")
    _check("123" not in serialized, "offline payload leaked password")
    _check(payload["tool_results"][0]["tool"] == "search", "tool name not retained")
    _check(
        payload["tool_results"][0]["diagnostics"]["status"] == "ok",
        "offline diagnostic status missing",
    )
    _check(payload["metadata"]["context_mode"] == "offline", "context mode incorrect")

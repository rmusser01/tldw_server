from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.core.Sync.v2.notes_moodboard_studio_contract import (
    JS_SAFE_INTEGER_MAX,
    NotesMoodboardStudioContractError,
    StudioDiagramManifestV1,
    canonical_json_bytes,
    diagram_render_hash,
    legacy_diagnostic_hash,
    legacy_source_diagnostic,
    notes_moodboard_note_object_hash,
    notes_moodboard_object_hash,
    notes_studio_document_object_hash,
    parse_notes_moodboard_note_tombstone_v1,
    parse_notes_moodboard_note_v1,
    parse_notes_moodboard_tombstone_v1,
    parse_notes_moodboard_v1,
    parse_notes_studio_document_tombstone_v1,
    parse_notes_studio_document_v1,
    placement_object_id,
    studio_result_hash,
)

MOODBOARD_ID = "253fbb6d-8bc9-4e7f-bce0-56ac1fd46227"
NOTE_ID = "28467075-bde3-4478-883c-125a5672873c"
SOURCE_NOTE_ID = "3978e2c4-33e4-43ae-a09b-a099226067de"
COLLECTION_ID = "dc20376a-69ca-411f-849c-53c59d7f645a"
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
ACCEPTED_AT = "2026-08-24T00:00:00Z"


def valid_moodboard_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "moodboard_id": MOODBOARD_ID,
        "name": "Research board",
        "description": "Portable visual notes",
        "smart_rule": {
            "query": "café",
            "keyword_tokens": ["priority"],
            "collection_sync_ids": [COLLECTION_ID],
            "sources": ["conversation"],
            "updated": {"after": None, "before": None},
        },
        "canvas": {"layout_mode": "freeform", "metadata": {"theme": "paper"}},
    }
    payload.update(overrides)
    return payload


def valid_placement_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "moodboard_id": MOODBOARD_ID,
        "note_id": NOTE_ID,
        "x": -12,
        "y": 24,
        "width": 320,
        "height": 220,
        "order_index": 7,
        "display": {"color": "amber"},
    }
    payload.update(overrides)
    return payload


def valid_studio_payload(
    *,
    with_source: bool = False,
    with_diagram: bool = False,
    **overrides: object,
) -> dict[str, object]:
    sections = [
        {"id": "cues", "kind": "cue", "title": "Questions", "items": ["Why?"]},
        {
            "id": "notes",
            "kind": "notes",
            "title": "Answer",
            "content": "Because.",
        },
    ]
    source_note_id = SOURCE_NOTE_ID if with_source else None
    source_revision = 3 if with_source else None
    source_hash = HASH_B if with_source else None
    provenance_kind = "derive" if with_source else "manual"
    provider = "openai" if with_source else None
    model = "gpt-test" if with_source else None
    excerpt = "First line\r\nSecond line" if with_source else None
    excerpt_hash = (
        "sha256:"
        + hashlib.sha256(b"First line\nSecond line").hexdigest()
        if with_source
        else None
    )
    diagram = None
    if with_diagram:
        context = "Questions\nWhy?\nAnswer\nBecause."
        diagram_text = "flowchart TD\n  A-->B"
        diagram = {
            "diagram_type": "flowchart",
            "source_section_ids": ["cues", "notes"],
            "source_graph": [
                {"id": "cues", "title": "Questions", "kind": "cue", "content": "Why?"},
                {
                    "id": "notes",
                    "title": "Answer",
                    "kind": "notes",
                    "content": "Because.",
                },
            ],
            "diagram": diagram_text,
            "format": "mermaid",
            "status": "ready",
            "render_hash": diagram_render_hash(
                diagram_type="flowchart", context=context, diagram=diagram_text
            ),
        }
    payload: dict[str, object] = {
        "note_id": NOTE_ID,
        "source_note_id": source_note_id,
        "payload_json": {"sections": sections},
        "template_type": "lined",
        "handwriting_mode": "accented",
        "excerpt_snapshot": excerpt,
        "excerpt_hash": excerpt_hash,
        "diagram_manifest_json": diagram,
        "companion_content_hash": HASH_A,
        "render_version": 1,
        "note_revision": 4,
        "note_hash": HASH_B,
        "accepted_provenance": {
            "kind": provenance_kind,
            "attestation": "server",
            "provider": provider,
            "model": model,
            "accepted_at": ACCEPTED_AT,
            "source_revision": source_revision,
            "source_hash": source_hash,
            "result_hash": HASH_A,
        },
    }
    payload.update(overrides)
    provenance = payload["accepted_provenance"]
    assert isinstance(provenance, dict)
    provenance["result_hash"] = studio_result_hash(payload)
    return payload


def parse_studio(payload: dict[str, object]):
    return parse_notes_studio_document_v1(
        payload,
        bound_attestation="server",
        bound_accepted_at=ACCEPTED_AT,
    )


def test_canonical_json_is_compact_sorted_utf8_and_does_not_escape_unicode() -> None:
    assert canonical_json_bytes({"z": "café", "a": [True, None, 1]}) == (
        b'{"a":[true,null,1],"z":"caf\xc3\xa9"}'
    )


@pytest.mark.parametrize("value", [1.0, float("nan"), float("inf")])
def test_canonical_json_rejects_integer_ambiguity_and_non_finite_numbers(value: float) -> None:
    with pytest.raises(NotesMoodboardStudioContractError, match="integer-only canonical numbers"):
        canonical_json_bytes({"x": value})


@pytest.mark.parametrize("value", [-(2**53), 2**53])
def test_canonical_json_rejects_integers_outside_the_js_safe_range(value: int) -> None:
    with pytest.raises(NotesMoodboardStudioContractError, match="JS-safe"):
        canonical_json_bytes({"x": value})


@pytest.mark.parametrize("key", ["not safe", "é", "", "a" * 65])
def test_canonical_json_rejects_noncanonical_object_keys(key: str) -> None:
    with pytest.raises(NotesMoodboardStudioContractError, match="safe ASCII"):
        canonical_json_bytes({key: 1})


def test_canonical_json_rejects_invalid_utf8_surrogates() -> None:
    with pytest.raises(NotesMoodboardStudioContractError, match="UTF-8"):
        canonical_json_bytes({"x": "\ud800"})


def test_placement_id_is_exact_namespaced_digest() -> None:
    assert placement_object_id(valid_placement_payload()) == (
        "notes.moodboard_note:sha256:"
        "01bd6602375d2452f7c0c3a35de5aeb54e76e8f3582b4c62c44df02eab9843cc"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("moodboard_id", MOODBOARD_ID.upper()),
        ("moodboard_id", "253fbb6d-8bc9-1e7f-bce0-56ac1fd46227"),
        ("note_id", "not-a-uuid"),
    ],
)
def test_placement_identity_requires_canonical_lowercase_uuid4(field: str, value: str) -> None:
    payload = valid_placement_payload(**{field: value})
    with pytest.raises(NotesMoodboardStudioContractError, match="lowercase UUIDv4"):
        placement_object_id(payload)


def test_moodboard_normalizes_smart_rule_sets_and_utc_timestamps() -> None:
    payload = valid_moodboard_payload()
    payload["smart_rule"] = {
        "query": "CAF\u00c9",
        "keyword_tokens": ["ZETA", "cafe\u0301", "CAF\u00c9"],
        "collection_sync_ids": [COLLECTION_ID, COLLECTION_ID],
        "sources": ["WEB", "web"],
        "updated": {
            "after": "2026-08-24T02:00:00+02:00",
            "before": "2026-08-24T03:30:00+02:00",
        },
    }

    parsed = parse_notes_moodboard_v1(payload)

    assert parsed.smart_rule is not None
    assert parsed.smart_rule.query == "café"
    assert parsed.smart_rule.keyword_tokens == ("café", "zeta")
    assert parsed.smart_rule.collection_sync_ids == (COLLECTION_ID,)
    assert parsed.smart_rule.sources == ("web",)
    assert parsed.smart_rule.updated.after == "2026-08-24T00:00:00Z"
    assert parsed.smart_rule.updated.before == "2026-08-24T01:30:00Z"


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-08-24 00:00:00Z",
        "2026-08-24X00:00:00Z",
        "2026-W34-1T00:00:00Z",
        "2026-08-24T00:00:00+0200",
        "2026-08-24T00:00:00.1234567Z",
        "2026-08-24T00:00:00.1234568Z",
    ],
)
def test_timestamps_reject_non_rfc3339_forms_and_excess_fractional_precision(
    timestamp: str,
) -> None:
    payload = valid_moodboard_payload()
    smart_rule = deepcopy(payload["smart_rule"])
    assert isinstance(smart_rule, dict)
    smart_rule["updated"] = {"after": timestamp, "before": None}
    payload["smart_rule"] = smart_rule

    with pytest.raises(NotesMoodboardStudioContractError, match="RFC 3339"):
        parse_notes_moodboard_v1(payload)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"name": ""}, "at least 1"),
        ({"name": "x" * 256}, "at most 255"),
        ({"description": "x" * 2_001}, "at most 2000"),
        ({"moodboard_id": MOODBOARD_ID.upper()}, "lowercase UUIDv4"),
        ({"owner_user_id": "client-choice"}, "extra inputs"),
    ],
)
def test_moodboard_rejects_outer_identity_bounds_and_server_scope_fields(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(NotesMoodboardStudioContractError, match=message):
        parse_notes_moodboard_v1(valid_moodboard_payload(**override))


def test_moodboard_smart_rule_is_optional_and_canvas_is_closed() -> None:
    parsed = parse_notes_moodboard_v1(valid_moodboard_payload(smart_rule=None))
    assert parsed.smart_rule is None

    payload = valid_moodboard_payload()
    payload["canvas"] = {"layout_mode": "masonry", "metadata": {}, "viewport": {}}
    with pytest.raises(NotesMoodboardStudioContractError, match="extra inputs"):
        parse_notes_moodboard_v1(payload)

    payload = valid_moodboard_payload(
        canvas={"layout_mode": "absolute", "metadata": {}}
    )
    with pytest.raises(NotesMoodboardStudioContractError, match="layout_mode"):
        parse_notes_moodboard_v1(payload)


@pytest.mark.parametrize(
    ("rule_update", "message"),
    [
        ({"query": "x" * 2_001}, "at most 2000"),
        ({"keyword_tokens": ["x"] * 101}, "at most 100"),
        ({"keyword_tokens": ["x" * 256]}, "at most 255"),
        ({"keyword_tokens": ["ß" * 255]}, "at most 255"),
        ({"collection_sync_ids": [COLLECTION_ID] * 101}, "at most 100"),
        ({"sources": [f"s{i}" for i in range(51)]}, "at most 50"),
        ({"sources": ["x" * 256]}, "at most 255"),
        ({"updated": {"after": "2026-08-25T00:00:00Z", "before": "2026-08-24T00:00:00Z"}}, "after.*before"),
    ],
)
def test_moodboard_smart_rule_rejects_each_bound_and_cross_field_rule(
    rule_update: dict[str, object], message: str
) -> None:
    payload = valid_moodboard_payload()
    rule = deepcopy(payload["smart_rule"])
    assert isinstance(rule, dict)
    rule.update(rule_update)
    payload["smart_rule"] = rule
    with pytest.raises(NotesMoodboardStudioContractError, match=message):
        parse_notes_moodboard_v1(payload)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({f"k{i}": i for i in range(65)}, "at most 64"),
        ({"not safe": 1}, "safe ASCII"),
        ({"a": {"b": {"c": {"d": {"e": 1}}}}}, "depth 4"),
        ({"blob": "x" * (16 * 1_024)}, "16 KiB"),
        ({"x": 1.0}, "integer-only"),
        ({"x": JS_SAFE_INTEGER_MAX + 1}, "JS-safe"),
        ({"owner_user_id": "leak"}, "reserved"),
        ({"hover": True}, "transient"),
        ({"api_key": "secret"}, "credential"),
        ({"provider.access_token": "secret"}, "credential"),
    ],
)
def test_canvas_extension_metadata_rejects_key_depth_count_byte_and_value_limits(
    metadata: dict[str, object], message: str
) -> None:
    payload = valid_moodboard_payload(canvas={"layout_mode": "masonry", "metadata": metadata})
    with pytest.raises(NotesMoodboardStudioContractError, match=message):
        parse_notes_moodboard_v1(payload)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"x": -(2**53)}, "greater than or equal"),
        ({"y": 2**53}, "less than or equal"),
        ({"order_index": True}, "valid integer"),
        ({"width": 0}, "greater than or equal"),
        ({"height": 1_000_001}, "less than or equal"),
        ({"x": "1"}, "valid integer"),
        ({"extra": 1}, "extra inputs"),
    ],
)
def test_placement_rejects_integer_bounds_strictness_and_extra_fields(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(NotesMoodboardStudioContractError, match=message):
        parse_notes_moodboard_note_v1(valid_placement_payload(**override))


@pytest.mark.parametrize(
    ("display", "message"),
    [
        ({f"k{i}": i for i in range(33)}, "at most 32"),
        ({"a": {"b": {"c": {"d": {"e": 1}}}}}, "depth 4"),
        ({"blob": "x" * (8 * 1_024)}, "8 KiB"),
        ({"drag_state": "moving"}, "transient"),
    ],
)
def test_placement_display_has_its_own_extension_limits(
    display: dict[str, object], message: str
) -> None:
    with pytest.raises(NotesMoodboardStudioContractError, match=message):
        parse_notes_moodboard_note_v1(valid_placement_payload(display=display))


@pytest.mark.parametrize("surface", ["canvas", "display"])
@pytest.mark.parametrize(
    "reserved_key",
    [
        "placement_id",
        "placement-id",
        "source_note_id",
        "source-note-id",
        "deleted_at",
        "deleted.at",
        "adapter_version",
        "adapter-version",
        "domain",
        "operation",
        "accepted_provenance",
        "accepted-provenance",
        "result_hash",
        "result.hash",
        "scope_type",
        "scope-type",
        "generated_preview",
        "generated-preview",
        "cached_svg",
        "cached-svg",
        "api_key",
        "api-key",
    ],
)
def test_extension_surfaces_reject_reserved_concepts_and_separator_variants(
    surface: str,
    reserved_key: str,
) -> None:
    extension = {reserved_key: "not portable metadata"}

    with pytest.raises(NotesMoodboardStudioContractError, match="cannot contain"):
        if surface == "canvas":
            parse_notes_moodboard_v1(
                valid_moodboard_payload(
                    canvas={"layout_mode": "masonry", "metadata": extension}
                )
            )
        else:
            parse_notes_moodboard_note_v1(
                valid_placement_payload(display=extension)
            )


@pytest.mark.parametrize("surface", ["canvas", "display"])
def test_extension_surfaces_retain_allowed_namespaced_keys(surface: str) -> None:
    extension = {"acme.theme": "paper", "org.example-card": True}

    if surface == "canvas":
        parsed = parse_notes_moodboard_v1(
            valid_moodboard_payload(
                canvas={"layout_mode": "masonry", "metadata": extension}
            )
        )
        assert parsed.canvas.metadata == extension
    else:
        parsed = parse_notes_moodboard_note_v1(
            valid_placement_payload(display=extension)
        )
        assert parsed.display == extension


@pytest.mark.parametrize("surface", ["canvas", "display"])
@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
@pytest.mark.parametrize(
    "reserved_alias",
    [
        "apiKey",
        "apikey",
        "APIKey",
        "APIKEY",
        "accessToken",
        "accesstoken",
        "AccessToken",
        "private_key",
        "privateKey",
        "privatekey",
        "PrivateKey",
        "ownerUserId",
        "owneruserid",
        "OwnerUserID",
        "deletedAt",
        "deletedat",
        "DeletedAt",
        "cachedSvg",
        "cachedsvg",
        "CachedSVG",
        "renderHash",
        "renderhash",
        "RenderHash",
    ],
)
def test_extension_surfaces_reject_semantic_aliases_recursively(
    surface: str,
    nested: bool,
    reserved_alias: str,
) -> None:
    reserved_value = {reserved_alias: "not portable metadata"}
    extension = {"nested": reserved_value} if nested else reserved_value

    with pytest.raises(NotesMoodboardStudioContractError, match="cannot contain"):
        if surface == "canvas":
            parse_notes_moodboard_v1(
                valid_moodboard_payload(
                    canvas={"layout_mode": "masonry", "metadata": extension}
                )
            )
        else:
            parse_notes_moodboard_note_v1(
                valid_placement_payload(display=extension)
            )


_REVIEWER_COMPACT_CREDENTIAL_KEYS = (
    "apisecret",
    "oauthclientsecret",
    "identitykey",
    "signingsecret",
    "encryptionsecret",
    "encryptionpassphrase",
    "privatekeypassphrase",
    "clienttoken",
    "apicredential",
)

_CREDENTIAL_GRAMMAR_CASES = (
    ("api", "secret"),
    ("oauth", "client", "secret"),
    ("identity", "key"),
    ("signing", "secret"),
    ("encryption", "secret"),
    ("encryption", "passphrase"),
    ("private", "key", "passphrase"),
    ("client", "token"),
    ("api", "credential"),
    ("client", "secret"),
    ("auth", "token"),
    ("bearer", "token"),
    ("oauth", "token"),
    ("id", "token"),
    ("session", "token"),
    ("access", "key"),
    ("secret", "key"),
    ("api", "token"),
    ("client", "key"),
    ("identity", "token"),
    ("signing", "key"),
    ("encryption", "key"),
    ("refresh", "token"),
    ("authorization", "token"),
    ("public", "key"),
    ("api", "credentials"),
    ("client", "password"),
    ("api", "authorization"),
)

_CREDENTIAL_KEY_STYLES = (
    "snake",
    "kebab",
    "dotted",
    "camel",
    "pascal",
    "acronym",
    "compact",
)

_CREDENTIAL_ACRONYM_CASE = {"api": "API", "oauth": "OAuth", "id": "ID"}

_EXACT_CREDENTIAL_KEYS = (
    "auth",
    "authorization",
    "bearer",
    "credential",
    "credentials",
    "password",
    "passphrase",
    "secret",
    "token",
)


def _credential_key_variant(parts: tuple[str, ...], style: str) -> str:
    if style == "snake":
        return "_".join(parts)
    if style == "kebab":
        return "-".join(parts)
    if style == "dotted":
        return ".".join(parts)
    if style == "camel":
        return parts[0] + "".join(part.title() for part in parts[1:])
    if style == "pascal":
        return "".join(part.title() for part in parts)
    if style == "acronym":
        return "".join(
            _CREDENTIAL_ACRONYM_CASE.get(part, part.title()) for part in parts
        )
    if style == "compact":
        return "".join(parts)
    raise AssertionError(f"unknown credential key style: {style}")


def _extension_at_position(key: str, position: str) -> dict[str, object]:
    reserved = {key: "must remain server-bound"}
    if position == "direct":
        return reserved
    if position == "nested":
        return {"nested": reserved}
    if position == "list":
        return {"items": [reserved]}
    raise AssertionError(f"unknown extension position: {position}")


def _parse_extension_surface(surface: str, extension: dict[str, object]) -> None:
    if surface == "canvas":
        parse_notes_moodboard_v1(
            valid_moodboard_payload(
                canvas={"layout_mode": "masonry", "metadata": extension}
            )
        )
        return
    parse_notes_moodboard_note_v1(valid_placement_payload(display=extension))


@pytest.mark.parametrize("surface", ["canvas", "display"])
@pytest.mark.parametrize("reserved_key", _REVIEWER_COMPACT_CREDENTIAL_KEYS)
def test_extension_credential_grammar_rejects_reviewer_compact_examples(
    surface: str,
    reserved_key: str,
) -> None:
    with pytest.raises(NotesMoodboardStudioContractError, match="credential"):
        _parse_extension_surface(surface, {reserved_key: "server-bound"})


@pytest.mark.parametrize("surface", ["canvas", "display"])
@pytest.mark.parametrize("position", ["direct", "nested", "list"])
@pytest.mark.parametrize(
    "parts",
    _CREDENTIAL_GRAMMAR_CASES,
    ids=lambda parts: "+".join(parts),
)
def test_extension_credential_grammar_normalizes_all_key_styles_recursively(
    surface: str,
    position: str,
    parts: tuple[str, ...],
) -> None:
    for style in _CREDENTIAL_KEY_STYLES:
        reserved_key = _credential_key_variant(parts, style)
        extension = _extension_at_position(reserved_key, position)
        with pytest.raises(NotesMoodboardStudioContractError, match="cannot contain"):
            _parse_extension_surface(surface, extension)


@pytest.mark.parametrize("surface", ["canvas", "display"])
@pytest.mark.parametrize("position", ["direct", "nested", "list"])
@pytest.mark.parametrize(
    "parts",
    [("access", "key", "id")],
    ids=["access+key+id"],
)
def test_extension_classifier_retains_exact_structured_credential_concepts(
    surface: str,
    position: str,
    parts: tuple[str, ...],
) -> None:
    for style in _CREDENTIAL_KEY_STYLES:
        reserved_key = _credential_key_variant(parts, style)
        extension = _extension_at_position(reserved_key, position)
        with pytest.raises(NotesMoodboardStudioContractError, match="credential"):
            _parse_extension_surface(surface, extension)


@pytest.mark.parametrize("surface", ["canvas", "display"])
@pytest.mark.parametrize("position", ["direct", "nested", "list"])
@pytest.mark.parametrize("reserved_key", _EXACT_CREDENTIAL_KEYS)
def test_extension_classifier_retains_exact_single_credential_concepts(
    surface: str,
    position: str,
    reserved_key: str,
) -> None:
    for alias in (reserved_key, reserved_key.title(), reserved_key.upper()):
        extension = _extension_at_position(alias, position)
        with pytest.raises(NotesMoodboardStudioContractError, match="credential"):
            _parse_extension_surface(surface, extension)


@pytest.mark.parametrize("surface", ["canvas", "display"])
@pytest.mark.parametrize("position", ["direct", "nested", "list"])
def test_extension_classifier_uses_exact_concepts_not_substrings(
    surface: str,
    position: str,
) -> None:
    allowed = {
        "ownership": "shared",
        "tokenizer": "unicode",
        "secretary": True,
        "accessibility": "high-contrast",
        "authentication": "delegated",
        "acme.theme": "midnight",
        "apisecretary": "assistant",
        "clienttokenizer": "portable",
        "encryptionaccessibility": "high-contrast",
        "oauthauthentication": "delegated",
    }
    if position == "direct":
        extension: dict[str, object] = allowed
    elif position == "nested":
        extension = {"nested": allowed}
    else:
        extension = {"items": [allowed]}

    if surface == "canvas":
        parsed = parse_notes_moodboard_v1(
            valid_moodboard_payload(
                canvas={"layout_mode": "masonry", "metadata": extension}
            )
        )
        assert parsed.canvas.metadata == extension
    else:
        parsed = parse_notes_moodboard_note_v1(
            valid_placement_payload(display=extension)
        )
        assert parsed.display == extension


def test_sections_only_studio_state_is_valid_and_acceptance_is_server_bound() -> None:
    payload = valid_studio_payload()
    parsed = parse_studio(payload)

    assert parsed.payload_json.model_dump(mode="json") == {
        "sections": [
            {"id": "cues", "kind": "cue", "title": "Questions", "items": ["Why?"]},
            {"id": "notes", "kind": "notes", "title": "Answer", "content": "Because."},
        ]
    }
    with pytest.raises(NotesMoodboardStudioContractError, match="server-bound attestation"):
        parse_notes_studio_document_v1(
            payload,
            bound_attestation="client_declared",
            bound_accepted_at=ACCEPTED_AT,
        )
    with pytest.raises(NotesMoodboardStudioContractError, match="server-bound acceptance time"):
        parse_notes_studio_document_v1(
            payload,
            bound_attestation="server",
            bound_accepted_at="2026-08-24T00:00:01Z",
        )


@pytest.mark.parametrize(
    ("payload_json", "message"),
    [
        ({"sections": [], "meta": {}}, "extra inputs"),
        ({"sections": [{"id": "a", "kind": "cue", "title": "A", "items": [], "content": "no"}]}, "extra inputs"),
        ({"sections": [{"id": "a", "kind": "notes", "title": "A", "content": "yes", "items": []}]}, "extra inputs"),
        ({"sections": [{"id": "a", "kind": "summary", "title": "A"}]}, "content"),
        ({"sections": [{"id": "a", "kind": "other", "title": "A", "content": "x"}]}, "union tag"),
        ({"sections": [{"id": "", "kind": "cue", "title": "A", "items": []}]}, "at least 1"),
        ({"sections": [{"id": "x" * 129, "kind": "cue", "title": "A", "items": []}]}, "at most 128"),
        ({"sections": [{"id": "a", "kind": "cue", "title": "", "items": []}]}, "at least 1"),
        ({"sections": [{"id": "a", "kind": "cue", "title": "x" * 501, "items": []}]}, "at most 500"),
        ({"sections": [{"id": "a", "kind": "cue", "title": "A", "items": ["x"] * 201}]}, "at most 200"),
        ({"sections": [{"id": "a", "kind": "cue", "title": "A", "items": ["x" * 2_001]}]}, "at most 2000"),
        ({"sections": [{"id": "a", "kind": "notes", "title": "A", "content": "é" * 32_769}]}, "65536 UTF-8 bytes"),
        ({"sections": [{"id": "same", "kind": "cue", "title": "A", "items": []}, {"id": "same", "kind": "notes", "title": "B", "content": "x"}]}, "unique"),
        ({"sections": [{"id": f"s{i}", "kind": "cue", "title": "A", "items": []} for i in range(101)]}, "at most 100"),
    ],
)
def test_studio_sections_reject_closed_shape_and_each_approved_bound(
    payload_json: dict[str, object], message: str
) -> None:
    payload = valid_studio_payload(payload_json=payload_json)
    with pytest.raises(NotesMoodboardStudioContractError, match=message):
        parse_studio(payload)


def test_diagram_manifest_validates_source_graph_and_hashes() -> None:
    payload = valid_studio_payload(with_diagram=True)
    parsed = parse_studio(payload)
    assert parsed.diagram_manifest_json is not None
    assert parsed.diagram_manifest_json.render_hash == diagram_render_hash(
        diagram_type="flowchart",
        context="Questions\nWhy?\nAnswer\nBecause.",
        diagram="flowchart TD\n  A-->B",
    )
    assert parsed.accepted_provenance.result_hash == studio_result_hash(parsed)


def test_diagram_source_ids_normalize_to_document_order_before_hash_validation() -> None:
    payload = valid_studio_payload(with_diagram=True)
    manifest = deepcopy(payload["diagram_manifest_json"])
    assert isinstance(manifest, dict)
    manifest["source_section_ids"] = ["notes", "cues"]
    payload["diagram_manifest_json"] = manifest

    parsed = parse_studio(payload)

    assert parsed.diagram_manifest_json is not None
    assert parsed.diagram_manifest_json.source_section_ids == ("cues", "notes")
    assert parsed.diagram_manifest_json.render_hash == diagram_render_hash(
        diagram_type="flowchart",
        context="Questions\nWhy?\nAnswer\nBecause.",
        diagram="flowchart TD\n  A-->B",
    )
    assert parsed.accepted_provenance.result_hash == studio_result_hash(parsed)


def test_diagram_model_input_normalizes_identically_to_mapping_input() -> None:
    mapping_payload = valid_studio_payload(with_diagram=True)
    mapping_manifest = deepcopy(mapping_payload["diagram_manifest_json"])
    assert isinstance(mapping_manifest, dict)
    mapping_manifest["source_section_ids"] = ["notes", "cues"]
    mapping_payload["diagram_manifest_json"] = mapping_manifest
    mapping_parsed = parse_studio(mapping_payload)

    model_payload = valid_studio_payload(with_diagram=True)
    model_manifest = deepcopy(model_payload["diagram_manifest_json"])
    assert isinstance(model_manifest, dict)
    model_manifest["source_section_ids"] = ["notes", "cues"]
    model_payload["diagram_manifest_json"] = StudioDiagramManifestV1.model_validate(
        model_manifest
    )

    model_parsed = parse_studio(model_payload)

    assert model_parsed.diagram_manifest_json == mapping_parsed.diagram_manifest_json
    assert model_parsed.accepted_provenance.result_hash == studio_result_hash(
        model_parsed
    )


@pytest.mark.parametrize(
    ("manifest_update", "message"),
    [
        ({"cached_svg": "<svg/>"}, "extra inputs"),
        ({"diagram_type": "mindmap"}, "diagram_type"),
        ({"source_section_ids": ["cues", "cues"]}, "unique"),
        ({"source_section_ids": ["unknown"]}, "existing section"),
        ({"source_section_ids": [f"s{i}" for i in range(51)]}, "at most 50"),
        ({"source_graph": []}, "source_graph"),
        ({"diagram": "x" * 131_073}, "131072 UTF-8 bytes"),
        ({"format": "svg"}, "format"),
        ({"status": "pending"}, "status"),
        ({"render_hash": HASH_A}, "render_hash"),
    ],
)
def test_diagram_manifest_rejects_caches_bounds_and_cross_field_mismatches(
    manifest_update: dict[str, object], message: str
) -> None:
    payload = valid_studio_payload(with_diagram=True)
    manifest = deepcopy(payload["diagram_manifest_json"])
    assert isinstance(manifest, dict)
    manifest.update(manifest_update)
    payload["diagram_manifest_json"] = manifest
    provenance = payload["accepted_provenance"]
    assert isinstance(provenance, dict)
    provenance["result_hash"] = studio_result_hash(payload)
    with pytest.raises(NotesMoodboardStudioContractError, match=message):
        parse_studio(payload)


@pytest.mark.parametrize(
    ("provenance_update", "message"),
    [
        ({"provider": "openai", "model": None}, "provider and model"),
        ({"kind": "derive", "provider": None, "model": None}, "provider and model"),
        ({"kind": "manual", "provider": "openai", "model": "gpt"}, "provider and model"),
        ({"source_revision": 1, "source_hash": None}, "source revision and hash"),
        ({"source_revision": 0, "source_hash": HASH_A}, "greater than or equal"),
        ({"kind": "legacy_bootstrap", "attestation": "server"}, "trusted_bootstrap_v1"),
        ({"kind": "manual", "attestation": "trusted_bootstrap_v1"}, "only legacy_bootstrap"),
        ({"provider": "x" * 101, "model": "gpt"}, "at most 100"),
        ({"provider": "openai", "model": "x" * 201}, "at most 200"),
        ({"result_hash": HASH_B}, "result_hash"),
        ({"prompt": "secret"}, "extra inputs"),
    ],
)
def test_studio_provenance_rejects_pairing_attestation_bounds_hashes_and_secrets(
    provenance_update: dict[str, object], message: str
) -> None:
    payload = valid_studio_payload()
    provenance = deepcopy(payload["accepted_provenance"])
    assert isinstance(provenance, dict)
    provenance.update(provenance_update)
    payload["accepted_provenance"] = provenance
    with pytest.raises(NotesMoodboardStudioContractError, match=message):
        parse_studio(payload)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"source_note_id": NOTE_ID}, "cannot equal"),
        ({"source_note_id": SOURCE_NOTE_ID}, "source revision and hash"),
        ({"excerpt_snapshot": "excerpt", "excerpt_hash": None}, "excerpt snapshot and hash"),
        ({"excerpt_snapshot": None, "excerpt_hash": HASH_A}, "excerpt snapshot and hash"),
        ({"excerpt_snapshot": "excerpt", "excerpt_hash": HASH_A}, "source_note_id"),
        ({"render_version": 2}, "render_version"),
        ({"template_type": "blank"}, "template_type"),
        ({"handwriting_mode": "full"}, "handwriting_mode"),
        ({"excerpt_snapshot": "x" * 5_000_001}, "at most 5000000"),
        ({"note_revision": 0}, "greater than or equal"),
        ({"note_revision": "4"}, "valid integer"),
        ({"note_hash": "SHA256:" + "a" * 64}, "lowercase SHA-256"),
        ({"companion_content_hash": "md5:no"}, "lowercase SHA-256"),
        ({"owner_user_id": "client-choice"}, "extra inputs"),
    ],
)
def test_studio_outer_contract_rejects_identity_binding_hash_and_strictness_errors(
    update: dict[str, object], message: str
) -> None:
    payload = valid_studio_payload(**update)
    with pytest.raises(NotesMoodboardStudioContractError, match=message):
        parse_studio(payload)


def test_excerpt_hash_uses_normalized_line_endings_and_must_match() -> None:
    payload = valid_studio_payload(with_source=True)
    parsed = parse_studio(payload)
    assert parsed.excerpt_hash == "sha256:" + hashlib.sha256(
        b"First line\nSecond line"
    ).hexdigest()

    payload["excerpt_hash"] = HASH_A
    provenance = payload["accepted_provenance"]
    assert isinstance(provenance, dict)
    provenance["result_hash"] = studio_result_hash(payload)
    with pytest.raises(NotesMoodboardStudioContractError, match="excerpt_hash"):
        parse_studio(payload)


def test_studio_acceptance_timestamp_normalizes_to_canonical_utc_z() -> None:
    payload = valid_studio_payload()
    provenance = payload["accepted_provenance"]
    assert isinstance(provenance, dict)
    provenance["accepted_at"] = "2026-08-24T02:00:00+02:00"

    parsed = parse_studio(payload)

    assert parsed.accepted_provenance.accepted_at == ACCEPTED_AT


def test_studio_source_binding_requires_exact_provenance_pair() -> None:
    payload = valid_studio_payload(with_source=True)
    provenance = payload["accepted_provenance"]
    assert isinstance(provenance, dict)
    provenance["source_hash"] = None
    provenance["source_revision"] = None
    provenance["result_hash"] = studio_result_hash(payload)
    with pytest.raises(NotesMoodboardStudioContractError, match="source revision and hash"):
        parse_studio(payload)


def test_complete_tombstones_parse_the_same_payload_and_hash_a_distinct_lifecycle() -> None:
    moodboard = parse_notes_moodboard_v1(valid_moodboard_payload())
    placement = parse_notes_moodboard_note_v1(valid_placement_payload())
    studio = parse_studio(valid_studio_payload(with_source=True, with_diagram=True))

    assert parse_notes_moodboard_tombstone_v1(moodboard) == moodboard
    assert parse_notes_moodboard_note_tombstone_v1(placement) == placement
    assert parse_notes_studio_document_tombstone_v1(studio) == studio
    assert notes_moodboard_object_hash(moodboard, revision=2, deleted=True) != (
        notes_moodboard_object_hash(moodboard, revision=2, deleted=False)
    )
    assert notes_moodboard_note_object_hash(placement, revision=2, deleted=True) != (
        notes_moodboard_note_object_hash(placement, revision=2, deleted=False)
    )
    assert notes_studio_document_object_hash(studio, revision=2, deleted=True) != (
        notes_studio_document_object_hash(studio, revision=2, deleted=False)
    )

    partial = valid_placement_payload()
    partial.pop("display")
    with pytest.raises(NotesMoodboardStudioContractError, match="display"):
        parse_notes_moodboard_note_tombstone_v1(partial)


def test_object_hash_uses_the_exact_required_frame() -> None:
    parsed = parse_notes_moodboard_v1(valid_moodboard_payload(smart_rule=None))
    exact_frame = (
        b'{"adapter_version":1,"domain":"notes.moodboard",'
        b'"identity":{"moodboard_id":"253fbb6d-8bc9-4e7f-bce0-56ac1fd46227"},'
        b'"lifecycle":"live","payload":{"canvas":{"layout_mode":"freeform",'
        b'"metadata":{"theme":"paper"}},"description":"Portable visual notes",'
        b'"moodboard_id":"253fbb6d-8bc9-4e7f-bce0-56ac1fd46227",'
        b'"name":"Research board","smart_rule":null},"revision":7}'
    )
    assert notes_moodboard_object_hash(parsed, revision=7, deleted=False) == (
        "sha256:" + hashlib.sha256(exact_frame).hexdigest()
    )
    with pytest.raises(NotesMoodboardStudioContractError, match="positive integer"):
        notes_moodboard_object_hash(parsed, revision=True, deleted=False)


def test_models_and_recursive_extensions_are_frozen_and_hash_stable() -> None:
    parsed = parse_notes_moodboard_v1(valid_moodboard_payload())
    before = notes_moodboard_object_hash(parsed, revision=1, deleted=False)

    with pytest.raises(ValidationError, match="frozen"):
        parsed.name = "changed"
    with pytest.raises(TypeError, match="immutable"):
        parsed.canvas.metadata["theme"] = "changed"
    assert notes_moodboard_object_hash(parsed, revision=1, deleted=False) == before


def test_legacy_diagnostics_are_bounded_and_do_not_expose_source_content() -> None:
    source = {"title": "sensitive note title", "token": "secret-value"}
    digest = legacy_diagnostic_hash(source)
    diagnostic = legacy_source_diagnostic("legacy_studio_shape_invalid", source)

    assert digest.startswith("sha256:") and len(digest) == 71
    assert diagnostic == {
        "code": "legacy_studio_shape_invalid",
        "source_hash": digest,
    }
    assert "sensitive" not in repr(diagnostic)
    assert "secret-value" not in repr(diagnostic)
    with pytest.raises(NotesMoodboardStudioContractError, match="at most 64"):
        legacy_source_diagnostic("x" * 65, source)
    with pytest.raises(NotesMoodboardStudioContractError, match="262144 bytes"):
        legacy_diagnostic_hash(b"x" * 262_145)

from __future__ import annotations

from collections.abc import Generator, Mapping
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.VN_Scripts.authoring_catalog import list_authoring_catalog
from tldw_Server_API.app.core.VN_Scripts.authoring_errors import VNScriptAuthoringError
from tldw_Server_API.app.core.VN_Scripts.service import VNScriptService
from tldw_Server_API.app.core.VN_Scripts.snippet_patcher import (
    MAX_SNIPPET_PARAMETER_DEPTH,
    MAX_SNIPPET_PARAMETER_PAYLOAD_BYTES,
    MAX_SNIPPET_PARAMETER_STRING_LENGTH,
    SnippetPatchResult,
    apply_snippet_patch,
)
from tldw_Server_API.app.core.VN_Scripts.validator import (
    forbidden_generation_routing_keys,
    known_script_ops,
    supported_generation_output_schemas,
)


def _operation_ids(catalog: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(operation["op"] for operation in catalog["operations"])


def _assert_object_schemas_are_closed(schema: Mapping[str, Any], path: str = "$") -> None:
    if schema.get("type") == "object":
        assert schema.get("additionalProperties") is False, path
    for key in ("properties", "patternProperties", "$defs", "definitions"):
        children = schema.get(key)
        if isinstance(children, Mapping):
            for child_name, child_schema in children.items():
                if isinstance(child_schema, Mapping):
                    _assert_object_schemas_are_closed(child_schema, f"{path}.{key}.{child_name}")
    items = schema.get("items")
    if isinstance(items, Mapping):
        _assert_object_schemas_are_closed(items, f"{path}.items")
    for key in ("oneOf", "anyOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for index, variant in enumerate(variants):
                if isinstance(variant, Mapping):
                    _assert_object_schemas_are_closed(variant, f"{path}.{key}[{index}]")


def _forbidden_keys_present(value: Any) -> set[str]:
    forbidden = set(forbidden_generation_routing_keys()) | {
        "raw_prompt",
        "prompt",
        "provider_config",
        "validation_codes",
    }
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in forbidden:
                found.add(str(key))
            found.update(_forbidden_keys_present(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_keys_present(child))
    return found


def _draft() -> dict[str, Any]:
    return {
        "schema_version": "vn_script_program.v1",
        "primary_asset_pack_id": 7,
        "entry_label": "start",
        "labels": {
            "start": [
                {"op": "narrate", "text": "Opening line."},
                {"op": "end"},
            ]
        },
    }


def _audio_draft() -> dict[str, Any]:
    return {
        "schema_version": "vn_script_program.v1",
        "primary_asset_pack_id": 7,
        "entry_label": "start",
        "labels": {
            "start": [
                {"op": "narrate", "text": "Opening line."},
                {"op": "end"},
            ]
        },
    }


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": "vn_asset_manifest.v1",
        "pack_id": 7,
        "title": "Starter Pack",
        "primary_character_id": None,
        "content_rating": "general",
        "assets": {"backgrounds": [], "sprites": [], "depth_companions": [], "cgs": []},
    }


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-script-authoring-catalog-test-client")
    yield database
    database.close_connection()


def _service(
    chacha_db: CharactersRAGDB,
    *,
    owner_user_id: int = 42,
    audio_ref_resolver: Any | None = None,
) -> VNScriptService:
    return VNScriptService(
        chacha_db,
        owner_user_id=owner_user_id,
        manifest_resolver=lambda asset_pack_id: _manifest(),
        audio_ref_resolver=audio_ref_resolver,
    )


def _create_script(service: VNScriptService, draft: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return service.create_script(
        title="Archive Door",
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
        content_rating="general",
        initial_draft=draft or _draft(),
        initial_diagnostics={"valid": True, "errors": [], "warnings": [], "stored": True},
    )


def _assert_authoring_error(
    exc_info: Any,
    *,
    code: str,
    detail_key: str,
    detail_value: Any,
) -> None:
    error = exc_info.value
    assert isinstance(error, VNScriptAuthoringError)
    assert error.code == code
    assert error.status_code == 400
    assert error.details[detail_key] == detail_value


def test_service_get_authoring_catalog_returns_catalog_metadata(chacha_db: CharactersRAGDB) -> None:
    service = _service(chacha_db)

    assert service.get_authoring_catalog() == list_authoring_catalog()


def test_service_build_snippet_patch_requires_ownership_and_uses_stored_draft_without_persisting(
    chacha_db: CharactersRAGDB,
) -> None:
    owner_service = _service(chacha_db, owner_user_id=42)
    other_service = _service(chacha_db, owner_user_id=7)
    script = _create_script(owner_service)

    with pytest.raises(ValueError, match="script_not_found"):
        other_service.build_snippet_patch(
            script["id"],
            "narration",
            {"label": "start", "op_index": 1, "mode": "before"},
            {"text": "Inserted line."},
        )

    result = owner_service.build_snippet_patch(
        script["id"],
        "narration",
        {"label": "start", "op_index": 1, "mode": "before"},
        {"text": "Inserted line."},
    )
    stored = owner_service.get_draft(script["id"])

    assert result["script"]["id"] == script["id"]
    assert result["base_revision"] == 1
    assert result["snippet_id"] == "narration"
    assert isinstance(result["patch"], SnippetPatchResult)
    assert result["patch"].draft["labels"]["start"][1] == {"op": "narrate", "text": "Inserted line."}
    assert stored["revision"] == 1
    assert stored["draft"] == _draft()
    assert stored["diagnostics"]["stored"] is True


def test_service_build_snippet_patch_with_supplied_draft_uses_current_base_revision_without_persisting(
    chacha_db: CharactersRAGDB,
) -> None:
    service = _service(chacha_db)
    script = _create_script(service)
    supplied_draft = _draft()
    supplied_draft["labels"]["start"][0]["text"] = "Supplied opening."

    result = service.build_snippet_patch(
        script["id"],
        "narration",
        {"label": "start", "op_index": 1, "mode": "before"},
        {"text": "Inserted line."},
        draft=supplied_draft,
        draft_revision=1,
    )
    stored = service.get_draft(script["id"])

    assert result["base_revision"] == 1
    assert result["patch"].draft["labels"]["start"][0] == {"op": "narrate", "text": "Supplied opening."}
    assert stored["revision"] == 1
    assert stored["draft"] == _draft()


def test_service_supplied_draft_requires_draft_revision(chacha_db: CharactersRAGDB) -> None:
    service = _service(chacha_db)
    script = _create_script(service)

    with pytest.raises(ValueError, match="draft_revision_required"):
        service.build_snippet_patch(
            script["id"],
            "narration",
            {"label": "start", "op_index": 1, "mode": "before"},
            {"text": "Inserted line."},
            draft=_draft(),
        )


def test_service_supplied_draft_requires_matching_revision_and_cannot_overwrite_newer_draft(
    chacha_db: CharactersRAGDB,
) -> None:
    service = _service(chacha_db)
    script = _create_script(service)
    stale_draft = service.get_draft(script["id"])["draft"]
    stale_build = service.build_snippet_patch(
        script["id"],
        "narration",
        {"label": "start", "op_index": 1, "mode": "before"},
        {"text": "Stale line."},
        draft=stale_draft,
        draft_revision=1,
    )
    service.replace_draft(
        script["id"],
        if_revision=1,
        draft={
            **_draft(),
            "labels": {"start": [{"op": "narrate", "text": "Current line."}, {"op": "end"}]},
        },
    )

    with pytest.raises(ValueError, match="draft_revision_conflict"):
        service.build_snippet_patch(
            script["id"],
            "narration",
            {"label": "start", "op_index": 1, "mode": "before"},
            {"text": "Stale line."},
            draft=stale_draft,
            draft_revision=1,
        )
    with pytest.raises(ValueError, match="draft_revision_conflict"):
        service.apply_snippet_patch_result(script["id"], "narration", stale_build["base_revision"], stale_build["patch"])

    stored = service.get_draft(script["id"])
    assert stored["revision"] == 2
    assert stored["draft"]["labels"]["start"][0] == {"op": "narrate", "text": "Current line."}


def test_service_preview_snippet_patch_validates_payload_without_mutating_stored_draft(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(chacha_db)
    script = _create_script(service)
    build = service.build_snippet_patch(
        script["id"],
        "narration",
        {"label": "start", "op_index": 1, "mode": "before"},
        {"text": "Inserted line."},
    )

    def fail_validate_draft(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("preview must not call validate_draft")

    monkeypatch.setattr(service, "validate_draft", fail_validate_draft)

    preview = service.preview_snippet_patch(
        script["id"],
        "narration",
        build["base_revision"],
        build["patch"],
    )
    stored = service.get_draft(script["id"])

    assert preview["script_id"] == script["id"]
    assert preview["base_revision"] == 1
    assert preview["snippet_id"] == "narration"
    assert preview["draft"]["labels"]["start"][1] == {"op": "narrate", "text": "Inserted line."}
    assert preview["diagnostics"]["valid"] is True
    assert preview["patch_summary"] == build["patch"].patch_summary
    assert preview["warnings"] == preview["diagnostics"]["warnings"]
    assert stored["revision"] == 1
    assert stored["draft"] == _draft()
    assert stored["diagnostics"]["stored"] is True


def test_service_apply_snippet_patch_result_validates_and_persists_with_revision(
    chacha_db: CharactersRAGDB,
) -> None:
    service = _service(chacha_db)
    script = _create_script(service)
    build = service.build_snippet_patch(
        script["id"],
        "narration",
        {"label": "start", "op_index": 1, "mode": "before"},
        {"text": "Inserted line."},
    )

    applied = service.apply_snippet_patch_result(
        script["id"],
        "narration",
        build["base_revision"],
        build["patch"],
    )
    stored = service.get_draft(script["id"])

    assert applied["script_id"] == script["id"]
    assert applied["revision"] == 2
    assert applied["snippet_id"] == "narration"
    assert applied["draft"]["labels"]["start"][1] == {"op": "narrate", "text": "Inserted line."}
    assert applied["diagnostics"]["valid"] is True
    assert applied["patch_summary"] == build["patch"].patch_summary
    assert stored["revision"] == 2
    assert stored["draft"] == applied["draft"]
    assert stored["diagnostics"] == applied["diagnostics"]


def test_service_stale_apply_raises_draft_revision_conflict(chacha_db: CharactersRAGDB) -> None:
    service = _service(chacha_db)
    script = _create_script(service)
    build = service.build_snippet_patch(
        script["id"],
        "narration",
        {"label": "start", "op_index": 1, "mode": "before"},
        {"text": "Inserted line."},
    )

    with pytest.raises(ValueError, match="draft_revision_conflict"):
        service.apply_snippet_patch_result(script["id"], "narration", 0, build["patch"])


def test_service_stale_apply_conflicts_before_validation_or_audio_resolution(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(chacha_db)
    script = _create_script(service, _audio_draft())
    build = service.build_snippet_patch(
        script["id"],
        "play_bgm",
        {"label": "start", "op_index": 1, "mode": "before"},
        {"media_ref": "missing.audio"},
    )

    def fail_validate_draft_payload(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("stale apply must not validate patched draft")

    monkeypatch.setattr(service, "validate_draft_payload", fail_validate_draft_payload)

    with pytest.raises(ValueError, match="draft_revision_conflict"):
        service.apply_snippet_patch_result(script["id"], "play_bgm", 0, build["patch"])


def test_service_duplicate_apply_against_same_revision_conflicts(chacha_db: CharactersRAGDB) -> None:
    service = _service(chacha_db)
    script = _create_script(service)
    build = service.build_snippet_patch(
        script["id"],
        "narration",
        {"label": "start", "op_index": 1, "mode": "before"},
        {"text": "Inserted line."},
    )

    service.apply_snippet_patch_result(script["id"], "narration", build["base_revision"], build["patch"])
    with pytest.raises(ValueError, match="draft_revision_conflict"):
        service.apply_snippet_patch_result(script["id"], "narration", build["base_revision"], build["patch"])


def test_service_audio_refs_are_resolved_from_patched_draft_for_validation(chacha_db: CharactersRAGDB) -> None:
    seen_drafts: list[Mapping[str, Any]] = []

    def audio_ref_resolver(program: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
        seen_drafts.append(program)
        has_audio_op = any(op.get("op") == "play_bgm" for op in program["labels"]["start"])
        return {"bgm.archive": {"mime_type": "audio/mpeg", "generated_file_id": 7001}} if has_audio_op else {}

    service = _service(chacha_db, audio_ref_resolver=audio_ref_resolver)
    script = _create_script(service, _audio_draft())
    build = service.build_snippet_patch(
        script["id"],
        "play_bgm",
        {"label": "start", "op_index": 1, "mode": "before"},
        {"media_ref": "bgm.archive"},
    )

    preview = service.preview_snippet_patch(script["id"], "play_bgm", build["base_revision"], build["patch"])
    applied = service.apply_snippet_patch_result(
        script["id"],
        "play_bgm",
        build["base_revision"],
        build["patch"],
        audio_refs={"bgm.explicit": {"mime_type": "audio/mpeg", "generated_file_id": 7002}},
    )

    assert preview["diagnostics"]["valid"] is True
    assert applied["diagnostics"]["valid"] is False
    assert applied["diagnostics"]["errors"][0]["code"] == "audio_media_ref_inaccessible"
    assert all(any(op.get("op") == "play_bgm" for op in draft["labels"]["start"]) for draft in seen_drafts)


def test_generated_choice_set_patch_inserts_generate_and_handler_without_mutating_draft() -> None:
    draft = _draft()
    original = {
        "schema_version": "vn_script_program.v1",
        "primary_asset_pack_id": 7,
        "entry_label": "start",
        "labels": {"start": [{"op": "narrate", "text": "Opening line."}, {"op": "end"}]},
    }

    result = apply_snippet_patch(
        draft,
        "generated_choice_set",
        {"label": "start", "op_index": 0, "mode": "after"},
        {"handler_label": "generated_choice", "max_choices": 3, "scope": "turn"},
    )

    assert draft == original
    assert result.draft is not draft
    assert result.draft["labels"]["start"] == [
        {"op": "narrate", "text": "Opening line."},
        {
            "op": "generate",
            "scope": "turn",
            "max_choices": 3,
            "output_schema": "choice_set",
            "on_generated_choice": "generated_choice",
        },
        {"op": "end"},
    ]
    assert result.draft["labels"]["generated_choice"] == [
        {"op": "narrate", "text": "Handle the selected generated choice here."},
        {"op": "end"},
    ]
    assert result.patch_summary == {
        "inserted_ops": 1,
        "created_labels": ["generated_choice"],
        "changed_paths": ["$.labels.start[1]", "$.labels.generated_choice"],
    }


def test_patch_rejects_nested_raw_generation_routing_keys() -> None:
    import pytest

    with pytest.raises(VNScriptAuthoringError) as exc_info:
        apply_snippet_patch(
            _draft(),
            "scene_update_generation",
            {"label": "start", "mode": "append"},
            {"safe": {"nested": {"model": "gpt-example"}}},
        )

    _assert_authoring_error(
        exc_info,
        code="snippet_parameter_invalid",
        detail_key="field_path",
        detail_value="$.parameters.safe.nested.model",
    )


def test_patch_rejects_root_and_nested_unknown_snippet_parameters() -> None:
    import pytest

    with pytest.raises(VNScriptAuthoringError) as root_exc:
        apply_snippet_patch(
            _draft(),
            "narration",
            {"label": "start", "mode": "append"},
            {"text": "Line.", "unexpected": True},
        )
    _assert_authoring_error(
        root_exc,
        code="snippet_parameter_invalid",
        detail_key="field_path",
        detail_value="$.parameters.unexpected",
    )

    with pytest.raises(VNScriptAuthoringError) as nested_exc:
        apply_snippet_patch(
            _draft(),
            "authored_choice",
            {"label": "start", "mode": "append"},
            {
                "choice_id": "door",
                "choices": [
                    {
                        "id": "open",
                        "text": "Open it.",
                        "target_label": "start",
                        "unexpected": True,
                    }
                ],
            },
        )
    _assert_authoring_error(
        nested_exc,
        code="snippet_parameter_invalid",
        detail_key="field_path",
        detail_value="$.parameters.choices[0].unexpected",
    )


def test_authored_choice_patch_accepts_public_shape_and_maps_to_internal_opcode() -> None:
    result = apply_snippet_patch(
        _draft(),
        "authored_choice",
        {"label": "start", "mode": "append"},
        {
            "choice_id": "door",
            "choices": [
                {"id": "open", "text": "Open it.", "target_label": "start"},
                {"id": "wait", "text": "Wait.", "target_label": "start"},
            ],
        },
    )

    assert result.draft["labels"]["start"][-1] == {
        "op": "choice",
        "id": "door",
        "choices": [
            {"id": "open", "text": "Open it.", "target": "start"},
            {"id": "wait", "text": "Wait.", "target": "start"},
        ],
    }


def test_patch_rejects_invalid_and_missing_anchors_with_typed_errors() -> None:
    import pytest

    with pytest.raises(VNScriptAuthoringError) as invalid_exc:
        apply_snippet_patch(_draft(), "narration", {"label": "start", "mode": "beside"}, {"text": "Line."})
    _assert_authoring_error(
        invalid_exc,
        code="snippet_anchor_invalid",
        detail_key="anchor",
        detail_value={"label": "start", "mode": "beside"},
    )

    with pytest.raises(VNScriptAuthoringError) as missing_exc:
        apply_snippet_patch(_draft(), "narration", {"label": "missing", "mode": "append"}, {"text": "Line."})
    _assert_authoring_error(
        missing_exc,
        code="snippet_anchor_not_found",
        detail_key="anchor",
        detail_value={"label": "missing", "mode": "append"},
    )


def test_patch_rejects_bool_op_index_as_invalid_anchor() -> None:
    import pytest

    for mode in ("before", "after"):
        anchor = {"label": "start", "op_index": True, "mode": mode}
        with pytest.raises(VNScriptAuthoringError) as exc_info:
            apply_snippet_patch(_draft(), "narration", anchor, {"text": "Line."})
        _assert_authoring_error(
            exc_info,
            code="snippet_anchor_invalid",
            detail_key="anchor",
            detail_value=anchor,
        )


def test_patch_rejects_label_conflicts() -> None:
    import pytest

    with pytest.raises(VNScriptAuthoringError) as exc_info:
        apply_snippet_patch(
            _draft(),
            "generated_choice_set",
            {"label": "start", "mode": "append"},
            {"handler_label": "start"},
        )

    _assert_authoring_error(
        exc_info,
        code="snippet_label_conflict",
        detail_key="label",
        detail_value="start",
    )


def test_patch_rejects_extremely_deep_parameters_with_typed_error_before_serialization() -> None:
    import pytest

    parameters: dict[str, Any] = {"text": "Line."}
    cursor = parameters
    for depth in range(1200):
        cursor["nested"] = {}
        cursor = cursor["nested"]

    with pytest.raises(VNScriptAuthoringError) as exc_info:
        apply_snippet_patch(_draft(), "narration", {"label": "start", "mode": "append"}, parameters)

    error = exc_info.value
    assert error.code == "snippet_parameter_invalid"
    assert error.status_code == 400
    assert "field_path" in error.details
    assert error.details["field_path"].startswith("$.parameters.nested")


def test_patch_rejects_excessive_parameter_limits_with_field_paths() -> None:
    import pytest

    too_deep: dict[str, Any] = {}
    cursor = too_deep
    for depth in range(MAX_SNIPPET_PARAMETER_DEPTH + 1):
        cursor[f"level_{depth}"] = {}
        cursor = cursor[f"level_{depth}"]
    with pytest.raises(VNScriptAuthoringError) as depth_exc:
        apply_snippet_patch(_draft(), "narration", {"label": "start", "mode": "append"}, too_deep)
    _assert_authoring_error(
        depth_exc,
        code="snippet_parameter_invalid",
        detail_key="field_path",
        detail_value="$.parameters.level_0.level_1.level_2.level_3.level_4.level_5.level_6.level_7.level_8",
    )

    with pytest.raises(VNScriptAuthoringError) as string_exc:
        apply_snippet_patch(
            _draft(),
            "narration",
            {"label": "start", "mode": "append"},
            {"text": "x" * (MAX_SNIPPET_PARAMETER_STRING_LENGTH + 1)},
        )
    _assert_authoring_error(
        string_exc,
        code="snippet_parameter_invalid",
        detail_key="field_path",
        detail_value="$.parameters.text",
    )

    with pytest.raises(VNScriptAuthoringError) as payload_exc:
        apply_snippet_patch(
            _draft(),
            "narration",
            {"label": "start", "mode": "append"},
            {"items": ["x" * 1000 for _ in range((MAX_SNIPPET_PARAMETER_PAYLOAD_BYTES // 1000) + 2)]},
        )
    _assert_authoring_error(
        payload_exc,
        code="snippet_parameter_invalid",
        detail_key="field_path",
        detail_value="$.parameters",
    )


def test_patch_rejects_invalid_generation_scopes_with_field_path() -> None:
    import pytest

    cases = [
        ("generated_choice_set", {"handler_label": "generated_choice", "scope": []}),
        ("generated_choice_set", {"handler_label": "generated_choice", "scope": "session"}),
        ("scene_update_generation", {"scope": []}),
        ("scene_update_generation", {"scope": "session"}),
    ]

    for snippet_id, parameters in cases:
        with pytest.raises(VNScriptAuthoringError) as exc_info:
            apply_snippet_patch(_draft(), snippet_id, {"label": "start", "mode": "append"}, parameters)
        _assert_authoring_error(
            exc_info,
            code="snippet_parameter_invalid",
            detail_key="field_path",
            detail_value="$.parameters.scope",
        )


def test_catalog_operations_exactly_match_validator_known_ops() -> None:
    catalog = list_authoring_catalog()

    assert catalog["schema_version"] == "vn_script_authoring_catalog.v1"
    assert catalog["program_schema_version"] == "vn_script_program.v1"
    assert _operation_ids(catalog) == known_script_ops()


def test_catalog_is_preview_safe_and_includes_canonical_capability_tokens() -> None:
    catalog = list_authoring_catalog()

    assert catalog["capability_tokens"] == [
        "script_authoring_catalog",
        "scripted_generation",
        "scripted_generation.output_schema.choice_set",
        "scripted_generation.output_schema.scene_update",
        "scripted_generation.user_confirmation",
    ]
    assert catalog["generation_output_schemas"] == list(supported_generation_output_schemas())
    assert set(catalog["operation_categories"]) == {"story", "branching", "visuals", "audio", "generation", "state"}
    assert _forbidden_keys_present(catalog) == set()


def test_catalog_includes_required_v1_snippets_with_parameters_schema() -> None:
    catalog = list_authoring_catalog()

    snippet_ids = {snippet["id"] for snippet in catalog["snippets"]}

    assert {
        "narration",
        "dialogue",
        "authored_choice",
        "generated_choice_set",
        "scene_update_generation",
        "confirm_gated_generation",
        "set_background",
        "show_sprite",
        "play_bgm",
        "set_variable",
        "ending",
    }.issubset(snippet_ids)
    for snippet in catalog["snippets"]:
        assert "parameters_schema" in snippet
        assert "parameter_schema" not in snippet


def test_generated_choice_snippet_exposes_handler_label_not_opcode_target() -> None:
    catalog = list_authoring_catalog()
    snippet = next(snippet for snippet in catalog["snippets"] if snippet["id"] == "generated_choice_set")

    schema = snippet["parameters_schema"]
    default_parameters = snippet["default_parameters"]

    assert schema["required"] == ["handler_label"]
    assert "handler_label" in schema["properties"]
    assert "on_generated_choice" not in schema["properties"]
    assert "handler_label" in default_parameters
    assert "on_generated_choice" not in default_parameters
    assert "handler_label" in str(snippet["preview"])
    assert "on_generated_choice" not in str(snippet["preview"])


def test_authored_choice_snippet_exposes_public_choice_fields_not_opcode_fields() -> None:
    catalog = list_authoring_catalog()
    snippet = next(snippet for snippet in catalog["snippets"] if snippet["id"] == "authored_choice")

    schema = snippet["parameters_schema"]
    choice_schema = schema["properties"]["choices"]["items"]

    assert schema["required"] == ["choice_id", "choices"]
    assert "choice_id" in schema["properties"]
    assert "id" not in schema["properties"]
    assert choice_schema["required"] == ["id", "text", "target_label"]
    assert "target_label" in choice_schema["properties"]
    assert "target" not in choice_schema["properties"]
    assert "choice_id" in str(snippet["preview"])
    assert "target_label" in str(snippet["preview"])


def test_catalog_json_contains_no_validation_codes_or_routing_secrets() -> None:
    catalog = list_authoring_catalog()

    serialized = str(catalog).lower()
    assert "validation_codes" not in serialized
    assert "api_key" not in serialized
    assert "provider_config" not in serialized
    assert "raw prompt" not in serialized
    assert "raw_prompt" not in serialized
    for key in forbidden_generation_routing_keys():
        assert key not in _forbidden_keys_present(catalog)


def test_snippet_parameter_schemas_forbid_extra_object_fields_recursively() -> None:
    catalog = list_authoring_catalog()

    assert catalog["snippets"]
    for snippet in catalog["snippets"]:
        assert snippet["schema_version"] == "vn_script_program.v1"
        _assert_object_schemas_are_closed(snippet["parameters_schema"], f"$.snippets.{snippet['id']}.parameters_schema")


def test_catalog_payloads_are_isolated_between_calls() -> None:
    first = list_authoring_catalog()
    first["operations"][0]["label"] = "mutated"
    first["snippets"][0]["parameters_schema"]["properties"]["unexpected"] = {"type": "string"}
    first["limits"]["max_title_length"] = 1

    second = list_authoring_catalog()

    assert second["operations"][0]["label"] != "mutated"
    assert "unexpected" not in second["snippets"][0]["parameters_schema"]["properties"]
    assert second["limits"]["max_title_length"] != 1

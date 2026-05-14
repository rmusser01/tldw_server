from __future__ import annotations

from collections.abc import Generator, Mapping
from copy import deepcopy
import json
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import VNAssetPackCreate
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Scripts.authoring_graph import (
    GRAPH_SEMANTICS_VERSION,
    MAX_EDGES,
    MAX_LABELS,
    MAX_OPS,
    MAX_SUPPLIED_DRAFT_BYTES,
    PROGRAM_SCHEMA_VERSION,
    SCHEMA_VERSION,
    build_script_authoring_graph,
    content_hash_for_program,
)
from tldw_Server_API.app.core.VN_Scripts.service import VNScriptService
from tldw_Server_API.app.core.VN_Scripts.validator import (
    VNScriptValidationContext,
    validate_script_program,
)


def _program() -> dict[str, Any]:
    return {
        "schema_version": PROGRAM_SCHEMA_VERSION,
        "primary_asset_pack_id": 7,
        "entry_label": "intro.scene",
        "labels": {
            "intro.scene": [
                {"op": "narrate", "text": "Opening."},
                {"op": "jump", "target": "end label"},
            ],
            "end label": [{"op": "end"}],
        },
    }


def _manifest(*, pack_id: int = 7, slot_key: str = "background.archive.default") -> dict[str, Any]:
    return {
        "schema_version": "vn_asset_manifest.v1",
        "pack_id": pack_id,
        "title": f"Pack {pack_id}",
        "primary_character_id": None,
        "content_rating": "general",
        "assets": {
            "backgrounds": [{"slot_key": slot_key, "item_id": 100, "mime_type": "image/png"}],
            "sprites": [],
            "depth_companions": [],
            "cgs": [],
        },
    }


def _service(chacha_db: CharactersRAGDB, *, owner_user_id: int = 42) -> VNScriptService:
    return VNScriptService(
        chacha_db,
        owner_user_id=owner_user_id,
        manifest_resolver=lambda asset_pack_id: _manifest(pack_id=asset_pack_id),
    )


def _create_script(service: VNScriptService, draft: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return service.create_script(
        title="Archive Door",
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
        content_rating="general",
        initial_draft=draft or _program(),
        initial_diagnostics={"valid": False, "errors": [{"code": "stale_stored"}], "warnings": []},
    )


def _create_character_pack(chacha_db: CharactersRAGDB, *, age_status: str = "adult") -> tuple[int, int]:
    character_id = chacha_db.add_character_card(
        {
            "name": "Adult Mira",
            "description": "A careful archivist.",
            "personality": "Patient and exacting.",
            "scenario": "Cataloging an orbital library.",
            "extensions": {"safety_metadata": {"age_status": age_status}},
        }
    )
    pack = VNAssetPackService(chacha_db, owner_user_id=42).create_pack(
        VNAssetPackCreate(title="Character Pack", primary_character_id=character_id)
    )
    return pack.id, character_id


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-script-authoring-graph-test-client")
    yield database
    database.close_connection()


def _diagnostic_codes(result: dict[str, Any], severity: str) -> list[str]:
    return [diagnostic["code"] for diagnostic in result["diagnostics"][severity]]


def test_build_script_authoring_graph_returns_envelope_outline_graph_and_hash() -> None:
    result = build_script_authoring_graph(
        _program(),
        source="supplied_draft",
        script_id=12,
        base_revision=4,
        validation_diagnostics={"valid": True, "errors": [], "warnings": []},
    )

    assert result["schema_version"] == SCHEMA_VERSION
    assert result["graph_semantics_version"] == GRAPH_SEMANTICS_VERSION
    assert result["program_schema_version"] == PROGRAM_SCHEMA_VERSION
    assert result["source"] == "supplied_draft"
    assert result["script_id"] == 12
    assert result["base_revision"] == 4
    assert result["version_id"] is None
    assert result["content_hash"].startswith("sha256:")
    assert result["validation_context_source"] == "current_draft_context"
    assert result["truncated"] is False
    assert result["limits"] == {
        "max_labels": MAX_LABELS,
        "max_ops": MAX_OPS,
        "max_edges": MAX_EDGES,
        "max_supplied_draft_bytes": MAX_SUPPLIED_DRAFT_BYTES,
    }
    assert result["validation_diagnostics"] == {"valid": True, "errors": [], "warnings": []}

    assert result["outline"]["entry_label"] == "intro.scene"
    assert result["outline"]["labels"][0] == {
        "id": "label:intro%2Escene",
        "label": "intro.scene",
        "source_path": "$.labels['intro.scene']",
        "op_count": 2,
        "incoming_edge_count": 0,
        "outgoing_edge_count": 1,
        "reachable": True,
        "terminal": "continues",
        "summary": "2 operations and 1 outgoing edge.",
    }
    assert result["graph"]["nodes"][0]["id"] == "label:intro%2Escene"
    assert result["graph"]["nodes"][1]["id"] == "op:intro%2Escene:0"
    assert result["graph"]["nodes"][2]["source_path"] == "$.labels['intro.scene'][1]"
    assert result["graph"]["edges"][0]["type"] == "jump"
    assert result["graph"]["edges"][0]["target_id"] == "label:end%20label"
    assert result["graph"]["edges"][0]["source_path"] == "$.labels['intro.scene'][1].target"


def test_label_ids_are_percent_encoded_but_display_label_remains_raw() -> None:
    result = build_script_authoring_graph(
        {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "entry_label": "route:1/a",
            "labels": {"route:1/a": [{"op": "end"}]},
        }
    )

    assert result["outline"]["labels"][0]["id"] == "label:route%3A1%2Fa"
    assert result["outline"]["labels"][0]["label"] == "route:1/a"
    assert result["graph"]["nodes"][1]["id"] == "op:route%3A1%2Fa:0"


def test_bracket_paths_escape_label_quotes_and_backslashes() -> None:
    result = build_script_authoring_graph(
        {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "entry_label": "intro'\\scene",
            "labels": {"intro'\\scene": [{"op": "end"}]},
        }
    )

    assert result["outline"]["labels"][0]["source_path"] == "$.labels['intro\\'\\\\scene']"
    assert result["graph"]["nodes"][1]["source_path"] == "$.labels['intro\\'\\\\scene'][0]"


def test_graph_extracts_only_static_jump_choice_generate_and_cancel_edges() -> None:
    program = {
        "schema_version": PROGRAM_SCHEMA_VERSION,
        "entry_label": "start",
        "labels": {
            "start": [
                {"op": "choice", "choices": [{"id": "left", "text": "Left", "target": "left"}]},
                {
                    "op": "generate",
                    "output_schema": "choice_set",
                    "on_generated_choice": "generated",
                    "on_cancel": "cancel",
                },
                {"op": "random", "branches": [{"target": "ignored"}]},
                {"op": "jump", "target": "done"},
            ],
            "left": [{"op": "end"}],
            "generated": [{"op": "end"}],
            "cancel": [{"op": "end"}],
            "done": [{"op": "end"}],
            "ignored": [{"op": "end"}],
        },
    }

    result = build_script_authoring_graph(program)

    assert [edge["type"] for edge in result["graph"]["edges"]] == [
        "choice",
        "generated_choice_handler",
        "generation_cancel",
        "jump",
    ]
    assert [edge["source_path"] for edge in result["graph"]["edges"]] == [
        "$.labels['start'][0].choices[0].target",
        "$.labels['start'][1].on_generated_choice",
        "$.labels['start'][1].on_cancel",
        "$.labels['start'][3].target",
    ]
    assert all(edge["target_label"] != "ignored" for edge in result["graph"]["edges"])


def test_missing_targets_emit_edges_and_diagnostics() -> None:
    result = build_script_authoring_graph(
        {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "entry_label": "start",
            "labels": {"start": [{"op": "jump", "target": "missing"}]},
        }
    )

    assert result["graph"]["edges"][0]["target_id"] is None
    assert result["graph"]["edges"][0]["missing_target"] is True
    assert result["graph"]["edges"][0]["id"] == "edge:op:start:0:jump:missing:missing"
    assert result["diagnostics"]["errors"][0]["code"] == "graph_target_missing"
    assert result["diagnostics"]["errors"][0]["severity"] == "error"
    assert result["diagnostics"]["errors"][0]["path"] == "$.labels['start'][0].target"
    assert result["diagnostics"]["errors"][0]["details"] == {
        "target_label": "missing",
        "edge_type": "jump",
    }


def test_graph_reachability_matches_validator_unreachable_warnings() -> None:
    program = {
        "schema_version": PROGRAM_SCHEMA_VERSION,
        "primary_asset_pack_id": 7,
        "entry_label": "start",
        "labels": {
            "start": [{"op": "choice", "choices": [{"text": "A", "target": "choice_target"}]}],
            "choice_target": [{"op": "generate", "on_cancel": "cancel", "on_generated_choice": "generated"}],
            "cancel": [{"op": "end"}],
            "generated": [{"op": "jump", "target": "done"}],
            "done": [{"op": "end"}],
            "orphan": [{"op": "end"}],
        },
    }

    graph = build_script_authoring_graph(program)
    validation = validate_script_program(program, VNScriptValidationContext()).to_dict()

    graph_unreachable = {
        diagnostic["details"]["label"]
        for diagnostic in graph["diagnostics"]["warnings"]
        if diagnostic["code"] == "graph_label_unreachable"
    }
    validator_unreachable = {
        diagnostic["details"]["label"]
        for diagnostic in validation["warnings"]
        if diagnostic["code"] == "label_unreachable"
    }

    assert graph_unreachable == validator_unreachable == {"orphan"}
    assert {label["label"] for label in graph["outline"]["labels"] if label["reachable"]} == {
        "start",
        "choice_target",
        "cancel",
        "generated",
        "done",
    }


def test_graph_warns_that_static_reachability_does_not_infer_fallthrough() -> None:
    result = build_script_authoring_graph(
        {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "entry_label": "start",
            "labels": {
                "start": [{"op": "narrate", "text": "Opening."}],
                "next": [{"op": "end"}],
            },
        }
    )

    fallthrough_warnings = [
        diagnostic
        for diagnostic in result["diagnostics"]["warnings"]
        if diagnostic["code"] == "graph_fallthrough_not_inferred"
    ]
    assert fallthrough_warnings == [
        {
            "code": "graph_fallthrough_not_inferred",
            "severity": "warning",
            "message": "Static graph reachability does not infer implicit fallthrough to the next label.",
            "path": "$.labels['start']",
            "details": {"label": "start", "next_label": "next"},
        }
    ]


def test_terminal_classification_is_conservative() -> None:
    result = build_script_authoring_graph(
        {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "entry_label": "start",
            "labels": {
                "start": [{"op": "jump", "target": "terminal"}],
                "terminal": [{"op": "narrate", "text": "Bye."}, {"op": "end"}],
                "returning": [{"op": "return"}],
                "randomized": [{"op": "random", "branches": [{"target": "terminal"}]}],
                "conditional": [{"op": "jump", "target": "terminal", "if": {"var": "x", "op": "eq", "value": 1}}],
                "malformed": ["not an opcode"],
                "ambiguous": [{"op": "narrate", "text": "No explicit control flow."}],
            },
        }
    )

    terminal_by_label = {label["label"]: label["terminal"] for label in result["outline"]["labels"]}

    assert terminal_by_label["start"] == "continues"
    assert terminal_by_label["terminal"] == "terminal"
    assert terminal_by_label["returning"] == "unknown"
    assert terminal_by_label["randomized"] == "unknown"
    assert terminal_by_label["conditional"] == "unknown"
    assert terminal_by_label["malformed"] == "unknown"
    assert terminal_by_label["ambiguous"] == "unknown"
    assert "graph_unsupported_dynamic_flow" in _diagnostic_codes(result, "warnings")


def test_malformed_but_parseable_drafts_return_partial_graph_and_diagnostics() -> None:
    result = build_script_authoring_graph(
        {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "entry_label": "start",
            "labels": {
                "start": [
                    "bad op",
                    {"op": "choice", "choices": {"bad": "shape"}},
                    {"op": "generate", "on_generated_choice": 123, "on_cancel": None},
                ],
                "broken": {"not": "a list"},
            },
        }
    )

    assert result["truncated"] is False
    assert result["outline"]["labels"][0]["label"] == "start"
    assert result["outline"]["labels"][1]["label"] == "broken"
    assert _diagnostic_codes(result, "errors") == [
        "graph_opcode_invalid",
        "graph_choice_options_invalid",
        "graph_generated_choice_handler_missing",
        "graph_cancel_target_missing",
        "graph_label_body_invalid",
    ]


def test_missing_labels_returns_empty_partial_graph_with_diagnostic() -> None:
    result = build_script_authoring_graph({"schema_version": PROGRAM_SCHEMA_VERSION, "entry_label": "start"})

    assert result["outline"]["labels"] == []
    assert result["graph"]["nodes"] == []
    assert result["graph"]["edges"] == []
    assert result["diagnostics"]["errors"][0]["code"] == "graph_labels_missing"


def test_content_hash_is_canonical_stable_and_semantics_scoped() -> None:
    left = {
        "labels": {"start": [{"text": "Opening.", "op": "narrate"}, {"op": "end"}]},
        "entry_label": "start",
        "schema_version": PROGRAM_SCHEMA_VERSION,
    }
    right = json.loads(json.dumps(left, indent=2))

    assert build_script_authoring_graph(left)["content_hash"] == build_script_authoring_graph(right)["content_hash"]
    assert content_hash_for_program(left) != content_hash_for_program(left, graph_semantics_version="future.v2")

    changed = deepcopy(left)
    changed["labels"]["start"][0]["text"] = "Changed."
    assert build_script_authoring_graph(left)["content_hash"] != build_script_authoring_graph(changed)["content_hash"]


def test_builder_does_not_mutate_input_or_leak_full_operation_payloads() -> None:
    program = {
        "schema_version": PROGRAM_SCHEMA_VERSION,
        "entry_label": "start",
        "labels": {
            "start": [
                {
                    "op": "generate",
                    "output_schema": "choice_set",
                    "profile_key": "safe_profile",
                    "on_generated_choice": "generated",
                    "api_key": "sk-secret",
                    "model": "gpt-secret",
                    "provider_config": {"base_url": "https://provider.invalid"},
                }
            ],
            "generated": [{"op": "end"}],
        },
    }
    original = deepcopy(program)

    result = build_script_authoring_graph(program)
    serialized = json.dumps(result, sort_keys=True)

    assert program == original
    assert "sk-secret" not in serialized
    assert "gpt-secret" not in serialized
    assert "https://provider.invalid" not in serialized
    assert all("payload" not in node and "operation" not in node for node in result["graph"]["nodes"])
    assert result["graph"]["nodes"][1] == {
        "id": "op:start:0",
        "type": "operation",
        "label": "start",
        "op_index": 0,
        "op": "generate",
        "source_path": "$.labels['start'][0]",
        "summary": "Generate choice_set using profile safe_profile.",
    }


def test_ordering_is_deterministic_with_entry_label_first_and_source_order_afterward() -> None:
    result = build_script_authoring_graph(
        {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "entry_label": "middle",
            "labels": {
                "alpha": [{"op": "end"}],
                "middle": [
                    {"op": "generate", "on_cancel": "omega", "on_generated_choice": "alpha"},
                    {"op": "choice", "choices": [{"target": "alpha"}, {"target": "omega"}]},
                ],
                "omega": [{"op": "end"}],
            },
        }
    )

    assert [label["label"] for label in result["outline"]["labels"]] == ["middle", "alpha", "omega"]
    assert [node["id"] for node in result["graph"]["nodes"]] == [
        "label:middle",
        "op:middle:0",
        "op:middle:1",
        "label:alpha",
        "op:alpha:0",
        "label:omega",
        "op:omega:0",
    ]
    assert [edge["type"] for edge in result["graph"]["edges"]] == [
        "generated_choice_handler",
        "generation_cancel",
        "choice",
        "choice",
    ]


def test_graph_limits_return_partial_truncated_graph_with_diagnostics() -> None:
    label_limited = build_script_authoring_graph(
        {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "entry_label": "start",
            "labels": {"start": [{"op": "jump", "target": "second"}], "second": [{"op": "end"}]},
        },
        limits={"max_labels": 1, "max_ops": 10, "max_edges": 10},
    )

    assert label_limited["truncated"] is True
    assert [label["label"] for label in label_limited["outline"]["labels"]] == ["start"]
    assert "graph_node_limit_exceeded" in _diagnostic_codes(label_limited, "warnings")

    op_limited = build_script_authoring_graph(
        {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "entry_label": "start",
            "labels": {"start": [{"op": "narrate"}, {"op": "end"}]},
        },
        limits={"max_labels": 10, "max_ops": 1, "max_edges": 10},
    )

    assert op_limited["truncated"] is True
    assert [node["id"] for node in op_limited["graph"]["nodes"]] == ["label:start", "op:start:0"]
    assert op_limited["diagnostics"]["warnings"][-1]["code"] == "graph_node_limit_exceeded"

    edge_limited = build_script_authoring_graph(
        {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "entry_label": "start",
            "labels": {
                "start": [
                    {
                        "op": "choice",
                        "choices": [
                            {"target": "a"},
                            {"target": "b"},
                        ],
                    }
                ],
                "a": [{"op": "end"}],
                "b": [{"op": "end"}],
            },
        },
        limits={"max_labels": 10, "max_ops": 10, "max_edges": 1},
    )

    assert edge_limited["truncated"] is True
    assert len(edge_limited["graph"]["edges"]) == 1
    assert "graph_edge_limit_exceeded" in _diagnostic_codes(edge_limited, "warnings")
    assert {label["label"] for label in edge_limited["outline"]["labels"] if label["reachable"]} == {
        "start",
        "a",
    }
    assert "b" in {
        diagnostic["details"]["label"]
        for diagnostic in edge_limited["diagnostics"]["warnings"]
        if diagnostic["code"] == "graph_label_unreachable"
    }


def test_label_limit_truncation_does_not_emit_edges_to_omitted_label_nodes() -> None:
    result = build_script_authoring_graph(
        {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "entry_label": "start",
            "labels": {
                "start": [{"op": "jump", "target": "omitted"}],
                "omitted": [{"op": "end"}],
            },
        },
        limits={"max_labels": 1, "max_ops": 10, "max_edges": 10},
    )

    emitted_node_ids = {node["id"] for node in result["graph"]["nodes"]}

    assert "label:omitted" not in emitted_node_ids
    assert result["graph"]["edges"][0]["target_id"] is None
    assert result["graph"]["edges"][0]["omitted_target"] is True
    assert result["diagnostics"]["warnings"][-1]["code"] == "graph_target_omitted"


def test_duplicate_static_edges_from_same_operation_have_unique_ids() -> None:
    result = build_script_authoring_graph(
        {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "entry_label": "start",
            "labels": {
                "start": [
                    {
                        "op": "choice",
                        "choices": [
                            {"id": "left", "text": "Left", "target": "done"},
                            {"id": "right", "text": "Right", "target": "done"},
                        ],
                    }
                ],
                "done": [{"op": "end"}],
            },
        }
    )

    edge_ids = [edge["id"] for edge in result["graph"]["edges"]]

    assert len(edge_ids) == 2
    assert len(set(edge_ids)) == 2


def test_default_validation_diagnostics_are_fresh_for_each_result() -> None:
    first = build_script_authoring_graph(_program())
    first["validation_diagnostics"]["errors"].append({"code": "mutated"})

    second = build_script_authoring_graph(_program())

    assert second["validation_diagnostics"] == {"valid": False, "errors": [], "warnings": []}


def test_service_get_draft_graph_is_non_mutating_and_uses_live_validation(chacha_db: CharactersRAGDB) -> None:
    service = _service(chacha_db)
    script = _create_script(service, draft=_program())
    before = service.get_draft(script["id"])

    result = service.get_draft_graph(script["id"])
    after = service.get_draft(script["id"])

    assert result["source"] == "stored_draft"
    assert result["script_id"] == script["id"]
    assert result["base_revision"] == before["revision"]
    assert result["validation_context_source"] == "current_draft_context"
    assert result["validation_diagnostics"]["valid"] is True
    assert "stale_stored" not in {
        error["code"] for error in result["validation_diagnostics"].get("errors", [])
    }
    assert after == before


def test_service_graph_methods_require_ownership(chacha_db: CharactersRAGDB) -> None:
    owner_service = _service(chacha_db, owner_user_id=42)
    other_service = _service(chacha_db, owner_user_id=7)
    script = _create_script(owner_service, draft=_program())

    with pytest.raises(ValueError, match="script_not_found"):
        other_service.get_draft_graph(script["id"])

    with pytest.raises(ValueError, match="script_not_found"):
        other_service.preview_draft_graph(script["id"], _program())


def test_service_preview_draft_graph_accepts_stale_revision_with_warning(chacha_db: CharactersRAGDB) -> None:
    service = _service(chacha_db)
    script = _create_script(service, draft=_program())
    before = service.get_draft(script["id"])

    supplied = _program()
    supplied["labels"]["intro.scene"][0]["text"] = "Unsaved opening."
    result = service.preview_draft_graph(script["id"], supplied, draft_revision=-1)
    after = service.get_draft(script["id"])

    assert result["source"] == "supplied_draft"
    assert result["script_id"] == script["id"]
    assert result["base_revision"] == before["revision"] == 1
    assert result["content_hash"] == content_hash_for_program(supplied)
    assert result["validation_context_source"] == "current_draft_context"
    assert any(
        diagnostic["code"] == "graph_preview_revision_stale"
        for diagnostic in result["diagnostics"]["warnings"]
    )
    assert after == before


def test_service_preview_draft_graph_rejects_invalid_shape_and_oversized_drafts(
    chacha_db: CharactersRAGDB,
) -> None:
    service = _service(chacha_db)
    script = _create_script(service, draft=_program())

    with pytest.raises(ValueError, match="supplied_draft_invalid_shape"):
        service.preview_draft_graph(script["id"], ["not", "a", "mapping"])  # type: ignore[arg-type]

    oversized_draft = {"schema_version": PROGRAM_SCHEMA_VERSION, "blob": "x" * MAX_SUPPLIED_DRAFT_BYTES}
    with pytest.raises(ValueError, match="supplied_draft_too_large"):
        service.preview_draft_graph(script["id"], oversized_draft)


def test_service_version_graph_uses_published_version_snapshot_context(chacha_db: CharactersRAGDB) -> None:
    service = _service(chacha_db)
    script = _create_script(service, draft=_program())
    published = service.publish_script(
        script["id"],
        draft_revision=1,
        label="v1",
        idempotency_key="publish-graph",
        acknowledgements=["character_safety_missing"],
    )
    service.update_script(
        script["id"],
        {
            "primary_asset_pack_id": 8,
            "policy_profile_id": "strict_hosted",
            "content_rating": "mature",
        },
    )
    mutated_draft = _program()
    mutated_draft["primary_asset_pack_id"] = 8
    mutated_draft["labels"]["intro.scene"][0]["text"] = "Mutable draft changed."
    service.replace_draft(script["id"], if_revision=1, draft=mutated_draft)

    result = service.get_version_graph(script["id"], published["version_id"])

    assert result["source"] == "published_version"
    assert result["script_id"] == script["id"]
    assert result["base_revision"] is None
    assert result["version_id"] == published["version_id"]
    assert result["content_hash"] == content_hash_for_program(_program())
    assert result["validation_context_source"] == "published_version_snapshot"
    assert result["validation_diagnostics"]["valid"] is True
    assert "primary_asset_pack_mismatch" not in {
        error["code"] for error in result["validation_diagnostics"].get("errors", [])
    }


def test_service_version_graph_reuses_published_validation_after_live_character_changes(
    chacha_db: CharactersRAGDB,
) -> None:
    pack_id, character_id = _create_character_pack(chacha_db, age_status="adult")
    service = VNScriptService(chacha_db, owner_user_id=42)
    program = _program()
    program["primary_asset_pack_id"] = pack_id
    script = service.create_script(
        title="Archive Door",
        primary_asset_pack_id=pack_id,
        policy_profile_id="strict_hosted",
        generation_profile_id="story_default",
        content_rating="general",
        initial_draft=program,
        initial_diagnostics={"valid": True, "errors": [], "warnings": []},
    )
    published = service.publish_script(
        script["id"],
        draft_revision=1,
        label="adult",
        idempotency_key="publish-adult-character",
        acknowledgements=[],
    )
    version = service.get_version(script["id"], published["version_id"])
    original_validation = dict(version["validation"])
    character = chacha_db.get_character_card_by_id(character_id)

    assert original_validation["valid"] is True
    assert character is not None
    assert chacha_db.update_character_card(
        character_id,
        {"extensions": {"safety_metadata": {"age_status": "missing"}}},
        int(character["version"]),
    )

    result = service.get_version_graph(script["id"], published["version_id"])

    assert result["validation_context_source"] == "published_version_snapshot"
    assert result["validation_diagnostics"] == original_validation

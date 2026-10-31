# tests/unit/core/Prompts_Management/test_prompts_interop.py
# Description:
#
# Imports
import json
import os

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase
from tldw_Server_API.app.core.DB_Management.prompts_db_helpers import (
    parse_stored_prompt_definition,
)

#
# Local Imports
from tldw_Server_API.app.core.Prompt_Management.Prompts_Interop import (
    InputError,
    PromptsInteropService,
    # Import standalone wrappers
    add_or_update_prompt_interop,
    export_prompts_formatted_interop,
    get_db_instance,
    initialize_interop,
    is_initialized,
    shutdown_interop,
)
from tldw_Server_API.app.core.Prompt_Management.Prompts_Interop import (
    add_prompt as interop_add_prompt,
)
from tldw_Server_API.app.core.Prompt_Management.Prompts_Interop import (
    fetch_prompt_details as interop_fetch_prompt_details,
)

#
#######################################################################################################################
#
# Functions:
TEST_INTEROP_CLIENT_ID = "test_interop_client"


def _make_recipe_definition() -> dict:
    return {
        "schema_version": 2,
        "format": "structured",
        "definition_kind": "single_text_recipe",
        "variables": [
            {
                "name": "topic",
                "label": "Topic",
                "required": True,
                "default_value": None,
                "input_type": "text",
            }
        ],
        "blocks": [
            {
                "id": "objective",
                "name": "Objective",
                "section_key": "objective",
                "role": "user",
                "kind": "objective",
                "content": "Explain {{topic}}.",
                "enabled": True,
                "order": 10,
                "is_template": True,
            }
        ],
        "assembly_config": {
            "assembly_mode": "single_text",
            "target_role": "user",
            "render_format": "markdown",
            "block_separator": "\n\n",
        },
    }


def _make_v1_definition() -> dict:
    return {
        "schema_version": 1,
        "variables": [],
        "blocks": [],
    }


@pytest.fixture(scope="function")  # function scope to ensure clean init/shutdown for each test
def interop_manager(tmp_path):
    """Manages initialization and shutdown of the interop layer for tests."""
    db_file = tmp_path / "interop_test_prompts.db"
    try:
        initialize_interop(db_path=db_file, client_id=TEST_INTEROP_CLIENT_ID)
        yield db_file  # provide the path if needed by tests
    finally:
        shutdown_interop()
        if os.path.exists(db_file):
            os.remove(db_file)


@pytest.mark.integration
def test_interop_initialization_and_shutdown(interop_manager):

    assert is_initialized() is True
    db_instance = get_db_instance()
    assert db_instance is not None
    assert isinstance(db_instance, PromptsDatabase)
    assert db_instance.client_id == TEST_INTEROP_CLIENT_ID

    shutdown_interop()  # Explicitly call shutdown within test for this case
    assert is_initialized() is False
    with pytest.raises(RuntimeError, match="Prompts Interop Library not initialized"):
        get_db_instance()
    # Re-initialize for the fixture's finally block to work correctly
    initialize_interop(db_path=interop_manager, client_id=TEST_INTEROP_CLIENT_ID)


@pytest.mark.integration
def test_interop_add_prompt_via_global_instance(interop_manager):

    assert is_initialized() is True  # Ensure interop_manager fixture worked
    p_id, p_uuid, msg = interop_add_prompt(name="Interop Prompt", author="Interop", details="Via global instance")
    assert p_id is not None
    assert "added" in msg

    details = interop_fetch_prompt_details(p_uuid)
    assert details is not None
    assert details["name"] == "Interop Prompt"


@pytest.mark.integration
def test_interop_add_prompt_preserves_legacy_keyword_positionals(interop_manager):

    p_id, p_uuid, msg = interop_add_prompt(
        "Interop Keyword Prompt",
        "Interop",
        "Keyword positional compatibility",
        None,
        None,
        ["legacy_kw"],
        False,
    )

    assert p_id is not None
    assert "added" in msg

    details = interop_fetch_prompt_details(p_uuid)
    assert details is not None
    assert "legacy_kw" in details["keywords"]


@pytest.mark.integration
def test_interop_add_prompt_passes_structured_fields_and_keywords(interop_manager):

    prompt_definition = {
        "schema_version": 1,
        "format": "structured",
        "variables": [],
        "blocks": [
            {
                "id": "task",
                "name": "Task",
                "role": "user",
                "content": "Summarize this input.",
                "enabled": True,
                "order": 10,
                "is_template": False,
            }
        ],
    }

    p_id, p_uuid, msg = interop_add_prompt(
        name="Interop Structured Prompt",
        author="Interop",
        details="Structured via global instance",
        prompt_format="structured",
        prompt_schema_version=1,
        prompt_definition=prompt_definition,
        keywords=["structured_kw"],
    )

    assert p_id is not None
    assert "added" in msg

    details = interop_fetch_prompt_details(p_uuid)
    assert details is not None
    assert details["prompt_format"] == "structured"
    assert details["prompt_schema_version"] == 1
    assert details["prompt_definition"]["blocks"][0]["content"] == "Summarize this input."
    assert "structured_kw" in details["keywords"]


@pytest.mark.integration
def test_interop_add_prompt_preserves_recipe_identity(interop_manager):
    definition = _make_recipe_definition()
    canonical_definition = parse_stored_prompt_definition(definition).model_dump()

    p_id, p_uuid, msg = interop_add_prompt(
        name="Interop Recipe",
        author="Interop",
        details="Recipe via global instance",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=definition,
        keywords=["recipe"],
    )

    assert p_id is not None
    assert "added" in msg
    details = interop_fetch_prompt_details(p_uuid)
    assert details is not None
    assert details["prompt_format"] == "structured"
    assert details["prompt_schema_version"] == 2
    assert details["prompt_definition"] == canonical_definition
    assert details["user_prompt"] == "## Objective\n\nExplain {{topic}}."


@pytest.mark.integration
def test_interop_json_export_import_round_trip_preserves_recipe_identity(tmp_path):
    source = PromptsInteropService(str(tmp_path / "source"), "interop-source")
    destination = PromptsInteropService(str(tmp_path / "destination"), "interop-destination")
    definition = _make_recipe_definition()
    canonical_definition = parse_stored_prompt_definition(definition).model_dump()
    try:
        prompt_id = source.create_prompt(
            name="Portable Recipe",
            content="Recipe details",
            author="Interop",
            keywords=["recipe"],
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=definition,
        )

        exported = source.bulk_export(prompt_ids=[prompt_id])
        assert exported["version"] == "1.0"
        assert exported["prompts"] == [
            {
                "name": "Portable Recipe",
                "content": "Recipe details",
                "author": "Interop",
                "keywords": ["recipe"],
                "prompt_format": "structured",
                "prompt_schema_version": 2,
                "prompt_definition": canonical_definition,
                "system_prompt": "",
                "user_prompt": "## Objective\n\nExplain {{topic}}.",
            }
        ]
        assert "runtime_values" not in source.export_prompts_json(prompt_ids=[prompt_id])

        result = destination.import_prompts(exported)
        imported = destination.get_prompt(result["prompt_ids"][0])
        assert imported["prompt_format"] == "structured"
        assert imported["prompt_schema_version"] == 2
        assert imported["prompt_definition"] == canonical_definition
        assert imported["user_prompt"] == "## Objective\n\nExplain {{topic}}."
    finally:
        source.close()
        destination.close()


@pytest.mark.integration
def test_interop_json_import_rejects_runtime_values_without_partial_write(tmp_path):
    service = PromptsInteropService(str(tmp_path / "guard"), "interop-guard")
    unsafe_definition = _make_recipe_definition()
    unsafe_definition["runtime_values"] = {"topic": "PRIVATE_VALUE"}
    try:
        with pytest.raises(InputError, match="invalid_recipe_runtime_values"):
            service.import_prompts(
                {
                    "version": "1.0",
                    "prompts": [
                        {
                            "name": "Unsafe Recipe",
                            "content": "Recipe details",
                            "author": "Interop",
                            "keywords": ["recipe"],
                            "prompt_format": "structured",
                            "prompt_schema_version": 2,
                            "prompt_definition": unsafe_definition,
                            "system_prompt": "",
                            "user_prompt": "PRIVATE_VALUE",
                        }
                    ],
                }
            )

        assert service.list_prompts() == []
    finally:
        service.close()


def _interop_state(service: PromptsInteropService) -> str:
    db = service._ensure_db()
    return json.dumps(
        {
            "prompts": service.list_prompts(),
            "sync": db.get_sync_log_entries(),
            "names": service._name_overrides,
            "keywords": service._orig_keywords,
        },
        sort_keys=True,
        default=str,
    )


@pytest.mark.parametrize("runtime_key", ["runtime_values", "variable_values", "resolved_values"])
@pytest.mark.parametrize("unsafe_index", [0, 1])
@pytest.mark.integration
def test_interop_json_import_rejects_runtime_maps_on_every_record_before_first_mutation(
    tmp_path,
    runtime_key: str,
    unsafe_index: int,
) -> None:
    service = PromptsInteropService(str(tmp_path / f"batch-{runtime_key}-{unsafe_index}"), "interop-batch")
    try:
        service.create_prompt(name="Good Local", content="Preserve me", author="Interop")
        good_import = {
            "name": "Good Import",
            "content": "runtime_values variable_values resolved_values are authored words",
            "author": "Interop",
            "keywords": [],
        }
        marker = f"SECRET_INTEROP_{runtime_key}_{unsafe_index}"
        unsafe_import = {
            "name": "Unsafe Import",
            "content": "Unsafe",
            "author": "Interop",
            "keywords": [],
            runtime_key: {"topic": marker},
        }
        records = [good_import, unsafe_import]
        if unsafe_index == 0:
            records.reverse()
        before = _interop_state(service)
        captured: list[str] = []
        sink_id = logger.add(captured.append, format="{message}")
        try:
            with pytest.raises(InputError, match="^invalid_recipe_runtime_values$"):
                service.import_prompts({"version": "1.0", "prompts": records})
        finally:
            logger.remove(sink_id)

        assert _interop_state(service) == before
        assert marker not in "\n".join(captured)
    finally:
        service.close()


@pytest.mark.integration
def test_interop_json_import_allows_runtime_key_words_inside_authored_strings(tmp_path) -> None:
    service = PromptsInteropService(str(tmp_path / "authored-words"), "interop-words")
    try:
        result = service.import_prompts(
            {
                "version": "1.0",
                "prompts": [
                    {
                        "name": "Authored Words",
                        "content": "runtime_values variable_values resolved_values",
                        "author": "Interop",
                        "keywords": [],
                    }
                ],
            }
        )
        assert result["imported"] == 1
    finally:
        service.close()


@pytest.mark.integration
def test_interop_json_import_rejects_recipe_without_identity_metadata(tmp_path):
    service = PromptsInteropService(str(tmp_path / "missing-identity"), "interop-identity")
    try:
        with pytest.raises(ValueError, match="invalid_recipe_prompt_format"):
            service.import_prompts(
                {
                    "version": "1.0",
                    "prompts": [
                        {
                            "name": "Unlabelled Recipe",
                            "content": "Recipe details",
                            "author": "Interop",
                            "keywords": ["recipe"],
                            "prompt_definition": _make_recipe_definition(),
                        }
                    ],
                }
            )

        assert service.list_prompts() == []
    finally:
        service.close()


@pytest.mark.parametrize(
    "identity_case",
    [
        "missing_format",
        "missing_schema",
        "missing_both",
        "legacy_format",
        "schema_mismatch",
        "future_schema",
    ],
)
@pytest.mark.parametrize("unsafe_index", [0, 1])
@pytest.mark.integration
def test_interop_json_import_requires_exact_v2_outer_identity_before_any_write(
    tmp_path,
    identity_case: str,
    unsafe_index: int,
) -> None:
    service = PromptsInteropService(
        str(tmp_path / f"v2-identity-{identity_case}-{unsafe_index}"),
        "interop-v2-identity",
    )
    try:
        service.create_prompt(name="Good Local", content="Preserve me", author="Interop")
        marker = f"PRIVATE_INTEROP_IDENTITY_{identity_case}_{unsafe_index}"
        invalid = {
            "name": "Invalid recipe identity",
            "content": marker,
            "author": "Interop",
            "keywords": [],
            "prompt_format": "structured",
            "prompt_schema_version": 2,
            "prompt_definition": _make_recipe_definition(),
            "system_prompt": marker,
            "user_prompt": marker,
        }
        if identity_case in {"missing_format", "missing_both"}:
            invalid.pop("prompt_format")
        if identity_case in {"missing_schema", "missing_both"}:
            invalid.pop("prompt_schema_version")
        if identity_case == "legacy_format":
            invalid["prompt_format"] = "legacy"
        if identity_case == "schema_mismatch":
            invalid["prompt_schema_version"] = 1
        if identity_case == "future_schema":
            invalid["prompt_schema_version"] = 99
            invalid["prompt_definition"]["schema_version"] = 99

        records = [
            {
                "name": "Good import",
                "content": "Good content",
                "author": "Interop",
                "keywords": [],
            },
            invalid,
        ]
        if unsafe_index == 0:
            records.reverse()
        before = _interop_state(service)
        captured: list[str] = []
        sink_id = logger.add(captured.append, format="{message}")
        try:
            with pytest.raises(InputError) as excinfo:
                service.import_prompts({"version": "1.0", "prompts": records})
        finally:
            logger.remove(sink_id)

        assert _interop_state(service) == before
        assert marker not in str(excinfo.value)
        assert marker not in "\n".join(captured)
    finally:
        service.close()


@pytest.mark.integration
def test_interop_json_import_retains_supported_v1_missing_outer_schema(tmp_path) -> None:
    service = PromptsInteropService(str(tmp_path / "v1-identity"), "interop-v1-identity")
    try:
        result = service.import_prompts(
            {
                "version": "1.0",
                "prompts": [
                    {
                        "name": "Compatible v1",
                        "content": "Legacy content",
                        "author": "Interop",
                        "keywords": [],
                        "prompt_format": "structured",
                        "prompt_definition": _make_v1_definition(),
                    }
                ],
            }
        )

        imported = service.get_prompt(result["prompt_ids"][0])
        assert imported["prompt_format"] == "structured"
        assert imported["prompt_schema_version"] == 1
        assert imported["prompt_definition"] == parse_stored_prompt_definition(
            _make_v1_definition()
        ).model_dump()
    finally:
        service.close()


@pytest.mark.parametrize("export_format", ["csv", "markdown"])
@pytest.mark.integration
def test_interop_lossy_formats_reject_recipe_export(interop_manager, export_format):
    interop_add_prompt(
        name="Non-lossy Recipe",
        author="Interop",
        details="Must retain schema identity",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=_make_recipe_definition(),
    )

    with pytest.raises(InputError, match="single_text_recipe_export_requires_json"):
        export_prompts_formatted_interop(export_format=export_format)


# --- Testing standalone wrapper functions from interop ---
# These take a db_instance, so we need to provide one.
# For these, the interop's global instance isn't directly used by the function itself,
# but the test setup might rely on `initialize_interop` if the function internally calls `get_db_instance`.
# The functions like `add_or_update_prompt_interop` DO use `get_db_instance()`.


@pytest.mark.integration
def test_interop_standalone_add_or_update_prompt(interop_manager):

    p_id, _, msg = add_or_update_prompt_interop(name="Interop SU Prompt", author="SU", details="Details")
    assert p_id is not None
    assert "added" in msg or "updated" in msg

    db_instance = get_db_instance()  # Get the globally managed instance
    fetched = db_instance.get_prompt_by_name("Interop SU Prompt")
    assert fetched is not None
    assert fetched["author"] == "SU"


@pytest.mark.integration
def test_interop_standalone_export_formatted(interop_manager):

    add_or_update_prompt_interop(name="Export Me Interop", author="Exporter", details="...")
    status_msg, file_path_str = export_prompts_formatted_interop(export_format="csv")

    assert "Successfully exported" in status_msg
    assert file_path_str != "None"
    if os.path.exists(file_path_str):  # file_path_str is temp file path
        with open(file_path_str) as f:
            assert "Export Me Interop" in f.read()
        os.remove(file_path_str)
    else:
        # This might happen if the DB was in-memory and export didn't write to disk for some reason
        # or if the test setup of interop_manager used :memory: (it uses tmp_path now)
        pytest.fail(f"Exported file {file_path_str} not found.")


@pytest.mark.integration
def test_interop_error_propagation(interop_manager):

    # Try to add a prompt with an empty name, should raise InputError from DB layer
    with pytest.raises(InputError):
        interop_add_prompt(name="", author="Test", details="...")


# Test calling get_db_instance when not initialized (outside fixture)
@pytest.mark.unit
def test_get_db_instance_not_initialized():
    # Ensure it's shutdown if a previous test didn't clean up fully in some error case
    if is_initialized():
        shutdown_interop()
    with pytest.raises(RuntimeError, match="Prompts Interop Library not initialized"):
        get_db_instance()


#
# End of test_prompts_interop.py
#######################################################################################################################

# tldw_Server_API/tests/Chat/test_prompt_template_manager.py
import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from tldw_Server_API.app.core.Chat.prompt_template_manager import (
    PromptTemplate,
    load_template,
    apply_template_to_string,
    get_available_templates,
    DEFAULT_RAW_PASSTHROUGH_TEMPLATE,
    _loaded_templates  # For clearing cache in tests
)


# Fixture to clear the template cache before each test
@pytest.fixture(autouse=True)
def clear_template_cache():
    _loaded_templates.clear()
    # Re-add the default passthrough because it's normally added at module load
    _loaded_templates["raw_passthrough"] = DEFAULT_RAW_PASSTHROUGH_TEMPLATE


@pytest.fixture
def mock_templates_dir(tmp_path: Path):
    templates_dir = tmp_path / "prompt_templates_test"
    templates_dir.mkdir()

    # Create a valid template file
    valid_template_data = {
        "name": "test_valid",
        "description": "A valid test template",
        "system_message_template": "System: {sys_var}",
        "user_message_content_template": "User: {user_var} - {message_content}"
    }
    with open(templates_dir / "test_valid.json", "w") as f:
        json.dump(valid_template_data, f)

    # Create an invalid JSON template file
    with open(templates_dir / "test_invalid_json.json", "w") as f:
        f.write("{'name': 'invalid', 'description': 'bad json'")  # Invalid JSON

    # Create an empty template file (valid JSON but might be handled as error by Pydantic)
    with open(templates_dir / "test_empty.json", "w") as f:
        json.dump({}, f)

    return templates_dir


@pytest.mark.unit
def test_load_template_success(mock_templates_dir):
    with patch("tldw_Server_API.app.core.Chat.prompt_template_manager.PROMPT_TEMPLATES_DIR", mock_templates_dir):
        template = load_template("test_valid")
        assert template is not None
        assert template.name == "test_valid"
        assert template.system_message_template == "System: {sys_var}"
        # Test caching
        template_cached = load_template("test_valid")
        assert template_cached is template  # Should be the same object from cache


@pytest.mark.unit
def test_load_template_not_found(mock_templates_dir):
    with patch("tldw_Server_API.app.core.Chat.prompt_template_manager.PROMPT_TEMPLATES_DIR", mock_templates_dir):
        template = load_template("non_existent_template")
        assert template is None


@pytest.mark.unit
def test_load_template_invalid_json(mock_templates_dir):
    with patch("tldw_Server_API.app.core.Chat.prompt_template_manager.PROMPT_TEMPLATES_DIR", mock_templates_dir):
        template = load_template("test_invalid_json")
        assert template is None  # Should fail to parse


@pytest.mark.unit
def test_load_template_empty_json_fails_validation(mock_templates_dir):
    with patch("tldw_Server_API.app.core.Chat.prompt_template_manager.PROMPT_TEMPLATES_DIR", mock_templates_dir):
        template = load_template("test_empty")
        # Pydantic will raise validation error because 'name' is missing,
        # load_template should catch this and return None.
        assert template is None


@pytest.mark.unit
# Inside test_apply_template_to_string():
def test_apply_template_to_string():
    template_str_jinja = "Hello {{name}}, welcome to {{place}}." # Use Jinja
    data_full = {"name": "Alice", "place": "Wonderland"}
    assert apply_template_to_string(template_str_jinja, data_full) == "Hello Alice, welcome to Wonderland."

    template_partial_jinja = "Hello {{name}}." # Use Jinja
    data_partial = {"name": "Bob"}
    assert apply_template_to_string(template_partial_jinja, data_partial) == "Hello Bob."

    # Test with missing data - Jinja renders empty for missing by default if not strict
    assert apply_template_to_string(template_partial_jinja, {}) == "Hello ."

    # Test with None template string
    assert apply_template_to_string(None, data_full) == ""


@pytest.mark.unit
def test_get_available_templates(mock_templates_dir):
    with patch("tldw_Server_API.app.core.Chat.prompt_template_manager.PROMPT_TEMPLATES_DIR", mock_templates_dir):
        available = get_available_templates()
        assert isinstance(available, list)
        assert "test_valid" in available
        assert "test_invalid_json" in available
        assert "test_empty" in available
        assert len(available) == 3


@pytest.mark.unit
def test_get_available_templates_no_dir():
    with patch("tldw_Server_API.app.core.Chat.prompt_template_manager.PROMPT_TEMPLATES_DIR", Path("/non/existent/dir")):
        available = get_available_templates()
        assert available == []


@pytest.mark.unit
def test_default_raw_passthrough_template():
    assert DEFAULT_RAW_PASSTHROUGH_TEMPLATE is not None
    assert DEFAULT_RAW_PASSTHROUGH_TEMPLATE.name == "raw_passthrough"
    data = {"message_content": "test content", "original_system_message_from_request": "system content"}

    # User message template (is "{{message_content}}")
    assert apply_template_to_string(DEFAULT_RAW_PASSTHROUGH_TEMPLATE.user_message_content_template,
                                    data) == "test content"
    # System message template (is "{{original_system_message_from_request}}")
    assert apply_template_to_string(DEFAULT_RAW_PASSTHROUGH_TEMPLATE.system_message_template,
                                    data) == "system content"

    data_empty_sys = {"original_system_message_from_request": ""}
    assert apply_template_to_string(DEFAULT_RAW_PASSTHROUGH_TEMPLATE.system_message_template,
                                    data_empty_sys) == ""

    data_missing_sys = {"message_content": "some_content"}  # original_system_message_from_request is missing
    assert apply_template_to_string(DEFAULT_RAW_PASSTHROUGH_TEMPLATE.system_message_template,
                                    data_missing_sys) == ""  # Jinja renders missing as empty


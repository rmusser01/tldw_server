# test_character_chat_lib.py
#
#
# --- Imports ---
import logging
import re

import hypothesis
import pytest
from unittest import mock
import base64
import binascii
import io
import json
import os
import time
import yaml
#
# Third Party Imports
from PIL import Image as PILImageReal
from loguru import logger as loguru_logger
from hypothesis import given, strategies as st, settings, HealthCheck
#
# Local Imports
from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib import (
    replace_placeholders,
    replace_user_placeholder,
    get_character_list_for_ui,
    extract_character_id_from_ui_choice,
    load_character_and_image,
    process_db_messages_to_ui_history,
    load_chat_and_character,
    load_character_wrapper,
    parse_character_book,
    extract_json_from_image_file,
    parse_v2_card,
    parse_v1_card,
    validate_character_book,
    validate_character_book_entry,
    validate_v2_card,
    import_character_card_from_json_string,
    load_character_card_from_string_content,
    import_and_save_character_from_file,
    load_chat_history_from_file_and_save_to_db,
    start_new_chat_session,
    list_character_conversations,
    get_conversation_metadata,
    update_conversation_metadata,
    delete_conversation_by_id,
    search_conversations_by_title_query,
    post_message_to_conversation,
    retrieve_message_details,
    retrieve_conversation_messages_for_ui,
    edit_message_content,
    set_message_ranking,
    remove_message_from_conversation,
    find_messages_in_conversation
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
    SchemaError
)
#
#######################################################################################################################
#
# Functions:

# --- MOCK time.strftime for deterministic timestamps in titles ---
MOCK_TIME_STRFTIME = "2023-10-27 10:30"

# --- Module Path for Patching (if still needed for non-DB external deps) ---
MODULE_PATH_PREFIX = "tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib"


# --- Mock PIL Image object for finer control during unit tests where PIL is mocked ---
class MockPILImageObject:
    def __init__(self, format="PNG", info=None, width=100, height=100, mode="RGBA"): # mode was added here
        self.format = format
        self.info = info if info is not None else {}
        self.width = width
        self.height = height
        self.fp = None
        self.mode = mode # And used here

    def convert(self, mode):
        new_mock = MockPILImageObject(format=self.format, info=self.info.copy(), width=self.width, height=self.height, mode=mode)
        return new_mock

    def close(self):
        if self.fp:
            self.fp.close()

    @property
    def size(self):
        return (self.width, self.height)


# --- Helper function to create dummy PNG bytes ---
def create_dummy_png_bytes(chara_data_json_str=None, is_png=True):
    if not is_png:
        return b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"

    base_png = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
    )
    # A very minimal valid IDAT chunk for a 1x1 transparent pixel
    idat_chunk = b'\x00\x00\x00\x0cIDAT\x08\xd7c`\x00\x00\x00\x02\x00\x01\xe2!\xbc\x33'
    iend_chunk = b'\x00\x00\x00\x00IEND\xaeB`\x82'

    if chara_data_json_str:
        keyword = b'chara'
        encoded_json_bytes = chara_data_json_str.encode('utf-8')
        b64_data_bytes = base64.b64encode(encoded_json_bytes)
        text_chunk_data = keyword + b'\x00' + b64_data_bytes
        chunk_type = b'tEXt'  # Using tEXt for simplicity, could be zTXt or iTXt
        chunk_len = len(text_chunk_data)
        chunk_len_bytes = chunk_len.to_bytes(4, 'big')
        # CRC calculation: type + data
        crc_input_data = chunk_type + text_chunk_data
        crc_val = binascii.crc32(crc_input_data)
        crc_val_unsigned = crc_val & 0xffffffff  # Ensure positive for to_bytes
        crc_bytes = crc_val_unsigned.to_bytes(4, 'big')
        text_chunk = chunk_len_bytes + chunk_type + text_chunk_data + crc_bytes
        return base_png + text_chunk + idat_chunk + iend_chunk
    return base_png + idat_chunk + iend_chunk


# --- Pytest Fixture for In-Memory DB Instance ---
@pytest.fixture
def db():
    """Provides a fresh in-memory CharactersRAGDB instance for each test."""
    db_instance = CharactersRAGDB(':memory:', client_id="pytest_client")
    yield db_instance
    db_instance.close_connection()


# --- Pytest Fixture for Capturing Logs (using standard logging) ---
@pytest.fixture
def caplog_handler(caplog):
    """
    Fixture to correctly set up loguru to work with pytest's caplog.
    It adds a handler that propagates loguru messages to the standard logging system.
    """

    # Ensure Loguru's default handler is removed if it exists, to avoid duplicate console output
    # or configure it not to print to stderr during tests if that's preferred.
    # For simplicity here, we just add a new handler for caplog.

    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    # Add a handler to loguru that propagates to standard logging
    # Use a unique name for the handler to avoid issues if tests are run multiple times in a session
    handler_id = loguru_logger.add(PropagateHandler(), format="{message}", level="DEBUG")

    caplog.set_level(logging.DEBUG, logger="tldw_Server_API")  # Capture DEBUG from your app's logger namespace
    caplog.set_level(logging.DEBUG)  # And generally

    yield caplog  # Test runs here

    # Remove the handler after the test
    loguru_logger.remove(handler_id)


# --- Unit Tests (Pure Logic) ---

@pytest.mark.parametrize("text, char_name, user_name, expected", [
    ("Hello {{char}}, I am {{user}}.", "Alice", "Bob", "Hello Alice, I am Bob."),
    ("User: <USER>, Char: <CHAR>", "Wizard", "Hero", "User: Hero, Char: Wizard"),
    ("{{random_user}} says hi.", None, "Guest", "Guest says hi."),
    ("", "Char", "User", ""), (None, "Char", "User", ""),
])
def test_replace_placeholders(text, char_name, user_name, expected):
    assert replace_placeholders(text, char_name, user_name) == expected


@pytest.mark.parametrize("history, user_name, expected", [
    ([("Hi {{user}}", "Hello back, {{user}}")], "Sam", [("Hi Sam", "Hello back, Sam")]),
    ([], "User", []),
])
def test_replace_user_placeholder(history, user_name, expected):
    assert replace_user_placeholder(history, user_name) == expected


@pytest.mark.parametrize("choice, expected_id, raises", [
    ("My Character (ID: 123)", 123, None), ("789", 789, None),
    ("Invalid Format", None, ValueError), ("", None, ValueError),
])
def test_extract_character_id_from_ui_choice(choice, expected_id, raises):
    if raises:
        with pytest.raises(raises):
            extract_character_id_from_ui_choice(choice)
    else:
        assert extract_character_id_from_ui_choice(choice) == expected_id


def test_process_db_messages_to_ui_history_unit():
    char_name = "Botty"
    user_name = "Human"
    db_messages = [
        {"sender": "User", "content": "Hi {{char}}"},
        {"sender": char_name, "content": "Hello {{user}}"}
    ]
    expected = [("Hi Botty", "Hello Human")]
    assert process_db_messages_to_ui_history(db_messages, char_name, user_name) == expected

    # Test consecutive user messages
    db_messages_consecutive_user = [
        {"sender": "User", "content": "Line 1 {{user}}"},
        {"sender": "User", "content": "Line 2 {{user}}"},
        {"sender": char_name, "content": "Reply from {{char}}"}
    ]
    expected_consecutive_user = [
        ("Line 1 Human", None),
        ("Line 2 Human", "Reply from Botty")
    ]
    assert process_db_messages_to_ui_history(db_messages_consecutive_user, char_name,
                                             user_name) == expected_consecutive_user

    # Test consecutive bot messages
    db_messages_consecutive_bot = [
        {"sender": char_name, "content": "Bot says 1"},
        {"sender": "User", "content": "User confirms"},
        {"sender": char_name, "content": "Bot says 2"},
        {"sender": char_name, "content": "Bot says 3"}
    ]
    expected_consecutive_bot = [
        (None, "Bot says 1"),
        ("User confirms", "Bot says 2"),  # Bot says 2 is paired here
        (None, "Bot says 3")  # Bot says 3 starts a new turn
    ]
    assert process_db_messages_to_ui_history(db_messages_consecutive_bot, char_name,
                                             user_name) == expected_consecutive_bot

    # Test unknown sender
    db_messages_unknown = [
        {"sender": "User", "content": "Question"},
        {"sender": "Narrator", "content": "Action: {{user}} ponders."},
        {"sender": char_name, "content": "Answer from {{char}}"}
    ]
    expected_unknown = [
        ("Question", "[Narrator] Action: Human ponders."),
        (None, "Answer from Botty")
    ]
    assert process_db_messages_to_ui_history(db_messages_unknown, char_name, user_name) == expected_unknown


MINIMAL_V2_DATA_NODE_UNIT = {
    "name": "TestV2", "description": "Desc", "personality": "Pers",
    "scenario": "Scen", "first_mes": "First", "mes_example": "Example"
}
MINIMAL_V2_CARD_UNIT = {"spec": "chara_card_v2", "spec_version": "2.0", "data": MINIMAL_V2_DATA_NODE_UNIT.copy()}
MINIMAL_V1_CARD_UNIT = {
    "name": "TestV1", "description": "Desc", "personality": "Pers",
    "scenario": "Scen", "first_mes": "First", "mes_example": "Example"
}


def test_parse_v2_card_unit():
    # Basic V2
    parsed = parse_v2_card(MINIMAL_V2_CARD_UNIT.copy())
    assert parsed is not None
    assert parsed["name"] == "TestV2"
    assert parsed["first_message"] == "First"
    assert "character_book" not in parsed.get("extensions", {})  # No book in minimal

    # V2 with character_book
    v2_with_book_data = MINIMAL_V2_DATA_NODE_UNIT.copy()
    book_content = {"name": "Lore",
                    "entries": [{"keys": ["key"], "content": "val", "enabled": True, "insertion_order": 0}]}
    v2_with_book_data["character_book"] = book_content
    v2_card_with_book = {"spec": "chara_card_v2", "spec_version": "2.0", "data": v2_with_book_data}

    parsed_with_book = parse_v2_card(v2_card_with_book)
    assert parsed_with_book is not None
    assert "character_book" in parsed_with_book.get("extensions", {})
    # parse_character_book should have been called and its result stored
    assert parsed_with_book["extensions"]["character_book"]["name"] == "Lore"
    assert len(parsed_with_book["extensions"]["character_book"]["entries"]) == 1


def test_parse_v1_card_unit():
    parsed = parse_v1_card(MINIMAL_V1_CARD_UNIT.copy())
    assert parsed is not None and parsed["name"] == "TestV1"
    v1_extra = {**MINIMAL_V1_CARD_UNIT, "custom_field": "custom_val"}
    parsed_extra = parse_v1_card(v1_extra)
    assert parsed_extra["extensions"]["custom_field"] == "custom_val"


def test_parse_character_book_unit():
    # Valid book
    book_data = {"name": "My Lore", "description": "Lore desc", "scan_depth": 10,
                 "entries": [
                     {"keys": ["topic"], "content": "info", "enabled": True, "insertion_order": 1, "name": "Entry1"}]}
    parsed = parse_character_book(book_data)
    assert parsed["name"] == "My Lore"
    assert len(parsed["entries"]) == 1
    assert parsed["entries"][0]["name"] == "Entry1"

    # Book with invalid entry (missing required fields for entry, should be skipped by parser)
    book_data_invalid_entry = {"entries": [{"content": "no keys here"}]}  # Missing keys, enabled, insertion_order
    parsed_invalid = parse_character_book(book_data_invalid_entry)
    assert len(parsed_invalid["entries"]) == 0  # Invalid entry skipped

    # Empty book entries
    parsed_empty = parse_character_book({"entries": []})
    assert parsed_empty["entries"] == []


VALID_BOOK_ENTRY_UNIT = {
    "keys": ["key1"], "content": "Entry content", "enabled": True, "insertion_order": 0
}
VALID_BOOK_UNIT = {"entries": [VALID_BOOK_ENTRY_UNIT.copy()]}


def test_validate_character_book_entry_unit():
    is_valid, errors = validate_character_book_entry(VALID_BOOK_ENTRY_UNIT.copy(), 0, set())
    assert is_valid and not errors
    # Test invalid: missing key
    entry_no_key = VALID_BOOK_ENTRY_UNIT.copy();
    del entry_no_key["keys"]
    is_valid_nk, errors_nk = validate_character_book_entry(entry_no_key, 0, set())
    assert not is_valid_nk and "Missing required field 'keys'" in errors_nk[0]
    # Test invalid position
    entry_bad_pos = {**VALID_BOOK_ENTRY_UNIT, "position": "invalid_pos"}
    is_valid_bp, errors_bp = validate_character_book_entry(entry_bad_pos, 0, set())
    assert not is_valid_bp and "'position' ('invalid_pos') is not a recognized value" in errors_bp[0]


def test_validate_character_book_unit():
    is_valid, errors = validate_character_book(VALID_BOOK_UNIT.copy())
    assert is_valid and not errors
    # Test invalid: entries not a list
    is_valid_nel, errors_nel = validate_character_book({"entries": "not_a_list"})
    assert not is_valid_nel and "must be a list" in errors_nel[0]


def test_validate_v2_card_unit():
    is_valid, errors = validate_v2_card(MINIMAL_V2_CARD_UNIT.copy())
    assert is_valid and not errors
    # Test invalid: missing spec
    card_no_spec = MINIMAL_V2_CARD_UNIT.copy();
    del card_no_spec["spec"]
    is_valid_ns, errors_ns = validate_v2_card(card_no_spec)
    assert not is_valid_ns and "Missing 'spec' field" in errors_ns[0]


@mock.patch(f"{MODULE_PATH_PREFIX}.validate_v2_card")
@mock.patch(f"{MODULE_PATH_PREFIX}.parse_v2_card")
@mock.patch(f"{MODULE_PATH_PREFIX}.parse_v1_card")
def test_import_character_card_from_json_string_unit(mock_parse_v1, mock_parse_v2, mock_validate_v2):
    mock_validate_v2.return_value = (True, [])
    mock_parse_v2.return_value = {"name": "ParsedV2"}
    v2_str = json.dumps(MINIMAL_V2_CARD_UNIT)
    assert import_character_card_from_json_string(v2_str)["name"] == "ParsedV2"

    mock_validate_v2.return_value = (False, ["Not V2"])
    mock_parse_v1.return_value = {"name": "ParsedV1"}
    v1_str = json.dumps(MINIMAL_V1_CARD_UNIT)
    assert import_character_card_from_json_string(v1_str)["name"] == "ParsedV1"


@mock.patch(f"{MODULE_PATH_PREFIX}.import_character_card_from_json_string")
@mock.patch(f"{MODULE_PATH_PREFIX}.yaml")
def test_load_character_card_from_string_content_unit(mock_yaml_module, mock_import_json_str,
                                                      caplog_handler):  # caplog_handler now works
    mock_import_json_str.return_value = {"name": "Loaded"}
    json_content = json.dumps(MINIMAL_V1_CARD_UNIT)
    assert load_character_card_from_string_content(json_content)["name"] == "Loaded"

    yaml_text = "name: YChar\ndescription: D\nfirst_mes: FM\nmes_example: ME\npersonality: P\nscenario: S"
    yaml_front = f"---\n{yaml_text}\n---"

    expected_dict_from_yaml = {
        "name": "YChar", "description": "D", "first_mes": "FM",
        "mes_example": "ME", "personality": "P", "scenario": "S"
    }
    mock_yaml_module.safe_load.return_value = expected_dict_from_yaml
    expected_json_to_import_func = json.dumps(expected_dict_from_yaml)

    load_character_card_from_string_content(yaml_front)
    mock_import_json_str.assert_called_with(expected_json_to_import_func)

    mock_yaml_module.YAMLError = yaml.YAMLError
    mock_yaml_module.safe_load.side_effect = ImportError("No module named 'yaml'")
    with pytest.raises(ImportError, match="No module named 'yaml'"):
        load_character_card_from_string_content(yaml_front)
    mock_yaml_module.safe_load.side_effect = None

    malformed_yaml = "---\nkey: [missing_bracket\n---"
    mock_yaml_module.safe_load.side_effect = yaml.YAMLError("YAML parsing error")
    mock_import_json_str.reset_mock()
    result = load_character_card_from_string_content(malformed_yaml)
    assert result is None
    assert "Error parsing YAML frontmatter" in caplog_handler.text  # Should pass now
    mock_import_json_str.assert_not_called()
    mock_yaml_module.safe_load.side_effect = None


@mock.patch(f"{MODULE_PATH_PREFIX}.Image", new_callable=mock.MagicMock)
@mock.patch(f"{MODULE_PATH_PREFIX}.base64")  # This is the base64 module *used by Character_Chat_Lib*
@mock.patch(f"{MODULE_PATH_PREFIX}.json")    # This mock is 'mock_json_loads_mod'
def test_extract_json_from_image_file_unit(mock_json_loads_mod, mock_base64_mod, MockPILImageModule, tmp_path,
                                           caplog_handler):
    # --- FIX: Configure the mocked json module ---
    # The SUT (Character_Chat_Lib) uses 'json.JSONDecodeError'.
    # Since MODULE_PATH_PREFIX.json is mocked (as mock_json_loads_mod),
    # we need to ensure that mock_json_loads_mod.JSONDecodeError refers to the actual exception class.
    # 'json' here refers to the 'import json' at the top of this test file (the real json module).
    mock_json_loads_mod.JSONDecodeError = json.JSONDecodeError
    # --- END FIX ---

    mock_img_instance = MockPILImageObject()
    MockPILImageModule.open.return_value = mock_img_instance

    chara_json_str = '{"name": "CharaFromImage"}'
    # base64 from 'import base64' (real module) is used for test data setup
    b64_encoded_bytes = base64.b64encode(chara_json_str.encode('utf-8'))
    b64_encoded_str = b64_encoded_bytes.decode('utf-8')

    mock_img_instance.info = {'chara': b64_encoded_str}
    mock_img_instance.format = 'PNG'

    # Default good path return values
    default_b64_return = chara_json_str.encode('utf-8')
    mock_base64_mod.b64decode.return_value = default_b64_return
    # mock_json_loads_mod.loads is the mocked function SUT will call.
    # json.loads(chara_json_str) uses the real 'json' module to prepare the expected return value.
    mock_json_loads_mod.loads.return_value = json.loads(chara_json_str)

    dummy_png_path = tmp_path / "test_unit.png"
    dummy_png_path.write_text("dummy_content_for_file_existence")

    result = extract_json_from_image_file(str(dummy_png_path))
    assert result == chara_json_str # This should be json.loads(result) == json.loads(chara_json_str) if result is JSON string or just comparing strings is fine.
                                      # The function returns a string, so string comparison is correct.
    MockPILImageModule.open.assert_called_once()
    assert isinstance(MockPILImageModule.open.call_args[0][0], io.BytesIO)
    mock_base64_mod.b64decode.assert_called_once_with(b64_encoded_str)
    mock_json_loads_mod.loads.assert_called_once_with(chara_json_str)
    MockPILImageModule.open.reset_mock(); mock_base64_mod.b64decode.reset_mock(); mock_json_loads_mod.loads.reset_mock(); caplog_handler.clear()

    MockPILImageModule.open.reset_mock()
    mock_base64_mod.b64decode.reset_mock()
    mock_json_loads_mod.loads.reset_mock()
    caplog_handler.clear()

    # Test Non-PNG with chara key
    mock_img_instance.format = 'JPEG'
    mock_base64_mod.b64decode.return_value = default_b64_return
    mock_json_loads_mod.loads.return_value = json.loads(chara_json_str)
    assert extract_json_from_image_file(str(dummy_png_path)) == chara_json_str
    assert "not in PNG format" in caplog_handler.text
    caplog_handler.clear()
    mock_img_instance.format = 'PNG'  # Reset format

    # --- Test error in b64decode / subsequent decode ---
    # To test the (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) block

    mock_base64_mod.b64decode.reset_mock()
    mock_base64_mod.b64decode.return_value = b'\xff\xfe\xfd'  # Invalid UTF-8 sequence, will cause .decode('utf-8') to fail
    mock_json_loads_mod.loads.reset_mock()  # Not reached if .decode() fails

    assert extract_json_from_image_file(str(dummy_png_path)) is None
    # This assertion should now pass because the correct logger.error will be called in the SUT
    assert "Error decoding 'chara' metadata" in caplog_handler.text
    mock_base64_mod.b64decode.assert_called_once_with(b64_encoded_str)
    mock_json_loads_mod.loads.assert_not_called()
    mock_base64_mod.b64decode.return_value = default_b64_return; mock_base64_mod.b64decode.side_effect = None; caplog_handler.clear()

    # Reset for next test section
    mock_base64_mod.b64decode.return_value = default_b64_return
    mock_base64_mod.b64decode.side_effect = None
    caplog_handler.clear()

    # Test PIL UnidentifiedImageError
    MockPILImageModule.open.reset_mock()
    MockPILImageModule.open.side_effect = PILImageReal.UnidentifiedImageError("bad image file")
    assert extract_json_from_image_file(str(dummy_png_path)) is None
    assert "Cannot open or read image file" in caplog_handler.text
    MockPILImageModule.open.side_effect = None; MockPILImageModule.open.return_value = mock_img_instance; caplog_handler.clear()


# --- Property Tests (using Hypothesis) ---

# Strategy for names that don't contain placeholders themselves
safe_name_st = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=0, max_size=20
).filter(
    lambda x: not any(p in x for p in ['{{char}}', '{{user}}', '<CHAR>', '<USER>', '{{random_user}}'])
)
optional_safe_name_st = st.one_of(st.none(), safe_name_st)
optional_general_text_st = st.one_of(st.none(), st.text(max_size=100))

@given(text=optional_general_text_st, char_name=optional_safe_name_st, user_name=optional_safe_name_st)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large], deadline=None)
def test_property_replace_placeholders(text, char_name, user_name):
    if text is None:
        assert replace_placeholders(text, char_name, user_name) == ""
        return

    char_name_actual = char_name if char_name is not None else "Character"
    user_name_actual = user_name if user_name is not None else "User"

    result = replace_placeholders(text, char_name, user_name)

    expected = text
    expected = expected.replace('{{char}}', char_name_actual)
    expected = expected.replace('<CHAR>', char_name_actual)
    expected = expected.replace('{{user}}', user_name_actual)
    expected = expected.replace('<USER>', user_name_actual)
    expected = expected.replace('{{random_user}}', user_name_actual)

    assert result == expected

id_st = st.integers(min_value=0, max_value=10**9)
simple_char_name_for_id_test_st = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1, max_size=20
).filter(lambda x: '(' not in x and ')' not in x) # Avoid parens in name to simplify test

@given(id_val=id_st, name=simple_char_name_for_id_test_st,
       id_internal_leading_spaces_count=st.integers(min_value=0, max_value=2),
       id_internal_trailing_spaces_count=st.integers(min_value=0, max_value=2),
       overall_leading_spaces_count=st.integers(min_value=0, max_value=2),
       overall_trailing_spaces_count=st.integers(min_value=0, max_value=2)
       )
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large], deadline=None)
def test_property_extract_character_id_from_ui_choice_valid_formats(
    id_val, name,
    id_internal_leading_spaces_count, id_internal_trailing_spaces_count,
    overall_leading_spaces_count, overall_trailing_spaces_count
):
    id_int_ls = ' ' * id_internal_leading_spaces_count
    id_int_ts = ' ' * id_internal_trailing_spaces_count
    overall_ls = ' ' * overall_leading_spaces_count
    overall_ts = ' ' * overall_trailing_spaces_count

    # Test "Name (ID: <id_val>)" format
    # The SUT's regex r'\(ID:\s*(\d+)\s*\)$' means the ')' must be the last non-whitespace char
    # if this pattern is to be matched by the regex.
    # So, if we add overall_trailing_spaces, the regex match will fail, and it will fall
    # to the choice.strip().isdigit() path, which would also fail for "Name (ID: id)".
    # Thus, for this specific format to be matched by the regex, overall_trailing_spaces must be empty.

    # Case 1: Regex match path " [Name] (ID: [spaces] id [spaces]) "
    # No overall trailing spaces for the regex to match with `$`
    choice1_core = f"{name} (ID:{id_int_ls}{id_val}{id_int_ts})"
    choice1_for_regex = f"{overall_ls}{choice1_core}" # No overall_ts
    assert extract_character_id_from_ui_choice(choice1_for_regex) == id_val

    # Case 2: Just ID path " [spaces] id [spaces] "
    choice2_core = str(id_val)
    choice2_with_overall_spaces = f"{overall_ls}{choice2_core}{overall_ts}"
    assert extract_character_id_from_ui_choice(choice2_with_overall_spaces) == id_val


# @given(invalid_choice=st.text(max_size=50).filter(lambda x:
#     x.strip() != "" and # Must not be empty or all whitespace after strip
#     not re.fullmatch(r'\s*\d+\s*', x) and # Not just a number with spaces
#     not re.search(r'\(ID:\s*\d+\s*\)$', x) # And does not end with the ID pattern
# ))
# @settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large], deadline=None)
# def test_property_extract_character_id_from_ui_choice_invalid_formats(invalid_choice):
#     with pytest.raises(ValueError, match="Invalid choice format"):
#         extract_character_id_from_ui_choice(invalid_choice)

@given(empty_or_whitespace_choice=st.text(alphabet=' \t\n\r', min_size=0, max_size=10))
@settings(suppress_health_check=[HealthCheck.data_too_large, HealthCheck.filter_too_much], deadline=None) # Added deadline
def test_property_extract_character_id_from_ui_choice_empty_or_whitespace(empty_or_whitespace_choice):
    if not empty_or_whitespace_choice:
        with pytest.raises(ValueError, match="No choice provided"):
            extract_character_id_from_ui_choice(empty_or_whitespace_choice)
    elif not empty_or_whitespace_choice.strip(): # Becomes empty after strip
        with pytest.raises(ValueError, match="Invalid choice format"):
            extract_character_id_from_ui_choice(empty_or_whitespace_choice)
    # If it's whitespace but not empty after strip (e.g. " 123 "), it's handled by valid_formats
    # If it's whitespace with non-numeric (e.g. " abc "), it's handled by invalid_formats


# For parse_v1_card
v1_required_fields_st = st.fixed_dictionaries({
    "name": st.text(min_size=1, max_size=50),
    "description": st.text(max_size=100),
    "personality": st.text(max_size=100),
    "scenario": st.text(max_size=100),
    "first_mes": st.text(min_size=1, max_size=100),
    "mes_example": st.text(max_size=100),
})
v1_optional_fields_st = st.fixed_dictionaries({
    "creator_notes": st.text(max_size=100), "system_prompt": st.text(max_size=100),
    "post_history_instructions": st.text(max_size=100),
    "alternate_greetings": st.lists(st.text(max_size=50), max_size=3),
    "tags": st.lists(st.text(max_size=20), max_size=5),
    "creator": st.text(max_size=30), "character_version": st.text(max_size=10),
    "char_image": st.one_of(st.none(), st.text(max_size=50)), # Can be None, empty string, or text
    "image": st.one_of(st.none(), st.text(max_size=50))      # Can be None, empty string, or text
})
known_v1_keys = {
    "name", "description", "personality", "scenario", "first_mes", "mes_example",
    "creator_notes", "system_prompt", "post_history_instructions",
    "alternate_greetings", "tags", "creator", "character_version", "char_image", "image"
}
extension_key_st = st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)).filter(lambda k: k not in known_v1_keys)
extension_value_st = st.one_of(st.text(max_size=50), st.integers(), st.booleans(), st.none())
extensions_st = st.dictionaries(extension_key_st, extension_value_st, max_size=3)

@given(required_data=v1_required_fields_st, optional_data=v1_optional_fields_st, extensions=extensions_st)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large], deadline=None)
def test_property_parse_v1_card_structure_and_extensions(required_data, optional_data, extensions):
    v1_card_data = {**required_data, **optional_data, **extensions}
    parsed = parse_v1_card(v1_card_data)

    assert parsed is not None
    assert parsed['name'] == required_data['name']
    assert parsed['first_message'] == required_data['first_mes']
    assert parsed['creator_notes'] == optional_data.get('creator_notes', '')
    assert parsed['alternate_greetings'] == optional_data.get('alternate_greetings', [])

    # Correctly test image_base64 based on SUT's `val1 or val2` logic
    expected_image_base64 = v1_card_data.get('char_image') or v1_card_data.get('image')
    assert parsed['image_base64'] == expected_image_base64

    parsed_extensions = parsed.get('extensions', {})
    for key, value in extensions.items():
        assert key in parsed_extensions and parsed_extensions[key] == value
    # Ensure standard keys (that were part of required_data or optional_data) are not in extensions
    for key in (set(required_data.keys()) | set(optional_data.keys())):
        if key not in extensions: # Unless it was *also* an extension key (unlikely with filter)
             assert key not in parsed_extensions


@given(base_card=v1_required_fields_st)
@settings(suppress_health_check=[HealthCheck.data_too_large], deadline=None)
def test_property_parse_v1_card_missing_required_fields(base_card):
    import random # Keep import local if only used here
    card_with_missing_field = base_card.copy()
    required_keys_list = ["name", "description", "personality", "scenario", "first_mes", "mes_example"]
    field_to_remove = random.choice(required_keys_list)
    del card_with_missing_field[field_to_remove]
    with pytest.raises(ValueError, match=f"Missing required field in V1 card: {field_to_remove}"):
        parse_v1_card(card_with_missing_field)

# For process_db_messages_to_ui_history
db_message_content_st = st.text(max_size=30)

@given(
    user_messages_content=st.lists(db_message_content_st, min_size=0, max_size=3),
    char_name=safe_name_st.filter(lambda x: x != "User" and x != "" and x is not None),
    user_name=optional_safe_name_st
)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large], deadline=None)
def test_property_process_db_messages_to_ui_history_only_user(user_messages_content, char_name, user_name):
    db_messages = [{"sender": "User", "content": content} for content in user_messages_content]
    user_name_actual = user_name if user_name is not None else "User"
    char_name_actual = char_name # Already filtered to be non-None, non-empty
    processed_history = process_db_messages_to_ui_history(db_messages, char_name_actual, user_name_actual)

    assert len(processed_history) == len(user_messages_content)
    for i, original_content in enumerate(user_messages_content):
        expected_processed_content = replace_placeholders(original_content, char_name_actual, user_name_actual)
        assert processed_history[i] == (expected_processed_content, None)

@given(
    char_messages_content=st.lists(db_message_content_st, min_size=0, max_size=3),
    char_name=safe_name_st.filter(lambda x: x != "User" and x != "" and x is not None),
    user_name=optional_safe_name_st
)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large], deadline=None)
def test_property_process_db_messages_to_ui_history_only_char(char_messages_content, char_name, user_name):
    char_name_actual = char_name # Already filtered
    db_messages = [{"sender": char_name_actual, "content": content} for content in char_messages_content]
    user_name_actual = user_name if user_name is not None else "User"

    processed_history = process_db_messages_to_ui_history(
        db_messages, char_name_actual, user_name_actual, actual_char_sender_id_in_db=char_name_actual
    )
    assert len(processed_history) == len(char_messages_content)
    for i, original_content in enumerate(char_messages_content):
        expected_processed_content = replace_placeholders(original_content, char_name_actual, user_name_actual)
        assert processed_history[i] == (None, expected_processed_content)

@given(
    message_pairs_content=st.lists(st.tuples(db_message_content_st, db_message_content_st), min_size=0, max_size=2),
    char_name=safe_name_st.filter(lambda x: x != "User" and x != "" and x is not None),
    user_name=optional_safe_name_st
)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large], deadline=None)
def test_property_process_db_messages_to_ui_history_alternating(message_pairs_content, char_name, user_name):
    char_name_actual = char_name # Already filtered
    db_messages = []
    for user_content, char_content_for_pair in message_pairs_content: # Renamed to avoid clash
        db_messages.append({"sender": "User", "content": user_content})
        db_messages.append({"sender": char_name_actual, "content": char_content_for_pair})
    user_name_actual = user_name if user_name is not None else "User"

    processed_history = process_db_messages_to_ui_history(
        db_messages, char_name_actual, user_name_actual, actual_char_sender_id_in_db=char_name_actual
    )
    assert len(processed_history) == len(message_pairs_content)
    for i, (orig_user_c, orig_char_c) in enumerate(message_pairs_content):
        exp_user_c = replace_placeholders(orig_user_c, char_name_actual, user_name_actual)
        exp_char_c = replace_placeholders(orig_char_c, char_name_actual, user_name_actual)
        assert processed_history[i] == (exp_user_c, exp_char_c)

# --- End of Property Tests ---


# --- Integration Tests (using the 'db' fixture) ---


@pytest.mark.integration
def test_get_character_list_for_ui_integration(db):
    char1_id = db.add_character_card({"name": "Charlie", "description": "C"})
    char2_id = db.add_character_card({"name": "Alice", "description": "A"})

    # Original problematic lines:
    # char3_data = db.get_character_card_by_name(db.add_character_card({"name": "Bob (deleted)", "description": "B_del"}))
    # db.soft_delete_character_card(char3_data['id'], char3_data['version'])

    # Corrected logic:
    char3_name_to_add = "Bob (deleted)"
    db.add_character_card({"name": char3_name_to_add, "description": "B_del"})  # Add the character
    char3_data_for_delete = db.get_character_card_by_name(char3_name_to_add)  # Fetch it by name

    assert char3_data_for_delete is not None, f"Character '{char3_name_to_add}' should have been found after adding."
    db.soft_delete_character_card(char3_data_for_delete['id'], char3_data_for_delete['version'])

    ui_list = get_character_list_for_ui(db, limit=10)
    # The list should be sorted by name: Alice, Charlie
    assert len(ui_list) == 2
    assert ui_list[0]["name"] == "Alice"
    assert ui_list[1]["name"] == "Charlie"


@pytest.mark.integration
@mock.patch(f"{MODULE_PATH_PREFIX}.Image", new_callable=mock.MagicMock)  # Patches PIL.Image used in Character_Chat_Lib
def test_load_character_and_image_integration(MockPILImageModule, db, caplog_handler):
    mock_opened_image = MockPILImageObject(format="PNG")  # This is what Image.open() will return
    mock_converted_image = MockPILImageObject(format="PNG", mode="RGBA")  # This is what .convert() will return

    # Configure the mock chain: Image.open().convert()
    MockPILImageModule.open.return_value = mock_opened_image
    mock_opened_image.convert = mock.Mock(
        return_value=mock_converted_image)  # Mock the convert method on the opened instance

    image_bytes = create_dummy_png_bytes()
    char_id = db.add_character_card(
        {"name": "Gandalf", "description": "W {{user}}", "first_message": "FM {{char}} {{user}}", "image": image_bytes})

    loaded_char, hist, img = load_character_and_image(db, char_id, "Frodo")

    assert loaded_char["name"] == "Gandalf"
    assert hist == [(None, "FM Gandalf Frodo")]
    # Check that the image returned is the one from .convert()
    assert img == mock_converted_image
    mock_opened_image.convert.assert_called_once_with("RGBA")  # Verify convert was called correctly

    # Test image processing error
    # Make Image.open() itself raise the error for this part of the test
    MockPILImageModule.open.side_effect = PILImageReal.UnidentifiedImageError("bad image")
    mock_opened_image.convert.reset_mock()  # Reset convert mock call count

    loaded_char_bad_img, _, img_bad = load_character_and_image(db, char_id, "Frodo")
    assert loaded_char_bad_img is not None  # Char data should still load
    assert img_bad is None
    assert f"Error processing image for character 'Gandalf' (ID: {char_id})" in caplog_handler.text  # This will use the new caplog_handler

    MockPILImageModule.open.side_effect = None  # Reset side effect
    MockPILImageModule.open.return_value = mock_opened_image  # Reset return value for other tests


@pytest.mark.integration
@mock.patch(f"{MODULE_PATH_PREFIX}.yaml")
@mock.patch(f"{MODULE_PATH_PREFIX}.Image", new_callable=mock.MagicMock)
def test_import_and_save_character_from_file_integration(MockPILImageModule, mock_yaml_module, db, tmp_path):
    # JSON file
    v1_content = {"name": "JSON Char", "description": "D", "first_mes": "FM", "mes_example": "ME", "personality": "P",
                  "scenario": "S"}
    json_file = tmp_path / "import.json";
    json_file.write_text(json.dumps(v1_content))
    char_id_json = import_and_save_character_from_file(db, str(json_file))
    assert db.get_character_card_by_id(char_id_json)["name"] == "JSON Char"

    # PNG with chara
    mock_img_instance = MockPILImageObject()  # Our custom mock
    MockPILImageModule.open.return_value = mock_img_instance
    png_chara_data = {"name": "PNG Chara", "first_mes": "FMpng"}
    png_chara_json_str = json.dumps({**MINIMAL_V1_CARD_UNIT, **png_chara_data})  # Ensure all V1 fields for parser
    b64_encoded = base64.b64encode(png_chara_json_str.encode('utf-8')).decode('utf-8')
    mock_img_instance.info = {'chara': b64_encoded};
    mock_img_instance.format = 'PNG'
    dummy_png_bytes = create_dummy_png_bytes(png_chara_json_str)
    png_file = tmp_path / "chara.png";
    png_file.write_bytes(dummy_png_bytes)

    char_id_png = import_and_save_character_from_file(db, str(png_file))
    retrieved_png = db.get_character_card_by_id(char_id_png)
    assert retrieved_png["name"] == "PNG Chara" and retrieved_png["image"] == dummy_png_bytes


@pytest.mark.integration
@mock.patch(f"{MODULE_PATH_PREFIX}.time.strftime", return_value=MOCK_TIME_STRFTIME)
def test_load_chat_history_from_file_and_save_to_db_integration(mock_strftime, db, tmp_path,
                                                                caplog_handler):  # caplog_handler now works
    char_name_in_db = "HistCharDB"  # Renamed to avoid clash with `char_name` variable if any
    char_id_db = db.add_character_card({"name": char_name_in_db, "description": "D"})
    log_user = "LogU"

    chat_data = {
        "char_name": char_name_in_db,
        "user_name": log_user,
        "history": {
            "internal": [
                ["U: {{user}}", "C: {{char}}"],
                "not a list",
                ["User only"],
                ["Msg1", "Msg2", "Msg3"],
                [None, None]
            ]
        }
    }
    hist_file_path = tmp_path / "hist.json"; hist_file_path.write_text(json.dumps(chat_data))
    conv_id, char_id_hist = load_chat_history_from_file_and_save_to_db(db, str(hist_file_path), user_name_for_placeholders=log_user)
    assert conv_id is not None and char_id_hist == char_id_db
    msgs = db.get_messages_for_conversation(conv_id)
    assert len(msgs) == 3
    assert "Skipping malformed message pair" in caplog_handler.text  # Should pass now

    assert msgs[0]["content"] == f"U: {log_user}"
    assert msgs[0]["sender"] == "User"
    assert msgs[1]["content"] == f"C: {char_name_in_db}"
    assert msgs[1]["sender"] == char_name_in_db
    assert msgs[2]["content"] == "User only"
    assert msgs[2]["sender"] == "User"


@pytest.mark.integration
@mock.patch(f"{MODULE_PATH_PREFIX}.time.strftime", return_value=MOCK_TIME_STRFTIME)
@mock.patch(f"{MODULE_PATH_PREFIX}.Image", new_callable=mock.MagicMock)
def test_full_chat_session_flow_integration(MockPILImageModule, mock_strftime, db):
    mock_img_instance = MockPILImageObject()
    MockPILImageModule.open.return_value = mock_img_instance

    char_id = db.add_character_card(
        {"name": "FlowChar", "first_message": "Hi {{char}}", "image": create_dummy_png_bytes()})
    user_name = "FlowUser"
    conv_id, _, init_hist, _ = start_new_chat_session(db, char_id, user_name, "FlowTitle")
    assert conv_id and init_hist == [(None, "Hi FlowChar")]

    user_msg_id = post_message_to_conversation(db, conv_id, "FlowChar", "User says {{user}}", True)
    char_resp_id = post_message_to_conversation(db, conv_id, "FlowChar", "Char says {{char}}", False)

    ui_hist = retrieve_conversation_messages_for_ui(db, conv_id, "FlowChar", user_name)
    assert ui_hist == [(None, "Hi FlowChar"), ("User says FlowUser", "Char says FlowChar")]

    msg_to_edit = db.get_message_by_id(user_msg_id)
    assert edit_message_content(db, user_msg_id, "Edited", msg_to_edit['version'])

    msg_to_rank = db.get_message_by_id(char_resp_id)
    assert set_message_ranking(db, char_resp_id, 3, msg_to_rank['version'])
    msg_to_remove = db.get_message_by_id(user_msg_id) # Re-fetch after edit for new version
    assert remove_message_from_conversation(db, user_msg_id, msg_to_remove['version'])

    conv_meta = db.get_conversation_by_id(conv_id)
    assert update_conversation_metadata(db, conv_id, {"title": "NewTitle"}, conv_meta['version'])

    assert len(search_conversations_by_title_query(db, "NewTitle")) == 1

    found_msgs = find_messages_in_conversation(db, conv_id, "Char says", "FlowChar", user_name)
    assert len(found_msgs) == 1 and found_msgs[0]["content"] == "Char says FlowChar"
    conv_to_del = db.get_conversation_by_id(conv_id) # Re-fetch after update for new version
    assert delete_conversation_by_id(db, conv_id, conv_to_del['version'])
    assert db.get_conversation_by_id(conv_id) is None


@pytest.mark.integration
def test_load_chat_and_character_integration(db):
    user_name = "Loader"
    char_id = db.add_character_card({"name": "LoadChar", "first_message": "FM"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Conv1"})
    db.add_message({"conversation_id": conv_id, "sender": "User", "content": "Hi {{char}}"})
    db.add_message({"conversation_id": conv_id, "sender": "LoadChar", "content": "Yo {{user}}"})

    with mock.patch(f"{MODULE_PATH_PREFIX}.Image", new_callable=mock.MagicMock) as MockPILImageModule:
        MockPILImageModule.open.return_value = MockPILImageObject()
        char_data, history, _ = load_chat_and_character(db, conv_id, user_name)
        assert char_data["name"] == "LoadChar" and history == [("Hi LoadChar", "Yo Loader")]


@pytest.mark.integration
def test_load_character_wrapper_integration(db):
    char_id = db.add_character_card({"name": "Wrap", "first_message": "FM {{user}}"})
    user_name = "Wrapper"
    with mock.patch(f"{MODULE_PATH_PREFIX}.Image", new_callable=mock.MagicMock):
        _, hist_int, _ = load_character_wrapper(db, char_id, user_name)
        assert hist_int == [(None, "FM Wrapper")]
        _, hist_str, _ = load_character_wrapper(db, f"Wrap (ID: {char_id})", user_name)
        assert hist_str == [(None, "FM Wrapper")]


@pytest.mark.integration
def test_get_conversation_metadata_integration(db):
    char_id = db.add_character_card({"name": "MetaChar", "description": "D"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "MetaTitle", "rating": 4})
    meta = get_conversation_metadata(db, conv_id)
    assert meta and meta["title"] == "MetaTitle" and meta["rating"] == 4
    assert get_conversation_metadata(db, "non_existent_conv") is None


@pytest.mark.integration
def test_retrieve_message_details_integration(db):
    char_id = db.add_character_card({"name": "MsgDetChar", "description": "D"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "MsgDetConv"})
    msg_id = db.add_message({"conversation_id": conv_id, "sender": "User", "content": "Test {{char}} from {{user}}"})
    details = retrieve_message_details(db, msg_id, "MsgDetChar", "TestUser")
    assert details and details["content"] == "Test MsgDetChar from TestUser"
    assert retrieve_message_details(db, "non_existent_msg", "Char", "User") is None

#
# End of test_character_chat_lib.py
########################################################################################################################

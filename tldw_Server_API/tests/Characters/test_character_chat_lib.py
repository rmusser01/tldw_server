# test_character_chat_lib.py
#
#
# --- Imports ---
import logging

import pytest
from unittest import mock
import base64
import binascii
import io
import json
import os
import time

import yaml
from PIL import Image as PILImageReal
#
# Third Party Imports
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
    def __init__(self, format="PNG", info=None, width=100, height=100):
        self.format = format
        self.info = info if info is not None else {}
        self.width = width
        self.height = height
        self.fp = None  # To mimic file pointer handling for close
        self.mode = "RGBA"  # Default mode after convert

    def convert(self, mode):
        new_mock = MockPILImageObject(format=self.format, info=self.info.copy(), width=self.width, height=self.height)
        new_mock.mode = mode
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
    """Fixture to correctly set up loguru to work with pytest's caplog."""
    # This assumes loguru is the logger used by Character_Chat_Lib.py
    # If Character_Chat_Lib.py directly uses standard logging, this might not be needed
    # or might need adjustment based on how loguru is configured in the library.
    # For now, we'll assume the library's logger is the root logger or a child.
    # Pytest's caplog fixture captures standard logging.
    # If loguru is configured to propagate to root, caplog should get it.
    # If Character_Chat_Lib specifically uses its own loguru instance, direct loguru capture needed.
    # For simplicity, let's assume standard logging capture works or library uses standard logging.
    # No special loguru setup needed here if it propagates.
    # If Character_Chat_Lib.logger is a specific loguru instance:
    # from loguru import logger as loguru_logger
    # handler_id = loguru_logger.add(your_capture_sink_function, level="DEBUG")
    # yield caplog
    # loguru_logger.remove(handler_id)
    # For now, relying on pytest's default caplog with standard logging.
    caplog.set_level(logging.DEBUG)  # Capture from DEBUG level upwards
    return caplog


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
def test_load_character_card_from_string_content_unit(mock_yaml_module, mock_import_json_str, caplog_handler):
    mock_import_json_str.return_value = {"name": "Loaded"}
    json_content = json.dumps(MINIMAL_V1_CARD_UNIT)
    assert load_character_card_from_string_content(json_content)["name"] == "Loaded"

    yaml_text = "name: YChar\nd: D\nfm: F\nme: ME\np: P\ns: S"  # simplified for this test
    yaml_front = f"---\n{yaml_text}\n---"
    mock_yaml_module.safe_load.return_value = {"name": "YChar", "d": "D", "fm": "F", "me": "ME", "p": "P", "s": "S"}
    expected_json = json.dumps(mock_yaml_module.safe_load.return_value)
    load_character_card_from_string_content(yaml_front)
    mock_import_json_str.assert_called_with(expected_json)

    # Test missing PyYAML
    mock_yaml_module.YAMLError = Exception  # To simulate yaml.YAMLError if yaml itself is mocked
    mock_yaml_module.safe_load.side_effect = ImportError("No module named 'yaml'")
    with pytest.raises(ImportError, match="No module named 'yaml'"):
        load_character_card_from_string_content(yaml_front)
    mock_yaml_module.safe_load.side_effect = None  # Reset

    # Test malformed YAML frontmatter (should fall through and potentially log error)
    malformed_yaml = "---\nkey: [missing_bracket\n---"
    mock_yaml_module.safe_load.side_effect = yaml.YAMLError("YAML parsing error")
    # It should then try to find JSON block, or fail. Assuming no JSON block:
    mock_import_json_str.reset_mock()
    result = load_character_card_from_string_content(malformed_yaml)
    assert result is None  # Fails to find any valid structure
    assert "Error parsing YAML frontmatter" in caplog_handler.text
    mock_yaml_module.safe_load.side_effect = None


@mock.patch(f"{MODULE_PATH_PREFIX}.Image", new_callable=mock.MagicMock)
@mock.patch(f"{MODULE_PATH_PREFIX}.base64")
@mock.patch(f"{MODULE_PATH_PREFIX}.json")
def test_extract_json_from_image_file_unit(mock_json_loads_mod, mock_base64_mod, MockPILImageModule, tmp_path,
                                           caplog_handler):
    mock_img_instance = MockPILImageObject()
    MockPILImageModule.open.return_value = mock_img_instance

    chara_json_str = '{"name": "CharaFromImage"}'
    b64_encoded = base64.b64encode(chara_json_str.encode('utf-8')).decode('utf-8')
    mock_img_instance.info = {'chara': b64_encoded};
    mock_img_instance.format = 'PNG'
    mock_base64_mod.b64decode.return_value = chara_json_str.encode('utf-8')
    mock_json_loads_mod.loads.return_value = json.loads(chara_json_str)

    # File path
    dummy_png_path = tmp_path / "test_unit.png";
    dummy_png_path.write_text("dummy_content")
    assert extract_json_from_image_file(str(dummy_png_path)) == chara_json_str

    # Non-PNG with chara key (should log warning but still process)
    mock_img_instance.format = 'JPEG'
    assert extract_json_from_image_file(str(dummy_png_path)) == chara_json_str
    assert "not in PNG format" in caplog_handler.text
    caplog_handler.clear()

    # Error in b64decode
    mock_base64_mod.b64decode.side_effect = binascii.Error("bad b64")
    assert extract_json_from_image_file(str(dummy_png_path)) is None
    assert "Error decoding 'chara' metadata" in caplog_handler.text
    mock_base64_mod.b64decode.side_effect = None;
    caplog_handler.clear()

    # PIL UnidentifiedImageError
    MockPILImageModule.open.side_effect = PILImageReal.UnidentifiedImageError("bad image file")
    assert extract_json_from_image_file(str(dummy_png_path)) is None
    assert "Cannot open or read image file" in caplog_handler.text
    MockPILImageModule.open.side_effect = None;
    MockPILImageModule.open.return_value = mock_img_instance;
    caplog_handler.clear()


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
@mock.patch(f"{MODULE_PATH_PREFIX}.Image", new_callable=mock.MagicMock)
def test_load_character_and_image_integration(MockPILImageModule, db, caplog_handler):
    mock_img_instance = MockPILImageObject()  # Using our own mock for better control
    MockPILImageModule.open.return_value = mock_img_instance

    image_bytes = create_dummy_png_bytes()
    char_id = db.add_character_card(
        {"name": "Gandalf", "description": "W {{user}}", "first_message": "FM {{char}} {{user}}", "image": image_bytes})
    loaded_char, hist, img = load_character_and_image(db, char_id, "Frodo")
    assert loaded_char["name"] == "Gandalf" and hist == [(None, "FM Gandalf Frodo")] and img == mock_img_instance

    # Test image processing error
    MockPILImageModule.open.side_effect = PILImageReal.UnidentifiedImageError("bad image")
    loaded_char_bad_img, _, img_bad = load_character_and_image(db, char_id, "Frodo")
    assert loaded_char_bad_img is not None  # Char data should still load
    assert img_bad is None
    assert f"Error processing image for character 'Gandalf' (ID: {char_id})" in caplog_handler.text
    MockPILImageModule.open.side_effect = None;
    MockPILImageModule.open.return_value = mock_img_instance  # Reset


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
def test_load_chat_history_from_file_and_save_to_db_integration(mock_strftime, db, tmp_path, caplog_handler):
    char_name = "HistChar";
    char_id_db = db.add_character_card({"name": char_name, "description": "D"})
    log_user = "LogU"
    # Valid and malformed pairs
    chat_data = {"char_name": char_name, "user_name": log_user, "history": {"internal": [
        ["U: {{user}}", "C: {{char}}"],  # Valid
        "not a list",  # Malformed string pair
        ["User only"],  # Valid (becomes User msg, None Bot msg)
        ["Msg1", "Msg2", "Msg3"],  # Malformed (too long)
        [None, None]  # Skipped (empty)
    ]}}
    hist_file = tmp_path / "hist.json";
    hist_file.write_text(json.dumps(chat_data))

    conv_id, char_id_hist = load_chat_history_from_file_and_save_to_db(db, str(hist_file),
                                                                       user_name_for_placeholders=log_user)
    assert conv_id and char_id_hist == char_id_db
    msgs = db.get_messages_for_conversation(conv_id)
    assert len(msgs) == 3  # (U,C) pair -> 2 msgs. ("User only", None) -> 1 msg.
    assert "Skipping malformed message pair" in caplog_handler.text  # For "not a list" and too long pair
    assert msgs[0]["content"] == f"U: {log_user}"
    assert msgs[1]["content"] == f"C: {char_name}"
    assert msgs[2]["content"] == "User only"


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

    msg_to_remove = db.get_message_by_id(user_msg_id)
    assert remove_message_from_conversation(db, user_msg_id, msg_to_remove['version'])

    conv_meta = db.get_conversation_by_id(conv_id)
    assert update_conversation_metadata(db, conv_id, {"title": "NewTitle"}, conv_meta['version'])

    assert len(search_conversations_by_title_query(db, "NewTitle")) == 1

    found_msgs = find_messages_in_conversation(db, conv_id, "Char says", "FlowChar", user_name)
    assert len(found_msgs) == 1 and found_msgs[0]["content"] == "Char says FlowChar"

    conv_to_del = db.get_conversation_by_id(conv_id)
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

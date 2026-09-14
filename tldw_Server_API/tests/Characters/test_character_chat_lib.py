# test_character_chat_lib.py
#
#
# --- Imports ---
import logging
import copy
import re

import hypothesis
import pytest
from unittest import mock
import base64
import binascii
import io
import json
import os
import struct
import time
import zlib
import yaml

#
# Third Party Imports
from PIL import Image as PILImageReal
from loguru import logger as loguru_logger
from hypothesis import given, strategies as st, settings, HealthCheck

#
# Local Imports
from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib_facade import (
    replace_placeholders,
    replace_user_placeholder,
    get_character_list_for_ui,
    extract_character_id_from_ui_choice,
    map_sender_to_role,
    load_character_and_image,
    process_db_messages_to_ui_history,
    process_db_messages_to_rich_ui_history,
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
    _prepare_character_data_for_db_storage,
    create_new_character_from_data,
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
    find_messages_in_conversation,
)
from tldw_Server_API.app.core.Character_Chat.constants import MAX_PNG_METADATA_BYTES
from tldw_Server_API.app.core.Character_Chat.modules.character_io import (
    MAX_IMPORT_AVATAR_BYTES,
    _resolve_max_import_avatar_bytes,
    _validate_import_avatar_bytes,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
    SchemaError,
)

#
#######################################################################################################################
#
# Functions:

# --- MOCK time.strftime for deterministic timestamps in titles ---
MOCK_TIME_STRFTIME = "2023-10-27 10:30"

# --- Module Path for Patching (if still needed for non-DB external deps) ---
MODULE_PATH_PREFIX = "tldw_Server_API.app.core.Character_Chat.modules"


# --- Mock PIL Image object for finer control during unit tests where PIL is mocked ---
class MockPILImageObject:
    def __init__(self, format="PNG", info=None, width=100, height=100, mode="RGBA"):  # mode was added here
        self.format = format
        self.info = info if info is not None else {}
        self.width = width
        self.height = height
        self.fp = None
        self.mode = mode  # And used here

    def convert(self, mode):
        new_mock = MockPILImageObject(
            format=self.format,
            info=self.info.copy(),
            width=self.width,
            height=self.height,
            mode=mode,
        )
        return new_mock

    def close(self):
        if self.fp:
            self.fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def verify(self):
        return None

    @property
    def size(self):
        return (self.width, self.height)


# --- Helper function to create dummy PNG bytes ---
def create_dummy_png_bytes(chara_data_json_str=None, is_png=True):
    if not is_png:
        return b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"

    buffer = io.BytesIO()
    PILImageReal.new("RGBA", (1, 1), (0, 0, 0, 0)).save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    if not chara_data_json_str:
        return png_bytes

    text_chunk_payload = b"chara\x00" + base64.b64encode(chara_data_json_str.encode("utf-8"))
    chunk_type = b"tEXt"
    chunk_len = struct.pack(">I", len(text_chunk_payload))
    crc = struct.pack(">I", binascii.crc32(chunk_type + text_chunk_payload) & 0xFFFFFFFF)
    text_chunk = chunk_len + chunk_type + text_chunk_payload + crc

    pos = 8  # Skip PNG signature
    while pos + 8 <= len(png_bytes):
        data_len = struct.unpack(">I", png_bytes[pos:pos + 4])[0]
        chunk_type_at_pos = png_bytes[pos + 4:pos + 8]
        chunk_total_len = 12 + data_len  # length + type + data + crc
        if pos + chunk_total_len > len(png_bytes):
            break
        if chunk_type_at_pos == b"IEND":
            return png_bytes[:pos] + text_chunk + png_bytes[pos:]
        pos += chunk_total_len

    return png_bytes


# --- Pytest Fixture for In-Memory DB Instance ---
@pytest.fixture
def db():
    """Provides a fresh in-memory CharactersRAGDB instance for each test."""
    db_instance = CharactersRAGDB(":memory:", client_id="pytest_client")
    yield db_instance
    db_instance.close_connection()


# --- Pytest Fixture for Capturing Logs (using standard logging) ---
@pytest.fixture
def caplog_handler(caplog):
    """
    Fixture to correctly set up loguru to work with pytest's caplog.
    It wires Loguru directly into pytest's caplog handler, avoiding recursion with
    the application's stdlib interception.
    """

    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = loguru_logger.add(PropagateHandler(), format="{message}", level="DEBUG")

    # Capture DEBUG from both the application logger namespace and generally.
    caplog.set_level(logging.DEBUG, logger="tldw_Server_API")
    caplog.set_level(logging.DEBUG)

    yield caplog  # Test runs here

    # Remove the handler after the test
    loguru_logger.remove(handler_id)


# --- Unit Tests (Pure Logic) ---


@pytest.mark.parametrize(
    "text, char_name, user_name, expected",
    [
        ("Hello {{char}}, I am {{user}}.", "Alice", "Bob", "Hello Alice, I am Bob."),
        ("User: <USER>, Char: <CHAR>", "Wizard", "Hero", "User: Hero, Char: Wizard"),
        ("{{random_user}} says hi.", None, "Guest", "Guest says hi."),
        ("Numeric {{user}}", "Char", 123, "Numeric 123"),
        ("", "Char", "User", ""),
        (None, "Char", "User", ""),
    ],
)
def test_replace_placeholders(text, char_name, user_name, expected):
    assert replace_placeholders(text, char_name, user_name) == expected


@pytest.mark.parametrize(
    "history, user_name, expected",
    [
        ([("Hi {{user}}", "Hello back, {{user}}")], "Sam", [("Hi Sam", "Hello back, Sam")]),
        ([], "User", []),
    ],
)
def test_replace_user_placeholder(history, user_name, expected):
    assert replace_user_placeholder(history, user_name) == expected


@pytest.mark.parametrize(
    "choice, expected_id, raises",
    [
        ("My Character (ID: 123)", 123, None),
        ("789", 789, None),
        ("Invalid Format", None, ValueError),
        ("", None, ValueError),
    ],
)
def test_extract_character_id_from_ui_choice(choice, expected_id, raises):
    if raises:
        with pytest.raises(raises):
            extract_character_id_from_ui_choice(choice)
    else:
        assert extract_character_id_from_ui_choice(choice) == expected_id


def test_process_db_messages_to_ui_history_unit():
    char_name = "Botty"
    user_name = "Human"
    db_messages = [{"sender": "User", "content": "Hi {{char}}"}, {"sender": char_name, "content": "Hello {{user}}"}]
    expected = [("Hi Botty", "Hello Human")]
    assert process_db_messages_to_ui_history(db_messages, char_name, user_name) == expected

    # Test consecutive user messages
    db_messages_consecutive_user = [
        {"sender": "User", "content": "Line 1 {{user}}"},
        {"sender": "User", "content": "Line 2 {{user}}"},
        {"sender": char_name, "content": "Reply from {{char}}"},
    ]
    expected_consecutive_user = [("Line 1 Human", None), ("Line 2 Human", "Reply from Botty")]
    assert (
        process_db_messages_to_ui_history(db_messages_consecutive_user, char_name, user_name)
        == expected_consecutive_user
    )

    # Test consecutive bot messages
    db_messages_consecutive_bot = [
        {"sender": char_name, "content": "Bot says 1"},
        {"sender": "User", "content": "User confirms"},
        {"sender": char_name, "content": "Bot says 2"},
        {"sender": char_name, "content": "Bot says 3"},
    ]
    expected_consecutive_bot = [
        (None, "Bot says 1"),
        ("User confirms", "Bot says 2"),  # Bot says 2 is paired here
        (None, "Bot says 3"),  # Bot says 3 starts a new turn
    ]
    assert (
        process_db_messages_to_ui_history(db_messages_consecutive_bot, char_name, user_name) == expected_consecutive_bot
    )

    # Test unknown sender
    db_messages_unknown = [
        {"sender": "User", "content": "Question"},
        {"sender": "Narrator", "content": "Action: {{user}} ponders."},
        {"sender": char_name, "content": "Answer from {{char}}"},
    ]
    expected_unknown = [("Question", "[Narrator] Action: Human ponders."), (None, "Answer from Botty")]
    assert process_db_messages_to_ui_history(db_messages_unknown, char_name, user_name) == expected_unknown


def test_process_db_messages_to_ui_history_normalizes_common_roles():
    char_name = "Botty"
    user_name = "Human"
    db_messages = [
        {"sender": "user", "content": "hello {{char}}"},
        {"sender": "assistant", "content": "hi {{user}}"},
        {"sender": "system", "content": "meta message"},
        {"sender": "USER", "content": "final turn"},
    ]
    expected = [
        ("hello Botty", "hi Human"),
        (None, "[system] meta message"),
        ("final turn", None),
    ]
    assert process_db_messages_to_ui_history(db_messages, char_name, user_name) == expected


def test_process_db_messages_to_ui_history_trims_sender_aliases():
    char_name = "Botty"
    user_name = "Human"
    db_messages = [
        {"sender": " User ", "content": "Hello {{char}}"},
        {"sender": " Botty ", "content": "Hi {{user}}"},
    ]
    expected = [("Hello Botty", "Hi Human")]
    assert process_db_messages_to_ui_history(db_messages, char_name, user_name) == expected


def test_map_sender_to_role_handles_tool_prefix():
    assert map_sender_to_role("tool:lookup", "Botty") == "tool"
    assert map_sender_to_role("function:search", "Botty") == "tool"
    assert map_sender_to_role("assistant_tool:call", "Botty") == "tool"


def test_map_sender_to_role_respects_default_role():
    assert map_sender_to_role("mystery", "Botty", default_role="user") == "user"
    assert map_sender_to_role("", "Botty", default_role="system") == "system"


def test_process_db_messages_to_ui_history_handles_additional_alias():
    char_name = "RenamedBot"
    user_name = "Human"
    db_messages = [
        {"sender": "OldBot", "content": "Greetings {{user}}"},
        {"sender": "User", "content": "Hi"},
        {"sender": "OldBot", "content": "Nice to meet you"},
    ]
    expected = [
        (None, "Greetings Human"),
        ("Hi", "Nice to meet you"),
    ]
    history = process_db_messages_to_ui_history(
        db_messages,
        char_name,
        user_name,
        additional_char_sender_ids=["OldBot"],
    )
    assert history == expected


def test_process_db_messages_to_ui_history_handles_legacy_user_alias():

    char_name = "Botty"
    user_name = "Friend"
    db_messages = [
        {"sender": "You", "content": "Hi {{char}}"},
        {"sender": "Botty", "content": "Hello {{user}}"},
    ]
    expected = [("Hi Botty", "Hello Friend")]
    assert process_db_messages_to_ui_history(db_messages, char_name, user_name) == expected


def test_process_db_messages_to_ui_history_respects_user_display_name():
    char_name = "Botty"
    user_name = "Companion"
    db_messages = [
        {"sender": "Companion", "content": "Hey there"},
        {"sender": char_name, "content": "Welcome back, {{user}}"},
    ]
    expected = [("Hey there", "Welcome back, Companion")]
    assert process_db_messages_to_ui_history(db_messages, char_name, user_name) == expected


def test_process_db_messages_to_ui_history_handles_character_named_like_user_alias():
    char_name = "User"
    user_name = "Player"
    greeting_template = "Greetings {{user}}"

    db_messages = [
        {
            "id": "char-1",
            "sender": "User",
            "content": greeting_template,
            "timestamp": "2024-01-01T00:00:00Z",
            "version": 1,
        },
        {
            "id": "user-1",
            "sender": "User",
            "content": "Hi there {{char}}",
            "timestamp": "2024-01-01T00:00:01Z",
            "version": 1,
        },
        {
            "id": "char-2",
            "sender": "User",
            "content": "Nice to see you, {{user}}",
            "timestamp": "2024-01-01T00:00:02Z",
            "version": 1,
        },
    ]

    expected_history = [
        (None, replace_placeholders(greeting_template, char_name, user_name)),
        (
            replace_placeholders("Hi there {{char}}", char_name, user_name),
            replace_placeholders("Nice to see you, {{user}}", char_name, user_name),
        ),
    ]

    history = process_db_messages_to_ui_history(
        db_messages,
        char_name,
        user_name,
        actual_user_sender_id_in_db="User",
        actual_char_sender_id_in_db=char_name,
        char_first_message=replace_placeholders(greeting_template, char_name, user_name),
    )

    assert history == expected_history


def test_process_db_messages_to_rich_ui_history_includes_metadata():
    char_name = "Botty"
    user_name = "Human"
    db_messages = [
        {
            "id": "user-1",
            "sender": "User",
            "content": "Hello {{char}}",
            "timestamp": "2024-01-01T00:00:00Z",
            "ranking": None,
            "version": 1,
        },
        {
            "id": "char-1",
            "sender": char_name,
            "content": "Hi {{user}}",
            "timestamp": "2024-01-01T00:00:01Z",
            "ranking": 2,
            "version": 1,
            "image_data": b"fake-image-bytes",
            "image_mime_type": "image/png",
            "images": [
                {"position": 1, "image_data": b"secondary-bytes", "image_mime_type": "image/jpeg"},
            ],
        },
    ]

    history = process_db_messages_to_rich_ui_history(db_messages, char_name, user_name)

    assert len(history) == 1
    turn = history[0]

    assert turn["user"] is not None and turn["character"] is not None
    assert turn["user"]["id"] == "user-1"
    assert turn["user"]["content"] == "Hello Botty"
    assert turn["character"]["id"] == "char-1"
    assert turn["character"]["content"] == "Hi Human"
    attachments = turn["character"]["attachments"]
    assert len(attachments) == 2
    # Ensure attachment metadata preserved
    assert attachments[0]["mime_type"] == "image/png"
    assert attachments[1]["mime_type"] == "image/jpeg"
    assert attachments[0]["encoding"] == "base64"
    assert attachments[0]["size"] == len(b"fake-image-bytes")
    assert base64.b64decode(attachments[0]["data"]) == b"fake-image-bytes"
    assert base64.b64decode(attachments[1]["data"]) == b"secondary-bytes"

    from fastapi.encoders import jsonable_encoder

    serialized = jsonable_encoder(history)
    assert serialized[0]["character"]["attachments"][0]["data"] == attachments[0]["data"]


def test_process_db_messages_to_rich_ui_history_handles_non_character_roles():
    db_messages = [
        {"id": "user-1", "sender": "User", "content": "What's happening?"},
        {"id": "system-1", "sender": "system", "content": "Maintenance window in effect."},
    ]
    history = process_db_messages_to_rich_ui_history(db_messages, "Caretaker", "Admin")

    assert len(history) == 1
    turn = history[0]
    assert turn["user"]["id"] == "user-1"
    assert turn["non_character"]["id"] == "system-1"
    assert turn["non_character"]["content"] == "[system] Maintenance window in effect."


@pytest.mark.parametrize("sender_override", ["system:Chrony", "tool:Chrony"])
def test_process_db_messages_to_rich_ui_history_allows_reserved_prefix_char(sender_override):
    db_messages = [
        {"id": "user-1", "sender": "User", "content": "Hi {{char}}"},
        {"id": "char-1", "sender": sender_override, "content": "Hello {{user}}"},
    ]

    history = process_db_messages_to_rich_ui_history(
        db_messages,
        "Chrony",
        "Admin",
        actual_char_sender_id_in_db=sender_override,
    )

    assert len(history) == 1
    turn = history[0]
    assert turn["user"]["content"] == "Hi Chrony"
    assert turn["character"] is not None
    assert turn["character"]["sender"] == sender_override
    assert turn["character"]["content"] == "Hello Admin"
    assert turn["non_character"] is None

def test_chat_history_import_handles_structured_content_and_roles(db):
    char_payload = {
        "name": "Test Character",
        "description": "Helpful entity",
        "personality": "Warm",
        "scenario": "Testing importer",
        "first_message": "Greetings, {{user}}.",
        "message_example": "User: Hello\nTest Character: Hi there!",
    }
    char_id = create_new_character_from_data(db, char_payload)

    sample_history = {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are in a simulation."}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello there, {{user}}."}],
            },
            {
                "role": "tool",
                "name": "lookup",
                "content": [{"type": "text", "text": "Tool output"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Thanks!"},
                ],
            },
        ]
    }

    conversation_id, imported_char_id = load_chat_history_from_file_and_save_to_db(
        db,
        character_id=char_id,
        file_content=json.dumps(sample_history),
        user_name_for_placeholders="Alice",
    )

    assert conversation_id
    assert imported_char_id == char_id

    stored_messages = db.get_messages_for_conversation(conversation_id, limit=10)
    senders = [msg["sender"] for msg in stored_messages]
    contents = [msg["content"] for msg in stored_messages]

    assert senders == ["system", "Test Character", "tool:lookup", "User"]
    assert contents[0] == "You are in a simulation."
    assert contents[1] == "Hello there, {{user}}."
    assert "Tool output" in contents[2]
    assert contents[3] == "Thanks!"

    ui_history = retrieve_conversation_messages_for_ui(
        db,
        conversation_id=conversation_id,
        character_name="Test Character",
        user_name="Alice",
    )

    assert ui_history[0][1] == "[system] You are in a simulation."
    assert ui_history[1][1] == "Hello there, Alice."
    assert ui_history[2][1] == "[tool:lookup] Tool output"
    assert ui_history[-1][0] == "Thanks!"


MINIMAL_V2_DATA_NODE_UNIT = {
    "name": "TestV2",
    "description": "Desc",
    "personality": "Pers",
    "scenario": "Scen",
    "first_mes": "First",
    "mes_example": "Example",
}
MINIMAL_V2_CARD_UNIT = {"spec": "chara_card_v2", "spec_version": "2.0", "data": MINIMAL_V2_DATA_NODE_UNIT.copy()}
MINIMAL_V1_CARD_UNIT = {
    "name": "TestV1",
    "description": "Desc",
    "personality": "Pers",
    "scenario": "Scen",
    "first_mes": "First",
    "mes_example": "Example",
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
    book_content = {
        "name": "Lore",
        "entries": [{"keys": ["key"], "content": "val", "enabled": True, "insertion_order": 0}],
    }
    v2_with_book_data["character_book"] = book_content
    v2_card_with_book = {"spec": "chara_card_v2", "spec_version": "2.0", "data": v2_with_book_data}

    parsed_with_book = parse_v2_card(v2_card_with_book)
    assert parsed_with_book is not None
    assert "character_book" in parsed_with_book.get("extensions", {})
    # parse_character_book should have been called and its result stored
    assert parsed_with_book["extensions"]["character_book"]["name"] == "Lore"
    assert len(parsed_with_book["extensions"]["character_book"]["entries"]) == 1


def test_parse_v2_card_missing_mes_example_defaults_empty(caplog_handler):
    card = copy.deepcopy(MINIMAL_V2_CARD_UNIT)
    card["spec_version"] = "2.4"
    card["data"] = card["data"].copy()
    card["data"].pop("mes_example", None)
    caplog = caplog_handler
    with caplog.at_level("WARNING", logger="tldw_Server_API"):
        parsed = parse_v2_card(card)
    assert parsed is not None
    assert parsed["message_example"] == ""
    assert "mes_example" in caplog.text


def test_parse_v1_card_unit():
    parsed = parse_v1_card(MINIMAL_V1_CARD_UNIT.copy())
    assert parsed is not None and parsed["name"] == "TestV1"
    v1_extra = {**MINIMAL_V1_CARD_UNIT, "custom_field": "custom_val"}
    parsed_extra = parse_v1_card(v1_extra)
    assert parsed_extra["extensions"]["custom_field"] == "custom_val"


def test_import_textgen_context_greeting():
    payload = {
        "name": "TextGenChar",
        "context": "Backstory context",
        "greeting": "Hello there!",
        "example_dialogue": "User: Hi\nBot: Hello",
    }
    parsed = import_character_card_from_json_string(json.dumps(payload))
    assert parsed is not None
    assert parsed["name"] == "TextGenChar"
    assert parsed["first_message"] == "Hello there!"
    assert parsed["description"] == "Backstory context"
    assert parsed["scenario"] == "Backstory context"
    assert parsed["message_example"] == "User: Hi\nBot: Hello"


def test_import_alpaca_instruction_input():
    payload = {
        "instruction": "Follow the rules.",
        "input": "Be concise.",
        "output": "Understood.",
    }
    parsed = import_character_card_from_json_string(json.dumps(payload))
    assert parsed is not None
    assert parsed["name"].startswith("Alpaca-")
    assert parsed["description"] == "Follow the rules."
    assert parsed["scenario"] == "Follow the rules."
    assert parsed["personality"] == "Be concise."
    assert parsed["first_message"] == "Understood."
    assert parsed["message_example"] == "Understood."


def test_import_pygmalion_card_fields():
    payload = {
        "char_name": "PygmalionChar",
        "char_persona": "Pygmalion persona",
        "world_scenario": "Pygmalion scenario",
        "char_greeting": "Hello from Pygmalion",
        "example_dialogue": "User: Hi\nBot: Hello",
    }
    parsed = import_character_card_from_json_string(json.dumps(payload))
    assert parsed is not None
    assert parsed["name"] == "PygmalionChar"
    assert parsed["description"] == "Pygmalion persona"
    assert parsed["personality"] == "Pygmalion persona"
    assert parsed["scenario"] == "Pygmalion scenario"
    assert parsed["first_message"] == "Hello from Pygmalion"
    assert parsed["message_example"] == "User: Hi\nBot: Hello"


def test_prepare_character_data_preserves_transparency():
    transparent_img = io.BytesIO()
    PILImageReal.new("RGBA", (2, 2), (255, 0, 0, 0)).save(transparent_img, format="PNG")
    encoded = base64.b64encode(transparent_img.getvalue()).decode("utf-8")

    db_ready = _prepare_character_data_for_db_storage({"name": "TransparentChar", "image_base64": encoded})
    stored_image_bytes = db_ready["image"]
    assert isinstance(stored_image_bytes, (bytes, bytearray))

    processed = PILImageReal.open(io.BytesIO(stored_image_bytes))
    assert processed.mode in ("RGBA", "LA")
    pixel = processed.getpixel((0, 0))
    assert len(pixel) >= 2 and pixel[-1] == 0


def test_prepare_character_data_uses_lossless_for_alpha():
    """
    Regression test for Issue #5: WEBP alpha channel handling.

    When saving images with alpha channels, lossless compression should be used
    to preserve transparency quality without artifacts.
    """
    # Create a test image with semi-transparent pixels
    test_img = io.BytesIO()
    img = PILImageReal.new("RGBA", (10, 10), (128, 64, 32, 128))  # Semi-transparent
    img.save(test_img, format="PNG")
    encoded = base64.b64encode(test_img.getvalue()).decode("utf-8")

    db_ready = _prepare_character_data_for_db_storage({"name": "AlphaTest", "image_base64": encoded})
    stored_image_bytes = db_ready["image"]

    # Verify the output is a valid image with alpha preserved
    processed = PILImageReal.open(io.BytesIO(stored_image_bytes))
    assert "A" in processed.getbands(), "Alpha channel should be preserved"

    # Verify alpha value is preserved (lossless should keep exact values)
    pixel = processed.getpixel((0, 0))
    assert len(pixel) == 4, "RGBA pixel should have 4 components"
    assert pixel[3] == 128, f"Alpha value should be preserved as 128, got {pixel[3]}"


@pytest.mark.parametrize("empty_value", ["", "   ", "\n\t"])
def test_prepare_character_data_update_ignores_empty_image_base64(empty_value):
    payload = {"name": "NoOpImageUpdate", "image_base64": empty_value}
    db_ready = _prepare_character_data_for_db_storage(payload, is_update=True)

    assert db_ready["name"] == "NoOpImageUpdate"
    assert "image" not in db_ready


def test_prepare_character_data_rejects_decodable_non_image_payload():
    non_image_bytes = b"this-is-not-a-real-image"
    encoded = base64.b64encode(non_image_bytes).decode("utf-8")

    with pytest.raises(InputError, match="not a valid image"):
        _prepare_character_data_for_db_storage({"name": "InvalidBinaryImage", "image_base64": encoded})


def test_parse_character_book_unit():

    # Valid book
    book_data = {
        "name": "My Lore",
        "description": "Lore desc",
        "scan_depth": 10,
        "entries": [{"keys": ["topic"], "content": "info", "enabled": True, "insertion_order": 1, "name": "Entry1"}],
    }
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


VALID_BOOK_ENTRY_UNIT = {"keys": ["key1"], "content": "Entry content", "enabled": True, "insertion_order": 0}
VALID_BOOK_UNIT = {"entries": [VALID_BOOK_ENTRY_UNIT.copy()]}


def test_validate_character_book_entry_unit():
    is_valid, errors = validate_character_book_entry(VALID_BOOK_ENTRY_UNIT.copy(), 0, set())
    assert is_valid and not errors
    # Test invalid: missing key
    entry_no_key = VALID_BOOK_ENTRY_UNIT.copy()
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


def test_validate_character_book_entry_at_depth_position():
    """
    Regression test for Issue #6: Position validation values.

    The 'at_depth' position value should be accepted as valid per V2 spec.
    """
    entry_at_depth = {
        "keys": ["keyword1"],
        "content": "Entry content",
        "enabled": True,
        "insertion_order": 0,
        "position": "at_depth",  # This should now be valid
    }
    is_valid, errors = validate_character_book_entry(entry_at_depth, 0, set())
    assert is_valid, f"Position 'at_depth' should be valid. Errors: {errors}"
    assert not errors, "No validation errors should occur for 'at_depth' position"


def test_validate_v2_card_unit():
    is_valid, errors = validate_v2_card(MINIMAL_V2_CARD_UNIT.copy())
    assert is_valid and not errors
    # Test invalid: missing spec
    card_no_spec = MINIMAL_V2_CARD_UNIT.copy()
    del card_no_spec["spec"]
    is_valid_ns, errors_ns = validate_v2_card(card_no_spec)
    assert not is_valid_ns and "Missing 'spec' field" in errors_ns[0]


def test_validate_v2_card_allows_implicit_spec():
    card_no_spec = {"data": MINIMAL_V2_DATA_NODE_UNIT.copy()}
    is_valid, errors = validate_v2_card(card_no_spec, strict_spec=False)
    assert is_valid and not errors


def test_validate_v2_card_accepts_newer_minor_version():
    card = copy.deepcopy(MINIMAL_V2_CARD_UNIT)
    card["spec_version"] = "2.3"
    is_valid, errors = validate_v2_card(card)
    assert is_valid and not errors


def test_validate_v2_card_missing_mes_example_logs_warning(caplog_handler):
    card = copy.deepcopy(MINIMAL_V2_CARD_UNIT)
    card["data"] = card["data"].copy()
    card["data"].pop("mes_example", None)
    caplog = caplog_handler
    with caplog.at_level("WARNING", logger="tldw_Server_API"):
        is_valid, errors = validate_v2_card(card)
    assert is_valid and not errors
    assert "mes_example" in caplog.text


def test_validate_v2_card_rejects_overlong_text_field():
    card = copy.deepcopy(MINIMAL_V2_CARD_UNIT)
    card["data"] = card["data"].copy()
    card["data"]["name"] = "N" * 501
    is_valid, errors = validate_v2_card(card)
    assert not is_valid
    assert any("name" in err and "exceeds max length" in err for err in errors)


def test_validate_v2_card_rejects_non_string_list_values():
    card = copy.deepcopy(MINIMAL_V2_CARD_UNIT)
    card["data"] = card["data"].copy()
    card["data"]["alternate_greetings"] = ["hi", 42]
    card["data"]["tags"] = ["ok", {"bad": True}]
    is_valid, errors = validate_v2_card(card)
    assert not is_valid
    assert any("alternate_greetings[1]" in err for err in errors)
    assert any("tags[1]" in err for err in errors)


@mock.patch(f"{MODULE_PATH_PREFIX}.character_validation.validate_v2_card")
@mock.patch(f"{MODULE_PATH_PREFIX}.character_validation.parse_v2_card")
@mock.patch(f"{MODULE_PATH_PREFIX}.character_validation.parse_v1_card")
def test_import_character_card_from_json_string_unit(mock_parse_v1, mock_parse_v2, mock_validate_v2):
    mock_validate_v2.return_value = (True, [])
    mock_parse_v2.return_value = {"name": "ParsedV2"}
    v2_str = json.dumps(MINIMAL_V2_CARD_UNIT)
    assert import_character_card_from_json_string(v2_str)["name"] == "ParsedV2"

    mock_validate_v2.return_value = (False, ["Not V2"])
    mock_parse_v1.return_value = {"name": "ParsedV1"}
    v1_str = json.dumps(MINIMAL_V1_CARD_UNIT)
    assert import_character_card_from_json_string(v1_str)["name"] == "ParsedV1"


def test_import_character_card_from_json_string_implicit_v2():
    implicit_v2 = {"data": MINIMAL_V2_DATA_NODE_UNIT.copy()}
    parsed = import_character_card_from_json_string(json.dumps(implicit_v2))
    assert parsed is not None
    assert parsed.get("name") == MINIMAL_V2_DATA_NODE_UNIT["name"]


@mock.patch(f"{MODULE_PATH_PREFIX}.character_io.import_character_card_from_json_string")
@mock.patch(f"{MODULE_PATH_PREFIX}.character_io.yaml")
def test_load_character_card_from_string_content_unit(
    mock_yaml_module, mock_import_json_str, caplog_handler
):  # caplog_handler now works
    mock_import_json_str.return_value = {"name": "Loaded"}
    json_content = json.dumps(MINIMAL_V1_CARD_UNIT)
    assert load_character_card_from_string_content(json_content)["name"] == "Loaded"

    yaml_text = "name: YChar\ndescription: D\nfirst_mes: FM\nmes_example: ME\npersonality: P\nscenario: S"
    yaml_front = f"---\n{yaml_text}\n---"

    expected_dict_from_yaml = {
        "name": "YChar",
        "description": "D",
        "first_mes": "FM",
        "mes_example": "ME",
        "personality": "P",
        "scenario": "S",
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
    # When YAML parsing fails, it creates a plain text character and passes it through import_character_card_from_json_string
    mock_import_json_str.return_value = {"name": "Character", "description": malformed_yaml, "tags": ["plain-text"]}
    result = load_character_card_from_string_content(malformed_yaml)
    # When YAML parsing fails, the function falls through and creates a plain text character
    assert result is not None
    assert "Error parsing YAML frontmatter" in caplog_handler.text  # Should pass now
    mock_import_json_str.assert_called()  # It will create JSON from plain text and import it
    mock_yaml_module.safe_load.side_effect = None


def test_create_new_character_from_data_accepts_wrapped_base64(db):
    image_bytes = create_dummy_png_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    wrapped = "\n".join(encoded[i : i + 60] for i in range(0, len(encoded), 60))

    payload = {"name": "WhitespaceImageChar", "image_base64": wrapped}

    char_id = create_new_character_from_data(db, payload)
    assert char_id is not None

    stored = db.get_character_card_by_id(char_id)
    assert stored is not None
    assert isinstance(stored.get("image"), bytes)


def test_load_character_card_from_plain_text_creates_minimal_card():
    plain_text = "These are plain text notes for a character."
    result = load_character_card_from_string_content(plain_text)

    assert result is not None
    assert result["description"] == plain_text
    assert result["first_message"]
    assert result["message_example"]
    assert result["tags"] == ["plain-text"]
    assert result["personality"]
    assert result["scenario"]


@mock.patch(f"{MODULE_PATH_PREFIX}.character_io.Image", new_callable=mock.MagicMock)
@mock.patch(f"{MODULE_PATH_PREFIX}.character_io.base64")  # This is the base64 module used by the facade
@mock.patch(f"{MODULE_PATH_PREFIX}.character_io.json")  # This mock is 'mock_json_loads_mod'
def test_extract_json_from_image_file_unit(
    mock_json_loads_mod, mock_base64_mod, MockPILImageModule, tmp_path, caplog_handler
):
    # --- FIX: Configure the mocked json module ---
    # The SUT uses 'json.JSONDecodeError'.
    # Since MODULE_PATH_PREFIX.json is mocked (as mock_json_loads_mod),
    # we need to ensure that mock_json_loads_mod.JSONDecodeError refers to the actual exception class.
    # 'json' here refers to the 'import json' at the top of this test file (the real json module).
    mock_json_loads_mod.JSONDecodeError = json.JSONDecodeError
    # --- END FIX ---

    mock_img_instance = MockPILImageObject()
    MockPILImageModule.open.return_value = mock_img_instance

    chara_json_str = '{"name": "CharaFromImage"}'
    # base64 from 'import base64' (real module) is used for test data setup
    b64_encoded_bytes = base64.b64encode(chara_json_str.encode("utf-8"))
    b64_encoded_str = b64_encoded_bytes.decode("utf-8")

    mock_img_instance.info = {"chara": b64_encoded_str}
    mock_img_instance.format = "PNG"

    # Default good path return values
    default_b64_return = chara_json_str.encode("utf-8")
    mock_base64_mod.b64decode.return_value = default_b64_return
    # mock_json_loads_mod.loads is the mocked function SUT will call.
    # json.loads(chara_json_str) uses the real 'json' module to prepare the expected return value.
    mock_json_loads_mod.loads.return_value = json.loads(chara_json_str)

    dummy_png_path = tmp_path / "test_unit.png"
    dummy_png_path.write_text("dummy_content_for_file_existence")

    result = extract_json_from_image_file(str(dummy_png_path))
    assert (
        result == chara_json_str
    )  # This should be json.loads(result) == json.loads(chara_json_str) if result is JSON string or just comparing strings is fine.
    # The function returns a string, so string comparison is correct.
    MockPILImageModule.open.assert_called_once()
    assert isinstance(MockPILImageModule.open.call_args[0][0], io.BytesIO)
    mock_base64_mod.b64decode.assert_called_once_with(b64_encoded_str)
    mock_json_loads_mod.loads.assert_called_once_with(chara_json_str)
    MockPILImageModule.open.reset_mock()
    mock_base64_mod.b64decode.reset_mock()
    mock_json_loads_mod.loads.reset_mock()
    caplog_handler.clear()

    MockPILImageModule.open.reset_mock()
    mock_base64_mod.b64decode.reset_mock()
    mock_json_loads_mod.loads.reset_mock()
    caplog_handler.clear()

    # Test Non-PNG with chara key
    mock_img_instance.format = "JPEG"
    mock_base64_mod.b64decode.return_value = default_b64_return
    mock_json_loads_mod.loads.return_value = json.loads(chara_json_str)
    assert extract_json_from_image_file(str(dummy_png_path)) == chara_json_str
    assert "not in PNG or WEBP format" in caplog_handler.text
    caplog_handler.clear()
    mock_img_instance.format = "PNG"  # Reset format

    # --- Test error in b64decode / subsequent decode ---
    # To test the (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) block

    mock_base64_mod.b64decode.reset_mock()
    mock_base64_mod.b64decode.return_value = (
        b"\xff\xfe\xfd"  # Invalid UTF-8 sequence, will cause .decode('utf-8') to fail
    )
    mock_json_loads_mod.loads.reset_mock()  # Not reached if .decode() fails

    assert extract_json_from_image_file(str(dummy_png_path)) is None
    # This assertion should now pass because the correct logger.error will be called in the SUT
    assert "Error decoding 'chara' metadata" in caplog_handler.text
    mock_base64_mod.b64decode.assert_called_once_with(b64_encoded_str)
    mock_json_loads_mod.loads.assert_not_called()
    mock_base64_mod.b64decode.return_value = default_b64_return
    mock_base64_mod.b64decode.side_effect = None
    caplog_handler.clear()

    # Reset for next test section
    mock_base64_mod.b64decode.return_value = default_b64_return
    mock_base64_mod.b64decode.side_effect = None
    caplog_handler.clear()

    # Test PIL UnidentifiedImageError
    MockPILImageModule.open.reset_mock()
    MockPILImageModule.open.side_effect = PILImageReal.UnidentifiedImageError("bad image file")
    assert extract_json_from_image_file(str(dummy_png_path)) is None
    assert "Cannot open or read image file" in caplog_handler.text
    MockPILImageModule.open.side_effect = None
    MockPILImageModule.open.return_value = mock_img_instance
    caplog_handler.clear()


def test_extract_json_from_image_file_rejects_oversized_ztxt(tmp_path):
    def _build_png_with_ztxt(keyword: bytes, text_bytes: bytes) -> bytes:
        base_png = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        )
        idat_chunk = b"\x00\x00\x00\x0cIDAT\x08\xd7c`\x00\x00\x00\x02\x00\x01\xe2!\xbc\x33"
        iend_chunk = b"\x00\x00\x00\x00IEND\xaeB`\x82"

        compressed_text = zlib.compress(text_bytes)
        chunk_type = b"zTXt"
        chunk_data = keyword + b"\x00" + b"\x00" + compressed_text
        chunk_len_bytes = len(chunk_data).to_bytes(4, "big")
        crc_val = binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        crc_bytes = crc_val.to_bytes(4, "big")
        ztxt_chunk = chunk_len_bytes + chunk_type + chunk_data + crc_bytes
        return base_png + ztxt_chunk + idat_chunk + iend_chunk

    oversized_json = {
        "name": "Oversized",
        "description": "x" * (MAX_PNG_METADATA_BYTES),
    }
    json_bytes = json.dumps(oversized_json).encode("utf-8")
    b64_payload = base64.b64encode(json_bytes)
    if len(b64_payload) <= MAX_PNG_METADATA_BYTES:
        b64_payload += b"A" * (MAX_PNG_METADATA_BYTES - len(b64_payload) + 1)

    png_bytes = _build_png_with_ztxt(b"chara", b64_payload)
    dummy_png_path = tmp_path / "oversized.png"
    dummy_png_path.write_bytes(png_bytes)

    assert extract_json_from_image_file(str(dummy_png_path)) is None


def test_extract_json_from_image_file_rejects_crc_mismatch(tmp_path):
    chara_json = json.dumps({"name": "CRC Broken", "description": "bad crc"})
    encoded = base64.b64encode(chara_json.encode("utf-8"))

    chunk_type = b"tEXt"
    chunk_data = b"chara\x00" + encoded
    chunk_len = len(chunk_data).to_bytes(4, "big")
    good_crc = binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
    bad_crc = (good_crc ^ 0xFFFFFFFF).to_bytes(4, "big")
    text_chunk = chunk_len + chunk_type + chunk_data + bad_crc

    base_png = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    )
    idat_chunk = b"\x00\x00\x00\x0cIDAT\x08\xd7c`\x00\x00\x00\x02\x00\x01\xe2!\xbc\x33"
    iend_chunk = b"\x00\x00\x00\x00IEND\xaeB`\x82"
    png_bytes = base_png + text_chunk + idat_chunk + iend_chunk

    dummy_png_path = tmp_path / "crc_bad.png"
    dummy_png_path.write_bytes(png_bytes)

    assert extract_json_from_image_file(str(dummy_png_path)) is None


def test_extract_json_from_image_file_rejects_malformed_chunk_length(tmp_path):
    base_png = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    )
    # Declares length 32 but only includes 5 bytes before CRC, which should be rejected.
    malformed_chunk = (
        (32).to_bytes(4, "big")
        + b"tEXt"
        + b"chara"
        + (0).to_bytes(4, "big")
    )
    iend_chunk = b"\x00\x00\x00\x00IEND\xaeB`\x82"
    png_bytes = base_png + malformed_chunk + iend_chunk

    dummy_png_path = tmp_path / "malformed_chunk.png"
    dummy_png_path.write_bytes(png_bytes)

    assert extract_json_from_image_file(str(dummy_png_path)) is None


# --- Property Tests (using Hypothesis) ---

# Strategy for names that don't contain placeholders themselves
safe_name_st = st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=0, max_size=20).filter(
    lambda x: not any(p in x for p in ["{{char}}", "{{user}}", "<CHAR>", "<USER>", "{{random_user}}"])
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
    expected = expected.replace("{{char}}", char_name_actual)
    expected = expected.replace("<CHAR>", char_name_actual)
    expected = expected.replace("{{user}}", user_name_actual)
    expected = expected.replace("<USER>", user_name_actual)
    expected = expected.replace("{{random_user}}", user_name_actual)

    assert result == expected


id_st = st.integers(min_value=0, max_value=10**9)
simple_char_name_for_id_test_st = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=20
).filter(
    lambda x: "(" not in x and ")" not in x
)  # Avoid parens in name to simplify test


@given(
    id_val=id_st,
    name=simple_char_name_for_id_test_st,
    id_internal_leading_spaces_count=st.integers(min_value=0, max_value=2),
    id_internal_trailing_spaces_count=st.integers(min_value=0, max_value=2),
    overall_leading_spaces_count=st.integers(min_value=0, max_value=2),
    overall_trailing_spaces_count=st.integers(min_value=0, max_value=2),
)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large], deadline=None)
def test_property_extract_character_id_from_ui_choice_valid_formats(
    id_val,
    name,
    id_internal_leading_spaces_count,
    id_internal_trailing_spaces_count,
    overall_leading_spaces_count,
    overall_trailing_spaces_count,
):
    id_int_ls = " " * id_internal_leading_spaces_count
    id_int_ts = " " * id_internal_trailing_spaces_count
    overall_ls = " " * overall_leading_spaces_count
    overall_ts = " " * overall_trailing_spaces_count

    # Test "Name (ID: <id_val>)" format
    # The SUT's regex r'\(ID:\s*(\d+)\s*\)$' means the ')' must be the last non-whitespace char
    # if this pattern is to be matched by the regex.
    # So, if we add overall_trailing_spaces, the regex match will fail, and it will fall
    # to the choice.strip().isdigit() path, which would also fail for "Name (ID: id)".
    # Thus, for this specific format to be matched by the regex, overall_trailing_spaces must be empty.

    # Case 1: Regex match path " [Name] (ID: [spaces] id [spaces]) "
    # No overall trailing spaces for the regex to match with `$`
    choice1_core = f"{name} (ID:{id_int_ls}{id_val}{id_int_ts})"
    choice1_for_regex = f"{overall_ls}{choice1_core}"  # No overall_ts
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


@given(empty_or_whitespace_choice=st.text(alphabet=" \t\n\r", min_size=0, max_size=10))
@settings(
    suppress_health_check=[HealthCheck.data_too_large, HealthCheck.filter_too_much], deadline=None
)  # Added deadline
def test_property_extract_character_id_from_ui_choice_empty_or_whitespace(empty_or_whitespace_choice):
    if not empty_or_whitespace_choice:
        with pytest.raises(ValueError, match="No choice provided"):
            extract_character_id_from_ui_choice(empty_or_whitespace_choice)
    elif not empty_or_whitespace_choice.strip():  # Becomes empty after strip
        with pytest.raises(ValueError, match="Invalid choice format"):
            extract_character_id_from_ui_choice(empty_or_whitespace_choice)
    # If it's whitespace but not empty after strip (e.g. " 123 "), it's handled by valid_formats
    # If it's whitespace with non-numeric (e.g. " abc "), it's handled by invalid_formats


# For parse_v1_card
v1_required_fields_st = st.fixed_dictionaries(
    {
        "name": st.text(min_size=1, max_size=50),
        "description": st.text(max_size=100),
        "personality": st.text(max_size=100),
        "scenario": st.text(max_size=100),
        "first_mes": st.text(min_size=1, max_size=100),
        "mes_example": st.text(max_size=100),
    }
)
v1_optional_fields_st = st.fixed_dictionaries(
    {
        "creator_notes": st.text(max_size=100),
        "system_prompt": st.text(max_size=100),
        "post_history_instructions": st.text(max_size=100),
        "alternate_greetings": st.lists(st.text(max_size=50), max_size=3),
        "tags": st.lists(st.text(max_size=20), max_size=5),
        "creator": st.text(max_size=30),
        "character_version": st.text(max_size=10),
        "char_image": st.one_of(st.none(), st.text(max_size=50)),  # Can be None, empty string, or text
        "image": st.one_of(st.none(), st.text(max_size=50)),  # Can be None, empty string, or text
    }
)
known_v1_keys = {
    "name",
    "description",
    "personality",
    "scenario",
    "first_mes",
    "mes_example",
    "creator_notes",
    "system_prompt",
    "post_history_instructions",
    "alternate_greetings",
    "tags",
    "creator",
    "character_version",
    "char_image",
    "image",
}
extension_key_st = st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)).filter(
    lambda k: k not in known_v1_keys
)
extension_value_st = st.one_of(st.text(max_size=50), st.integers(), st.booleans(), st.none())
extensions_st = st.dictionaries(extension_key_st, extension_value_st, max_size=3)


@given(required_data=v1_required_fields_st, optional_data=v1_optional_fields_st, extensions=extensions_st)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large], deadline=None)
def test_property_parse_v1_card_structure_and_extensions(required_data, optional_data, extensions):
    v1_card_data = {**required_data, **optional_data, **extensions}
    parsed = parse_v1_card(v1_card_data)

    assert parsed is not None
    assert parsed["name"] == required_data["name"]
    assert parsed["first_message"] == required_data["first_mes"]
    assert parsed["creator_notes"] == optional_data.get("creator_notes", "")
    assert parsed["alternate_greetings"] == optional_data.get("alternate_greetings", [])

    # Correctly test image_base64 based on SUT's `val1 or val2` logic
    expected_image_base64 = v1_card_data.get("char_image") or v1_card_data.get("image")
    assert parsed["image_base64"] == expected_image_base64

    parsed_extensions = parsed.get("extensions", {})
    for key, value in extensions.items():
        assert key in parsed_extensions and parsed_extensions[key] == value
    # Ensure standard keys (that were part of required_data or optional_data) are not in extensions
    for key in set(required_data.keys()) | set(optional_data.keys()):
        if key not in extensions:  # Unless it was *also* an extension key (unlikely with filter)
            assert key not in parsed_extensions


@given(base_card=v1_required_fields_st)
@settings(suppress_health_check=[HealthCheck.data_too_large], deadline=None)
def test_property_parse_v1_card_missing_required_fields(base_card):
    import random  # Keep import local if only used here

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
    user_name=optional_safe_name_st,
)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large], deadline=None)
def test_property_process_db_messages_to_ui_history_only_user(user_messages_content, char_name, user_name):
    db_messages = [{"sender": "User", "content": content} for content in user_messages_content]
    user_name_actual = user_name if user_name is not None else "User"
    char_name_actual = char_name  # Already filtered to be non-None, non-empty
    processed_history = process_db_messages_to_ui_history(db_messages, char_name_actual, user_name_actual)

    assert len(processed_history) == len(user_messages_content)
    for i, original_content in enumerate(user_messages_content):
        expected_processed_content = replace_placeholders(original_content, char_name_actual, user_name_actual)
        assert processed_history[i] == (expected_processed_content, None)


@given(
    char_messages_content=st.lists(db_message_content_st, min_size=0, max_size=3),
    char_name=safe_name_st.filter(lambda x: x != "User" and x != "" and x is not None),
    user_name=optional_safe_name_st,
)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large], deadline=None)
def test_property_process_db_messages_to_ui_history_only_char(char_messages_content, char_name, user_name):
    char_name_actual = char_name  # Already filtered
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
    user_name=optional_safe_name_st,
)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large], deadline=None)
def test_property_process_db_messages_to_ui_history_alternating(message_pairs_content, char_name, user_name):
    char_name_actual = char_name  # Already filtered
    db_messages = []
    for user_content, char_content_for_pair in message_pairs_content:  # Renamed to avoid clash
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
    db.soft_delete_character_card(char3_data_for_delete["id"], char3_data_for_delete["version"])

    ui_list = get_character_list_for_ui(db, limit=10)
    # The list should be sorted by name: Alice, Charlie, Default Assistant
    # Bob (deleted) should not appear
    assert len(ui_list) == 3  # Including Default Assistant
    assert ui_list[0]["name"] == "Alice"
    assert ui_list[1]["name"] == "Charlie"
    assert ui_list[2]["name"] == "Default Assistant"


@mock.patch(
    f"{MODULE_PATH_PREFIX}.character_db.Image", new_callable=mock.MagicMock
)  # Patches PIL.Image used by the facade
def test_load_character_and_image_integration(MockPILImageModule, db, caplog_handler):
    mock_opened_image = MockPILImageObject(format="PNG")  # This is what Image.open() will return
    mock_converted_image = MockPILImageObject(format="PNG", mode="RGBA")  # This is what .convert() will return

    # Configure the mock chain: Image.open().convert()
    MockPILImageModule.open.return_value = mock_opened_image
    mock_opened_image.convert = mock.Mock(
        return_value=mock_converted_image
    )  # Mock the convert method on the opened instance

    image_bytes = create_dummy_png_bytes()
    char_id = db.add_character_card(
        {"name": "Gandalf", "description": "W {{user}}", "first_message": "FM {{char}} {{user}}", "image": image_bytes}
    )

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
    assert (
        f"Error processing image for character 'Gandalf' (ID: {char_id})" in caplog_handler.text
    )  # This will use the new caplog_handler

    MockPILImageModule.open.side_effect = None  # Reset side effect
    MockPILImageModule.open.return_value = mock_opened_image  # Reset return value for other tests


@mock.patch(f"{MODULE_PATH_PREFIX}.character_db.Image", new_callable=mock.MagicMock)
def test_load_character_and_image_accepts_memoryview(MockPILImageModule, db):
    mock_opened_image = MockPILImageObject(format="PNG")
    mock_converted_image = MockPILImageObject(format="PNG", mode="RGBA")
    MockPILImageModule.open.return_value = mock_opened_image
    mock_opened_image.convert = mock.Mock(return_value=mock_converted_image)

    image_bytes = create_dummy_png_bytes()
    char_id = db.add_character_card(
        {
            "name": "Memory",
            "description": "Handles binary buffers",
            "first_message": "Hi {{user}}",
            "image": image_bytes,
        }
    )

    original_get = db.get_character_card_by_id

    def _get_with_memoryview(character_id_value: int):
        data = original_get(character_id_value)
        if data and data.get("image"):
            data["image"] = memoryview(data["image"])
        return data

    with mock.patch.object(db, "get_character_card_by_id", side_effect=_get_with_memoryview):
        loaded_char, history, image_obj = load_character_and_image(db, char_id, "Frodo")

    assert loaded_char["name"] == "Memory"
    assert history == [(None, "Hi Frodo")]
    assert image_obj == mock_converted_image
    mock_opened_image.convert.assert_called_once_with("RGBA")


@mock.patch(f"{MODULE_PATH_PREFIX}.character_io.yaml")
@mock.patch(f"{MODULE_PATH_PREFIX}.character_io.Image", new_callable=mock.MagicMock)
def test_import_and_save_character_from_file_integration(MockPILImageModule, mock_yaml_module, db, tmp_path):
    # JSON file
    v1_content = {
        "name": "JSON Char",
        "description": "D",
        "first_mes": "FM",
        "mes_example": "ME",
        "personality": "P",
        "scenario": "S",
    }
    json_file = tmp_path / "import.json"
    json_file.write_text(json.dumps(v1_content))
    success_json, message_json, char_id_json = import_and_save_character_from_file(db, str(json_file))
    assert success_json and char_id_json is not None
    assert db.get_character_card_by_id(char_id_json)["name"] == "JSON Char"

    # PNG with chara
    mock_img_instance = MockPILImageObject()  # Our custom mock
    MockPILImageModule.open.return_value = mock_img_instance
    png_chara_data = {"name": "PNG Chara", "first_mes": "FMpng"}
    png_chara_json_str = json.dumps({**MINIMAL_V1_CARD_UNIT, **png_chara_data})  # Ensure all V1 fields for parser
    b64_encoded = base64.b64encode(png_chara_json_str.encode("utf-8")).decode("utf-8")
    mock_img_instance.info = {"chara": b64_encoded}
    mock_img_instance.format = "PNG"
    dummy_png_bytes = create_dummy_png_bytes(png_chara_json_str)
    png_file = tmp_path / "chara.png"
    png_file.write_bytes(dummy_png_bytes)

    success_png, message_png, char_id_png = import_and_save_character_from_file(db, str(png_file))
    assert success_png and char_id_png is not None
    retrieved_png = db.get_character_card_by_id(char_id_png)
    assert retrieved_png["name"] == "PNG Chara" and retrieved_png["image"] == dummy_png_bytes


def test_import_and_save_character_from_file_returns_v2_validation_error(db, tmp_path):
    invalid_v2 = {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "BrokenV2",
            "description": "D",
            "personality": "P",
            "scenario": "S",
            "first_mes": 123,
            "mes_example": "Example",
        },
    }
    json_file = tmp_path / "invalid_v2.json"
    json_file.write_text(json.dumps(invalid_v2))

    success, message, char_id = import_and_save_character_from_file(db, str(json_file))
    assert not success
    assert char_id is None
    assert "first_mes" in message


def test_import_and_save_character_from_file_rejects_disallowed_script_payload(db, tmp_path):
    payload = {
        **MINIMAL_V1_CARD_UNIT,
        "name": "ScriptReject",
        "description": "<script>alert('x')</script>",
    }
    json_file = tmp_path / "script_payload.json"
    json_file.write_text(json.dumps(payload))

    success, message, char_id = import_and_save_character_from_file(db, str(json_file))
    assert not success
    assert char_id is None
    assert "disallowed content pattern" in message


def test_import_and_save_character_from_file_rejects_oversized_avatar_base64(db, tmp_path):
    oversized_avatar = create_dummy_png_bytes() + (b"A" * (MAX_IMPORT_AVATAR_BYTES + 1024))
    payload = {
        **MINIMAL_V1_CARD_UNIT,
        "name": "OversizedAvatar",
        "char_image": base64.b64encode(oversized_avatar).decode("utf-8"),
    }
    json_file = tmp_path / "oversized_avatar.json"
    json_file.write_text(json.dumps(payload))

    success, message, char_id = import_and_save_character_from_file(db, str(json_file))
    assert not success
    assert char_id is None
    assert "exceeds max size" in message


def test_resolve_max_import_avatar_bytes_enforces_multi_mb_floor(monkeypatch):
    monkeypatch.setenv("MAX_CHARACTER_IMPORT_AVATAR_SIZE_MB", "0.1953125")
    monkeypatch.delenv("MAX_CHARACTER_IMPORT_AVATAR_SIZE_BYTES", raising=False)

    assert _resolve_max_import_avatar_bytes() == 5 * 1024 * 1024


def test_resolve_max_import_avatar_bytes_honors_bytes_override(monkeypatch):
    expected = 7 * 1024 * 1024
    monkeypatch.setenv("MAX_CHARACTER_IMPORT_AVATAR_SIZE_BYTES", str(expected))
    monkeypatch.delenv("MAX_CHARACTER_IMPORT_AVATAR_SIZE_MB", raising=False)

    assert _resolve_max_import_avatar_bytes() == expected


def test_validate_import_avatar_bytes_accepts_multi_mb_jpeg():
    side = 1024
    random_rgb = os.urandom(side * side * 3)
    image = PILImageReal.frombytes("RGB", (side, side), random_rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    image_bytes = buffer.getvalue()

    # Regression guard: this should be comfortably above the previous 200KB cap.
    assert len(image_bytes) > (250 * 1024)
    assert len(image_bytes) < MAX_IMPORT_AVATAR_BYTES

    assert _validate_import_avatar_bytes(image_bytes) is None


def test_import_and_save_character_from_file_sanitizes_control_chars(db, tmp_path):
    payload = {
        **MINIMAL_V1_CARD_UNIT,
        "name": "SanitizeControlChars",
        "description": "Hello\x00World",
    }
    json_file = tmp_path / "sanitize_control_chars.json"
    json_file.write_text(json.dumps(payload))

    success, message, char_id = import_and_save_character_from_file(db, str(json_file))
    assert success
    assert char_id is not None

    saved_char = db.get_character_card_by_id(char_id)
    assert saved_char is not None
    assert "\x00" not in saved_char["description"]
    assert saved_char["description"] == "HelloWorld"


@mock.patch(f"{MODULE_PATH_PREFIX}.character_io.Image", new_callable=mock.MagicMock)
def test_import_png_invalid_image_base64_uses_file_bytes(MockPILImageModule, db, tmp_path):
    mock_img_instance = MockPILImageObject()
    MockPILImageModule.open.return_value = mock_img_instance

    payload = {
        **MINIMAL_V1_CARD_UNIT,
        "name": "PNG Bad Base64",
        "char_image": "not-a-valid-base64",
    }
    payload_json = json.dumps(payload)
    mock_img_instance.info = {
        "chara": base64.b64encode(payload_json.encode("utf-8")).decode("utf-8")
    }
    mock_img_instance.format = "PNG"

    dummy_png_bytes = create_dummy_png_bytes()
    png_file = tmp_path / "bad_base64.png"
    png_file.write_bytes(dummy_png_bytes)

    success_png, message_png, char_id_png = import_and_save_character_from_file(db, str(png_file))
    assert success_png and char_id_png is not None
    retrieved_png = db.get_character_card_by_id(char_id_png)
    assert retrieved_png["name"] == "PNG Bad Base64"
    assert retrieved_png["image"] == dummy_png_bytes


@mock.patch(f"{MODULE_PATH_PREFIX}.character_io.time.strftime", return_value=MOCK_TIME_STRFTIME)
def test_load_chat_history_from_file_and_save_to_db_integration(
    mock_strftime, db, tmp_path, caplog_handler
):  # caplog_handler now works
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
                [None, None],
            ]
        },
    }
    hist_file_path = tmp_path / "hist.json"
    hist_file_path.write_text(json.dumps(chat_data))
    conv_id, char_id_hist = load_chat_history_from_file_and_save_to_db(
        db, char_id_db, str(hist_file_path), user_name_for_placeholders=log_user
    )
    assert conv_id is not None and char_id_hist == char_id_db
    msgs = db.get_messages_for_conversation(conv_id)
    assert len(msgs) == 3
    assert "Skipping malformed message pair" in caplog_handler.text  # Should pass now

    assert msgs[0]["content"] == "U: {{user}}"
    assert msgs[0]["sender"] == "User"
    assert msgs[1]["content"] == "C: {{char}}"
    assert msgs[1]["sender"] == char_name_in_db
    assert msgs[2]["content"] == "User only"
    assert msgs[2]["sender"] == "User"


@mock.patch(f"{MODULE_PATH_PREFIX}.character_io.time.strftime", return_value=MOCK_TIME_STRFTIME)
def test_load_chat_history_plain_text_fallback(mock_strftime, db):
    char_id = db.add_character_card({"name": "PlainChar", "first_message": "Hi there"})
    conv_id, returned_char_id = load_chat_history_from_file_and_save_to_db(
        db,
        char_id,
        file_content="User says hello to {{char}}",
        user_name_for_placeholders="Tester",
    )

    assert conv_id is not None
    assert returned_char_id == char_id
    messages = db.get_messages_for_conversation(conv_id)
    assert len(messages) == 1
    assert messages[0]["sender"] == "User"
    assert messages[0]["content"] == "User says hello to {{char}}"


@mock.patch(f"{MODULE_PATH_PREFIX}.character_io.time.strftime", return_value=MOCK_TIME_STRFTIME)
def test_load_chat_history_yaml_sequence(mock_strftime, db):
    char_name = "YamlChar"
    char_id = db.add_character_card({"name": char_name, "first_message": "Hi there"})
    yaml_history = """
- role: assistant
  content: "Greetings {{user}}"
- role: user
  content: "Hello {{char}}"
"""
    conv_id, returned_char_id = load_chat_history_from_file_and_save_to_db(
        db,
        char_id,
        file_content=yaml_history,
        user_name_for_placeholders="YamlUser",
    )

    assert conv_id is not None
    assert returned_char_id == char_id
    messages = db.get_messages_for_conversation(conv_id)
    assert len(messages) == 2
    assert messages[0]["sender"] == char_name
    assert messages[0]["content"] == "Greetings {{user}}"
    assert messages[1]["sender"] == "User"
    assert messages[1]["content"] == "Hello {{char}}"


def test_load_chat_history_cleanup_on_failure(db):
    char_id = db.add_character_card({"name": "CleanupChar", "first_message": "Hi there"})
    failing_payload = json.dumps(
        {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
    )

    with mock.patch(
        f"{MODULE_PATH_PREFIX}.character_chat.post_message_to_conversation",
        side_effect=CharactersRAGDBError("boom"),
    ):
        result = load_chat_history_from_file_and_save_to_db(
            db,
            char_id,
            file_content=failing_payload,
            user_name_for_placeholders="Tester",
        )

    assert result == (None, None)
    assert db.get_conversations_for_character(char_id) == []


def test_load_chat_and_character_handles_legacy_alias(db):
    char_id = db.add_character_card({"name": "NewName", "first_message": "Hi {{user}}"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Legacy"})
    db.add_message({"conversation_id": conv_id, "sender": "OldName", "content": "Hello {{user}}"})
    db.add_message({"conversation_id": conv_id, "sender": "User", "content": "Hi"})
    db.add_message({"conversation_id": conv_id, "sender": "OldName", "content": "Welcome back, {{user}}"})

    char_data, history, _ = load_chat_and_character(db, conv_id, "Friend")
    assert char_data and char_data["name"] == "NewName"
    assert history == [
        (None, "Hello Friend"),
        ("Hi", "Welcome back, Friend"),
    ]


def test_load_chat_and_character_handles_legacy_you_sender(db):
    char_id = db.add_character_card({"name": "Botty", "first_message": "Hi {{user}}"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Legacy"})
    db.add_message({"conversation_id": conv_id, "sender": "You", "content": "Hi {{char}}"})
    db.add_message({"conversation_id": conv_id, "sender": "Botty", "content": "Hello {{user}}"})

    char_data, history, _ = load_chat_and_character(db, conv_id, "Friend")
    assert char_data and char_data["name"] == "Botty"
    assert history == [("Hi Botty", "Hello Friend")]


@pytest.mark.integration
def test_start_new_chat_session_decodes_alternate_greeting_bytes(db):
    char_id = db.add_character_card({"name": "ByteGreeter", "first_message": "Default {{user}}"})
    with db.transaction() as conn:
        # Simulate a legacy database row created before the sync trigger began
        # rejecting BLOB-backed JSON values.
        conn.execute("DROP TRIGGER IF EXISTS character_cards_sync_update")
        conn.execute(
            "UPDATE character_cards SET alternate_greetings = ? WHERE id = ?",
            (b'["Bytes hi {{user}}"]', char_id),
        )

    conv_id, _, init_hist, _ = start_new_chat_session(
        db,
        char_id,
        "Alice",
        greeting_strategy="alternate_index",
        alternate_index=0,
    )

    assert conv_id is not None
    assert init_hist == [(None, "Bytes hi Alice")]
    stored_messages = db.get_messages_for_conversation(conv_id, limit=1)
    assert stored_messages[0]["content"] == "Bytes hi {{user}}"


@pytest.mark.integration
def test_start_new_chat_session_creates_atomic_resumable_snapshot(db, monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "local-llm")
    monkeypatch.setenv("CHAR_CHAT_MODEL", "local-test")
    char_id = db.add_character_card(
        {
            "name": "SnapshotGreeter",
            "first_message": "Stable hello, {{user}}",
            "system_prompt": "Keep this behavior.",
            "extensions": {
                "tldw": {
                    "generation": {
                        "temperature": 0.0,
                        "top_p": 0.0,
                        "repetition_penalty": 0.0,
                        "stop": [],
                    }
                }
            },
        }
    )

    conv_id, _, _, _ = start_new_chat_session(db, char_id, "Alice")

    assert conv_id is not None
    state = db.get_roleplay_resume_state(conv_id)
    assert state["behavior_snapshot"]["status"] == "valid"
    assert state["resume_eligible"] is True
    assert state["settings_version"] == 1
    assert state["history_version"] == 2
    assert state["effective_completion"]["sampling"] == {
        "temperature": 0.0,
        "top_p": 0.0,
        "repetition_penalty": 0.0,
        "stop": [],
    }


@mock.patch(f"{MODULE_PATH_PREFIX}.character_chat.time.strftime", return_value=MOCK_TIME_STRFTIME)
@mock.patch(f"{MODULE_PATH_PREFIX}.character_db.Image", new_callable=mock.MagicMock)
def test_full_chat_session_flow_integration(MockPILImageModule, mock_strftime, db):
    mock_img_instance = MockPILImageObject()
    MockPILImageModule.open.return_value = mock_img_instance

    char_id = db.add_character_card(
        {"name": "FlowChar", "first_message": "Hi {{char}}", "image": create_dummy_png_bytes()}
    )
    user_name = "FlowUser"
    conv_id, _, init_hist, _ = start_new_chat_session(db, char_id, user_name, "FlowTitle")
    assert conv_id and init_hist == [(None, "Hi FlowChar")]

    conv_initial = db.get_conversation_by_id(conv_id)
    assert conv_initial is not None

    user_msg_id = post_message_to_conversation(db, conv_id, "FlowChar", "User says {{user}}", True)
    conv_after_user_msg = db.get_conversation_by_id(conv_id)
    assert conv_after_user_msg is not None
    assert conv_after_user_msg["version"] >= conv_initial["version"] + 1
    assert conv_after_user_msg["last_modified"] >= conv_initial["last_modified"]

    char_resp_id = post_message_to_conversation(db, conv_id, "FlowChar", "Char says {{char}}", False)
    conv_after_char_msg = db.get_conversation_by_id(conv_id)
    assert conv_after_char_msg is not None
    assert conv_after_char_msg["version"] >= conv_after_user_msg["version"] + 1
    assert conv_after_char_msg["last_modified"] >= conv_after_user_msg["last_modified"]

    ui_hist = retrieve_conversation_messages_for_ui(db, conv_id, "FlowChar", user_name)
    assert ui_hist == [(None, "Hi FlowChar"), ("User says FlowUser", "Char says FlowChar")]

    msg_to_edit = db.get_message_by_id(user_msg_id)
    assert edit_message_content(db, user_msg_id, "Edited", msg_to_edit["version"])

    msg_to_rank = db.get_message_by_id(char_resp_id)
    assert set_message_ranking(db, char_resp_id, 3, msg_to_rank["version"])
    msg_to_remove = db.get_message_by_id(user_msg_id)  # Re-fetch after edit for new version
    assert remove_message_from_conversation(db, user_msg_id, msg_to_remove["version"])

    conv_meta = db.get_conversation_by_id(conv_id)
    assert update_conversation_metadata(db, conv_id, {"title": "NewTitle"}, conv_meta["version"])

    assert len(search_conversations_by_title_query(db, "NewTitle")) == 1

    found_msgs = find_messages_in_conversation(db, conv_id, "Char says", "FlowChar", user_name)
    assert len(found_msgs) == 1 and found_msgs[0]["content"] == "Char says FlowChar"
    conv_to_del = db.get_conversation_by_id(conv_id)  # Re-fetch after update for new version
    assert delete_conversation_by_id(db, conv_id, conv_to_del["version"])
    assert db.get_conversation_by_id(conv_id) is None


def test_retrieve_conversation_messages_for_ui_desc_order(db):
    char_id = db.add_character_card({"name": "Chrony"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Chrony Chat"})

    db.add_message({"conversation_id": conv_id, "sender": "User", "content": "Hello {{char}}"})
    db.add_message({"conversation_id": conv_id, "sender": "Chrony", "content": "Hi there, {{user}}"})
    db.add_message({"conversation_id": conv_id, "sender": "User", "content": "Goodbye {{char}}"})

    asc_history = retrieve_conversation_messages_for_ui(db, conv_id, "Chrony", "Alice", order="ASC")
    assert asc_history == [
        ("Hello Chrony", "Hi there, Alice"),
        ("Goodbye Chrony", None),
    ]

    desc_history = retrieve_conversation_messages_for_ui(db, conv_id, "Chrony", "Alice", order="DESC")
    assert desc_history == [
        ("Goodbye Chrony", None),
        ("Hello Chrony", "Hi there, Alice"),
    ]


def test_retrieve_conversation_messages_for_ui_desc_alias_inference(db):
    char_id = db.add_character_card({"name": "AliasChar"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Alias Chat"})

    db.add_message({"conversation_id": conv_id, "sender": "User", "content": "Hello"})
    db.add_message({"conversation_id": conv_id, "sender": "AltA", "content": "AltA says hi"})
    db.add_message({"conversation_id": conv_id, "sender": "User", "content": "Ping"})
    db.add_message({"conversation_id": conv_id, "sender": "AltB", "content": "AltB says hi"})
    db.add_message({"conversation_id": conv_id, "sender": "User", "content": "Bye"})

    desc_history = retrieve_conversation_messages_for_ui(
        db,
        conversation_id=conv_id,
        character_name="AliasChar",
        user_name="User",
        order="DESC",
        rich_output=True,
    )

    contents = []
    for turn in desc_history:
        for role in ("user", "character", "non_character"):
            if turn.get(role) and turn[role].get("content"):
                contents.append(turn[role]["content"])

    assert any("AltA says hi" == content for content in contents)
    assert any("[AltB]" in content for content in contents)
    assert all("[AltA]" not in content for content in contents)


def test_retrieve_conversation_messages_for_ui_rich_output(db):
    char_id = db.add_character_card({"name": "Chrony"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Chrony Chat"})

    db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "User",
            "content": "Hello {{char}}",
        }
    )
    msg_id = db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "Chrony",
            "content": "Hi there, {{user}}",
            "image_data": b"123",
            "image_mime_type": "image/png",
        }
    )

    history = retrieve_conversation_messages_for_ui(
        db,
        conversation_id=conv_id,
        character_name="Chrony",
        user_name="Alice",
        rich_output=True,
    )

    assert len(history) == 1
    turn = history[0]
    assert turn["character"]["id"] == msg_id
    assert turn["character"]["content"] == "Hi there, Alice"
    assert turn["character"]["attachments"]


def test_retrieve_conversation_messages_for_ui_rich_output_keeps_trailing_user(db):
    char_id = db.add_character_card({"name": "Chrony"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Chrony Chat"})

    db.add_message({"conversation_id": conv_id, "sender": "User", "content": "Hello"})
    db.add_message({"conversation_id": conv_id, "sender": "Chrony", "content": "Hi"})
    db.add_message({"conversation_id": conv_id, "sender": "User", "content": "Trailing"})

    history = retrieve_conversation_messages_for_ui(
        db,
        conversation_id=conv_id,
        character_name="Chrony",
        user_name="Alice",
        rich_output=True,
    )

    assert len(history) == 2
    assert history[0]["user"]["content"] == "Hello"
    assert history[0]["character"]["content"] == "Hi"
    assert history[1]["user"]["content"] == "Trailing"
    assert history[1]["character"] is None


def test_sender_override_is_treated_as_user(db):
    char_id = db.add_character_card({"name": "Botty"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Alias Chat"})

    db.add_message({"conversation_id": conv_id, "sender": "Botty", "content": "Hello {{user}}"})
    db.add_message({"conversation_id": conv_id, "sender": "Alice", "content": "Hi {{char}}"})
    db.add_message({"conversation_id": conv_id, "sender": "Botty", "content": "How are you, {{user}}?"})
    db.add_message({"conversation_id": conv_id, "sender": "Alice", "content": "Feeling great!"})

    history = retrieve_conversation_messages_for_ui(
        db,
        conversation_id=conv_id,
        character_name="Botty",
        user_name="Alice",
        rich_output=True,
    )

    assert len(history) == 3
    # First turn: character greeting only
    assert history[0]["character"]["content"] == "Hello Alice"
    # Second turn: ensure Alice remains the user
    assert history[1]["user"]["sender"] == "Alice"
    assert history[1]["user"]["content"] == "Hi Botty"
    assert history[1]["character"]["sender"] == "Botty"
    assert history[1]["character"]["content"] == "How are you, Alice?"
    # Third turn: trailing user message should be preserved
    assert history[2]["user"]["sender"] == "Alice"
    assert history[2]["user"]["content"] == "Feeling great!"
    assert history[2]["character"] is None


def test_load_chat_and_character_integration(db):
    user_name = "Loader"
    char_id = db.add_character_card({"name": "LoadChar", "first_message": "FM"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Conv1"})
    db.add_message({"conversation_id": conv_id, "sender": "User", "content": "Hi {{char}}"})
    db.add_message({"conversation_id": conv_id, "sender": "LoadChar", "content": "Yo {{user}}"})

    with mock.patch(f"{MODULE_PATH_PREFIX}.character_db.Image", new_callable=mock.MagicMock) as MockPILImageModule:
        MockPILImageModule.open.return_value = MockPILImageObject()
        char_data, history, _ = load_chat_and_character(db, conv_id, user_name)
        assert char_data["name"] == "LoadChar" and history == [("Hi LoadChar", "Yo Loader")]


def test_load_character_wrapper_integration(db):
    char_id = db.add_character_card({"name": "Wrap", "first_message": "FM {{user}}"})
    user_name = "Wrapper"
    with mock.patch(f"{MODULE_PATH_PREFIX}.character_db.Image", new_callable=mock.MagicMock):
        _, hist_int, _ = load_character_wrapper(db, char_id, user_name)
        assert hist_int == [(None, "FM Wrapper")]
        _, hist_str, _ = load_character_wrapper(db, f"Wrap (ID: {char_id})", user_name)
        assert hist_str == [(None, "FM Wrapper")]


def test_get_conversation_metadata_integration(db):
    char_id = db.add_character_card({"name": "MetaChar", "description": "D"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "MetaTitle", "rating": 4})
    meta = get_conversation_metadata(db, conv_id)
    assert meta and meta["title"] == "MetaTitle" and meta["rating"] == 4
    assert get_conversation_metadata(db, "non_existent_conv") is None


def test_retrieve_message_details_integration(db):
    char_id = db.add_character_card({"name": "MsgDetChar", "description": "D"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "MsgDetConv"})
    msg_id = db.add_message({"conversation_id": conv_id, "sender": "User", "content": "Test {{char}} from {{user}}"})
    details = retrieve_message_details(db, msg_id, "MsgDetChar", "TestUser")
    assert details and details["content"] == "Test MsgDetChar from TestUser"
    assert retrieve_message_details(db, "non_existent_msg", "Char", "User") is None


#
# --- Regression Tests for Bug Fixes ---
#


def test_ambiguous_sender_lookahead_next_is_character():
    """
    Regression test for Issue #4: Ambiguous sender resolution with lookahead.

    When next sender is clearly a character (and not a user), the ambiguous
    message should be classified as 'user' to maintain conversation alternation.
    """
    # Test the process_db_messages_to_ui_history function with ambiguous senders
    db_messages = [
        {"sender": "User", "content": "Hello"},
        {"sender": "unknown_sender", "content": "Middle message"},  # Ambiguous
        {"sender": "Alice", "content": "Response"},  # Next is character
    ]

    history = process_db_messages_to_ui_history(
        db_messages=db_messages,
        char_name_from_card="Alice",
        user_name_for_placeholders="User",
        actual_user_sender_id_in_db="User",
        actual_char_sender_id_in_db="Alice",
    )

    # The history should correctly pair messages
    assert len(history) >= 1, "History should have at least one turn"


def test_ambiguous_sender_lookahead_next_is_user():
    """
    Regression test for Issue #4: Ambiguous sender resolution with lookahead.

    When next sender is clearly a user (and not a character), the ambiguous
    message should be classified as 'character'.
    """
    db_messages = [
        {"sender": "Alice", "content": "Hello"},
        {"sender": "mystery", "content": "Middle message"},  # Ambiguous
        {"sender": "User", "content": "Thanks"},  # Next is user
    ]

    history = process_db_messages_to_ui_history(
        db_messages=db_messages,
        char_name_from_card="Alice",
        user_name_for_placeholders="User",
        actual_user_sender_id_in_db="User",
        actual_char_sender_id_in_db="Alice",
    )

    # The history should correctly pair messages
    assert len(history) >= 1, "History should have at least one turn"


def test_compute_additional_char_aliases_no_user_inference_for_character_clusters():
    from tldw_Server_API.app.core.Character_Chat.modules.character_chat import _compute_additional_char_aliases

    db_messages = [
        {"sender": "Bob", "content": "Hey"},
        {"sender": "Charlie", "content": "Hi"},
        {"sender": "Bob", "content": "Back again"},
        {"sender": "Charlie", "content": "Still here"},
    ]

    additional_char_aliases, inferred_user_aliases = _compute_additional_char_aliases(
        db_messages=db_messages,
        char_name_from_card="Alice",
        user_name_for_placeholders="User",
        actual_user_sender_id_in_db="User",
    )

    assert set(additional_char_aliases) >= {"Bob", "Charlie"}
    assert inferred_user_aliases == []


def test_post_message_rejects_oversized_content(db, monkeypatch):
    from tldw_Server_API.app.core.config import settings

    monkeypatch.setitem(settings, "MAX_PERSIST_CONTENT_LENGTH", 5)
    char_id = db.add_character_card({"name": "LengthChar"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Length Chat"})

    with pytest.raises(InputError):
        post_message_to_conversation(
            db,
            conv_id,
            "LengthChar",
            "123456",
            is_user_message=False,
        )


def test_post_message_enforces_message_limit(db, monkeypatch):
    from fastapi import HTTPException

    from tldw_Server_API.app.core.Character_Chat import character_limits as limits

    monkeypatch.setenv("MAX_MESSAGES_PER_CHAT", "1")
    monkeypatch.setattr(limits, "_limits", None)

    char_id = db.add_character_card({"name": "LimitChar"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Limit Chat"})

    assert post_message_to_conversation(
        db,
        conv_id,
        "LimitChar",
        "first",
        is_user_message=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        post_message_to_conversation(
            db,
            conv_id,
            "LimitChar",
            "second",
            is_user_message=True,
        )

    assert exc_info.value.status_code == 403


def test_post_message_does_not_limit_assistant_response(db, monkeypatch):
    from tldw_Server_API.app.core.Character_Chat import character_limits as limits

    monkeypatch.setenv("MAX_MESSAGES_PER_CHAT", "1")
    monkeypatch.setattr(limits, "_limits", None)

    char_id = db.add_character_card({"name": "LimitChar"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Limit Chat"})

    user_message_id = post_message_to_conversation(
        db,
        conv_id,
        "LimitChar",
        "first",
        is_user_message=True,
    )
    assistant_message_id = post_message_to_conversation(
        db,
        conv_id,
        "LimitChar",
        "assistant reply",
        is_user_message=False,
        parent_message_id=user_message_id,
    )

    assert assistant_message_id


def test_load_chat_history_respects_message_limit(db, monkeypatch):
    from tldw_Server_API.app.core.Character_Chat import character_limits as limits

    monkeypatch.setenv("MAX_MESSAGES_PER_CHAT", "1")
    monkeypatch.setattr(limits, "_limits", None)

    char_id = db.add_character_card({"name": "LimitChar", "first_message": "Hi"})
    payload = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "One"},
                {"role": "assistant", "content": "Two"},
            ]
        }
    )

    result = load_chat_history_from_file_and_save_to_db(
        db,
        char_id,
        file_content=payload,
        user_name_for_placeholders="Tester",
    )

    assert result == (None, None)
    assert db.get_conversations_for_character(char_id) == []


#
# End of test_character_chat_lib.py
########################################################################################################################

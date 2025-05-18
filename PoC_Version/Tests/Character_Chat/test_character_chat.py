# test_character_chat_lib.py
import base64
import os
import sys

import pytest
from unittest.mock import patch, MagicMock, mock_open

from PIL.PngImagePlugin import PngInfo

#
# Adjust the path to the parent directory of App_Function_Libraries
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)

# Local Libraries
from PoC_Version.App_Function_Libraries.Character_Chat.Character_Chat_Lib import *
#
########################################################################################################################
#
# Sample data for testing
SAMPLE_CHAR_DATA = {
    'name': 'TestBot',
    'first_mes': 'Hello {{user}}!',
    'mes_example': 'Example {{char}} message',
    'scenario': 'Scenario with {{user}} and {{char}}',
    'description': 'Description for {{char}}',
    'personality': '{{char}} is friendly',
    'image': None
}

SAMPLE_CHAR_DATA_WITH_IMAGE = {
    **SAMPLE_CHAR_DATA,
    'image': base64.b64encode(b"fake_image_data").decode('utf-8')
}

VALID_V2_CARD = {
    "spec": "chara_card_v2",
    "spec_version": "2.0",
    "data": {
        "name": "Test Character",
        "description": "Test Description",
        "personality": "Test Personality",
        "scenario": "Test Scenario",
        "first_mes": "Hello!",
        "mes_example": "Test example",
        "character_book": {
            "entries": [
                {
                    "keys": ["test"],
                    "content": "test content",
                    "extensions": {},
                    "enabled": True,
                    "insertion_order": 1
                }
            ]
        }
    }
}


# Fixtures
@pytest.fixture
def mock_db():
    with patch('App_Function_Libraries.Character_Chat.Character_Chat_Lib.get_character_card_by_id') as mock_get_char, \
         patch('App_Function_Libraries.Character_Chat.Character_Chat_Lib.get_character_chat_by_id') as mock_get_chat:
        yield mock_get_char, mock_get_chat


@pytest.fixture
def mock_metrics():
    with patch('App_Function_Libraries.Metrics.metrics_logger.log_counter') as mock_log_counter, \
         patch('App_Function_Libraries.Metrics.metrics_logger.log_histogram') as mock_log_histogram:
        yield mock_log_counter, mock_log_histogram


@pytest.fixture
def sample_image_with_metadata():
    img = Image.new('RGBA', (1, 1), color=(0, 0, 0, 0))
    pnginfo = PngInfo()
    chara_data = base64.b64encode(json.dumps({'test': 'data'}).encode()).decode()
    pnginfo.add_text('chara', chara_data)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', pnginfo=pnginfo)
    buffer.seek(0)
    return buffer


# Unit Tests
def test_replace_placeholders():
    text = "{{char}} says hello to {{user}} and {{random_user}}"
    result = replace_placeholders(text, "Bot", "Alice")
    assert result == "Bot says hello to Alice and Alice"


def test_replace_user_placeholder():
    history = [("Hi {{user}}!", "Hello {{user}}"), (None, "Hey {{user}}")]
    result = replace_user_placeholder(history, "Alice")
    assert result == [("Hi Alice!", "Hello Alice"), (None, "Hey Alice")]


def test_extract_character_id_valid():
    assert extract_character_id("Name (ID: 123)") == 123


def test_extract_character_id_invalid():
    with pytest.raises(ValueError):
        extract_character_id("Invalid format")


def test_parse_character_book():
    book_data = {
        "entries": [{
            "keys": ["key1"],
            "content": "content",
            "extensions": {},
            "enabled": True,
            "insertion_order": 1
        }]
    }
    parsed = parse_character_book(book_data)
    assert len(parsed['entries']) == 1


def test_validate_v2_card_valid():
    valid, messages = validate_v2_card(VALID_V2_CARD)
    assert valid
    assert len(messages) == 0


def test_validate_v2_card_invalid_spec():
    invalid = {**VALID_V2_CARD, "spec": "invalid"}
    valid, messages = validate_v2_card(invalid)
    assert not valid
    assert "Invalid 'spec' value" in messages[0]


# Integration Tests
def test_load_character_and_image_success(mock_db, mock_metrics):
    mock_db[0].return_value = SAMPLE_CHAR_DATA
    char_data, history, img = load_character_and_image(123, "TestUser")

    assert char_data['first_mes'] == "Hello TestUser!"
    mock_db[0].assert_called_with(123)



def test_load_character_and_image_with_image(mock_db):
    mock_db[0].return_value = SAMPLE_CHAR_DATA_WITH_IMAGE
    _, _, img = load_character_and_image(123, "TestUser")
    assert img is None  # Since we used fake image data


def test_load_chat_and_character(mock_db):
    mock_chat = {
        'character_id': 123,
        'chat_history': [("Hi", "Hello")]
    }
    mock_db[1].return_value = mock_chat
    mock_db[0].return_value = SAMPLE_CHAR_DATA

    char_data, history, img = load_chat_and_character(456, "TestUser")
    assert char_data['name'] == "TestBot"
    assert len(history) == 1


def test_extract_json_from_image_metadata(sample_image_with_metadata):
    # Reset the buffer position before testing
    sample_image_with_metadata.seek(0)
    result = extract_json_from_image(sample_image_with_metadata)
    assert json.loads(result) == {'test': 'data'}


def test_process_chat_history():
    history = [("Hi {{user}}", "I'm {{char}}")]
    processed = process_chat_history(history, "Bot", "Alice")
    assert processed == [("Hi Alice", "I'm Bot")]


# Error Handling Tests
def test_load_character_and_image_error(mock_db, mock_metrics):
    mock_db[0].side_effect = Exception("DB error")
    result = load_character_and_image(123, "TestUser")
    assert result == (None, [], None)


# Validation Tests
def test_validate_character_book_entry_invalid():
    entry = {
        "keys": "not a list",
        "content": 123,
        "enabled": "not bool",
        "insertion_order": "invalid"
    }
    valid, messages = validate_character_book_entry(entry, 0, set())
    assert not valid
    assert len(messages) >= 3


def test_validate_character_book_duplicate_ids():
    book = {
        "entries": [
            {
                "keys": ["a"],
                "content": "a",
                "enabled": True,
                "insertion_order": 1,
                "id": 1,
                "extensions": {}
            },
            {
                "keys": ["b"],
                "content": "b",
                "enabled": True,
                "insertion_order": 2,
                "id": 1,
                "extensions": {}
            }
        ]
    }
    valid, messages = validate_character_book(book)
    assert not valid
    assert "Duplicate 'id' value" in messages[0]

#
# End of Tests
########################################################################################################################

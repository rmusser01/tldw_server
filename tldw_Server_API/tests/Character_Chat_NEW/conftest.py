"""
Character Chat Module Test Configuration and Fixtures

Provides fixtures for testing character chat functionality including
character cards, chat sessions, world books, dictionaries, and rate limiting.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional
from unittest.mock import MagicMock, AsyncMock, Mock, patch
from datetime import datetime, timedelta
import uuid
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

# Import actual character chat components for integration tests
# NOTE: CharacterChatManager doesn't exist - using individual services instead
# from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib import CharacterChatManager
from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import CharacterRateLimiter
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

# =====================================================================
# Test Markers
# =====================================================================

def pytest_configure(config):

    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "unit: Unit tests with minimal mocking")
    config.addinivalue_line("markers", "integration: Integration tests with real components")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1 second")
    config.addinivalue_line("markers", "rate_limit: Rate limiting tests")
    config.addinivalue_line("markers", "world_book: World book functionality tests")

    # Mark legacy/integration suites to skip at collection time
    config.addinivalue_line("markers", "legacy_skip: Skip tests targeting legacy behaviors")

# =====================================================================
# Environment Configuration
# =====================================================================

@pytest.fixture(autouse=True)
def stable_provider_override_snapshot():
    """Keep production override-cache TTLs from making long test runs order-dependent."""
    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides

    with llm_provider_overrides._OVERRIDE_LOCK:
        original_overrides = dict(llm_provider_overrides._OVERRIDE_CACHE)
        original_healthy = llm_provider_overrides._OVERRIDE_CACHE_HEALTHY
        original_ttl_disabled = (
            llm_provider_overrides._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
        )

    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})
    try:
        yield
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
            original_overrides,
            healthy=original_healthy,
            ttl_enabled=not original_ttl_disabled,
        )


@pytest.fixture(scope="session")
def test_env_vars():
    """Set up test environment variables."""
    original_env = os.environ.copy()

    # Set test mode
    os.environ["TEST_MODE"] = "true"
    os.environ["CHARACTER_DB_PATH"] = ":memory:"
    os.environ["MAX_CHAT_HISTORY"] = "100"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

# =====================================================================
# Database Fixtures
# =====================================================================

@pytest.fixture
def test_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file that gets cleaned up."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = Path(tmp_file.name)

    yield db_path

    # Cleanup
    try:
        if db_path.exists():
            db_path.unlink()
    except Exception as e:
        print(f"Warning: Could not delete test database: {e}")

@pytest.fixture
def character_db(test_db_path) -> Generator[CharactersRAGDB, None, None]:
    """Create a real CharactersRAGDB instance for testing."""
    # Use a temporary file database instead of in-memory to avoid threading issues
    # In-memory databases are not shared between connections in different threads
    db = CharactersRAGDB(
        db_path=str(test_db_path),
        client_id="test_client"
    )

    # Schema is initialized automatically in constructor

    yield db

    # Cleanup
    try:
        db.close_all_connections()
    except:
        _ = None

@pytest.fixture
def populated_character_db(character_db) -> CharactersRAGDB:
    """Create a CharactersRAGDB with test data."""
    db = character_db

    # Create test character cards using the actual database method
    char1_id = db.add_character_card({
        "name": "Test Character 1",
        "description": "A helpful test character",
        "personality": "Friendly and knowledgeable",
        "first_message": "Hello! I'm Test Character 1.",
        "creator": "test_user"
    })

    char2_id = db.add_character_card({
        "name": "Fantasy Wizard",
        "description": "A wise wizard from a fantasy world",
        "personality": "Mysterious and ancient",
        "first_message": "Greetings, traveler. What brings you to my tower?",
        "creator": "test_user"
    })

    char3_id = db.add_character_card({
        "name": "Science Assistant",
        "description": "An AI assistant specialized in science",
        "personality": "Analytical and precise",
        "first_message": "Hello! Ready to explore the world of science?",
        "creator": "science_user"
    })

    # Create test chats using the actual database method
    import uuid
    chat1_id = str(uuid.uuid4())
    chat2_id = str(uuid.uuid4())

    db.add_conversation({
        'id': chat1_id,
        'character_id': char1_id,
        'title': "First Chat",
        'root_id': chat1_id,
        'parent_id': None,
        'active': 1,
        'deleted': 0,
        'client_id': 'test_client',
        'version': 1
    })

    db.add_conversation({
        'id': chat2_id,
        'character_id': char2_id,
        'title': "Wizard Chat",
        'root_id': chat2_id,
        'parent_id': None,
        'active': 1,
        'deleted': 0,
        'client_id': 'test_client',
        'version': 1
    })

    # Add messages to chats using the actual database method
    import uuid
    msg1_id = str(uuid.uuid4())
    msg2_id = str(uuid.uuid4())

    db.add_message({
        'id': msg1_id,
        'conversation_id': chat1_id,
        'sender': 'user',
        'content': 'Hello!',
        'parent_message_id': None,
        'deleted': 0,
        'client_id': 'test_client',
        'version': 1
    })

    db.add_message({
        'id': msg2_id,
        'conversation_id': chat1_id,
        'sender': 'assistant',
        'content': 'Hello! How can I help you today?',
        'parent_message_id': msg1_id,
        'deleted': 0,
        'client_id': 'test_client',
        'version': 1
    })

    return db

# NOTE: CharacterChatManager doesn't exist - this fixture is disabled
# @pytest.fixture
# def chat_manager(test_db_path) -> Generator[CharacterChatManager, None, None]:
#     """Create a CharacterChatManager instance for testing."""
#     manager = CharacterChatManager(db_path=str(test_db_path))
#
#     yield manager
#
#     # Cleanup
#     try:
#         manager.close()
#     except:
#         pass

@pytest.fixture
def chat_dictionary_service(test_db_path) -> Generator[ChatDictionaryService, None, None]:
    """Create a ChatDictionaryService instance for testing."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(str(test_db_path), client_id="test_client")
    service = ChatDictionaryService(db)

    try:
        yield service
    finally:
        try:
            service.close()
        except Exception:
            _ = None
        try:
            db.close_all_connections()
        except Exception:
            _ = None

@pytest.fixture
def world_book_service(test_db_path) -> Generator[WorldBookService, None, None]:
    """Create a WorldBookService instance for testing."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(str(test_db_path), client_id="test_client")
    service = WorldBookService(db)

    try:
        yield service
    finally:
        try:
            service.close()
        except Exception:
            _ = None
        try:
            db.close_all_connections()
        except Exception:
            _ = None

@pytest.fixture
def mock_character_db():
    """Create a mock CharactersRAGDB for unit tests."""
    db = MagicMock(spec=CharactersRAGDB)

    # Mock initialization
    db.initialize_db = Mock(return_value=None)

    # Mock character methods
    db.add_character_card = Mock(return_value=1)  # Matches facade behaviour
    db.create_character_card = Mock(return_value=1)  # Keep for direct tests
    db.get_character_card = Mock(return_value={  # Some tests expect this generic method
        'id': 1,
        'name': 'Test Character',
        'description': 'Test description',
        'personality': 'Test personality',
        'first_message': 'Hello!',
        'created_at': datetime.utcnow().isoformat()
    })
    db.get_character_card_by_id = Mock(return_value={
        'id': 1,
        'name': 'Test Character',
        'description': 'Test description',
        'personality': 'Test personality',
        'first_message': 'Hello!',
        'created_at': datetime.utcnow().isoformat()
    })
    db.get_character_card_by_name = Mock(return_value=None)  # No conflicts by default
    db.list_character_cards = Mock(return_value=[])
    db.update_character_card = Mock(return_value=True)
    db.soft_delete_character_card = Mock(return_value=True)
    db.delete_character_card = Mock(return_value=True)

    # Mock chat methods
    db.create_chat = Mock(return_value=1)
    db.get_chat = Mock(return_value={
        'id': 1,
        'character_id': 1,
        'user_id': 'test_user',
        'title': 'Test Chat',
        'created_at': datetime.utcnow().isoformat()
    })
    db.list_chats = Mock(return_value=[])
    db.add_message = Mock(return_value=1)
    db.get_messages = Mock(return_value=[])
    db.delete_chat_session = Mock(return_value=True)

    return db

@pytest.fixture
def mock_chat_manager(mock_character_db):
    """Create a CharacterChatManager with mocked database for unit tests."""
    from tldw_Server_API.tests.Character_Chat_NEW.test_utils import CharacterChatManager

    with patch('tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB.CharactersRAGDB', return_value=mock_character_db):
        manager = CharacterChatManager(db_path=":memory:")
        manager.db = mock_character_db
        yield manager

# =====================================================================
# Character Card Fixtures
# =====================================================================

@pytest.fixture
def sample_character_card():
    """Sample character card data."""
    return {
        'name': 'Test Character',
        'description': 'A helpful AI assistant for testing',
        'personality': 'Friendly, helpful, and knowledgeable',
        'first_message': 'Hello! How can I assist you today?',
        'creator': 'test_user',
        'tags': ['test', 'assistant', 'AI'],
        'scenario': 'You are chatting with a helpful AI assistant.',
        'example_messages': [
            {'role': 'user', 'content': 'What can you do?'},
            {'role': 'assistant', 'content': 'I can help with various tasks!'}
        ]
    }

@pytest.fixture
def sample_character_cards():
    """Multiple sample character cards."""
    return [
        {
            'name': 'Science Teacher',
            'description': 'Expert in various sciences',
            'personality': 'Patient and thorough',
            'first_message': 'Welcome to science class!',
            'creator': 'educator',
            'tags': ['education', 'science']
        },
        {
            'name': 'Fantasy Guide',
            'description': 'Your guide through fantasy worlds',
            'personality': 'Adventurous and wise',
            'first_message': 'Greetings, adventurer!',
            'creator': 'game_master',
            'tags': ['fantasy', 'roleplay']
        },
        {
            'name': 'Code Helper',
            'description': 'Programming assistant',
            'personality': 'Logical and precise',
            'first_message': 'Ready to debug some code?',
            'creator': 'dev_user',
            'tags': ['programming', 'technical']
        }
    ]

@pytest.fixture
def character_card_v3_format():
    """Character card in V3 format for import/export."""
    return {
        'spec': 'chara_card_v3',
        'spec_version': '3.0',
        'data': {
            'name': 'Imported Character',
            'description': 'Character from V3 format',
            'personality': 'Standard V3 personality',
            'first_mes': 'Greetings from V3!',
            'mes_example': '<START>\n{{user}}: Hi\n{{char}}: Hello!\n',
            'scenario': 'V3 scenario',
            'creator_notes': 'Created for testing',
            'system_prompt': 'You are {{char}}',
            'post_history_instructions': 'Remember context',
            'alternate_greetings': ['Alt greeting 1', 'Alt greeting 2'],
            'tags': ['imported', 'v3']
        }
    }

# =====================================================================
# Chat Session Fixtures
# =====================================================================

@pytest.fixture
def sample_chat_session():
    """Sample chat session data."""
    return {
        'character_id': 1,
        'user_id': 'test_user',
        'title': 'Test Chat Session',
        'messages': [
            {'role': 'system', 'content': 'You are Test Character.'},
            {'role': 'assistant', 'content': 'Hello! How can I help?'},
            {'role': 'user', 'content': 'Tell me about yourself.'},
            {'role': 'assistant', 'content': 'I am a helpful AI assistant.'}
        ]
    }

@pytest.fixture
def long_chat_history():
    """Long chat history for testing context management."""
    messages = []
    for i in range(50):
        messages.extend([
            {'role': 'user', 'content': f'Question {i}?'},
            {'role': 'assistant', 'content': f'Answer {i}.'}
        ])
    return messages

# =====================================================================
# World Book Fixtures
# =====================================================================

@pytest.fixture
def sample_world_book():
    """Sample world book data."""
    return {
        'name': 'Fantasy World',
        'description': 'A comprehensive fantasy world setting',
        'entries': [
            {
                'keywords': ['dragon', 'dragons'],
                'content': 'Dragons are ancient magical creatures.',
                'priority': 100,
                'enabled': True
            },
            {
                'keywords': ['magic', 'spell'],
                'content': 'Magic flows through the world.',
                'priority': 90,
                'enabled': True
            },
            {
                'keywords': ['kingdom'],
                'content': 'The kingdom spans three continents.',
                'priority': 80,
                'enabled': True
            }
        ]
    }

@pytest.fixture
def complex_world_book():
    """Complex world book with recursive entries."""
    return {
        'name': 'Complex World',
        'description': 'World with recursive scanning',
        'entries': [
            {
                'keywords': ['hero'],
                'content': 'The hero wields a legendary sword.',
                'priority': 100,
                'enabled': True,
                'recursive_scanning': True
            },
            {
                'keywords': ['sword'],
                'content': 'The sword was forged by ancient smiths.',
                'priority': 95,
                'enabled': True
            },
            {
                'keywords': ['smiths', 'ancient'],
                'content': 'Ancient smiths lived in mountain forges.',
                'priority': 90,
                'enabled': True
            }
        ]
    }

# =====================================================================
# Chat Dictionary Fixtures
# =====================================================================

@pytest.fixture
def sample_dictionary():
    """Sample chat dictionary data."""
    return {
        'name': 'Test Dictionary',
        'description': 'Dictionary for text replacements',
        'entries': [
            {
                'pattern': 'AI',
                'replacement': 'Artificial Intelligence',
                'type': 'literal',
                'enabled': True
            },
            {
                'pattern': r'\b(\d+)\s*F\b',
                'replacement': r'\1 Fahrenheit',
                'type': 'regex',
                'enabled': True
            },
            {
                'pattern': 'lol',
                'replacement': 'laugh out loud',
                'type': 'literal',
                'probability': 0.5,
                'enabled': True
            }
        ]
    }

@pytest.fixture
def markdown_dictionary():
    """Dictionary in markdown format for import/export."""
    return """# Test Dictionary

## Entry: AI
- **Type**: literal
- **Replacement**: Artificial Intelligence
- **Enabled**: true

## Entry: Temperature
- **Type**: regex
- **Pattern**: \\b(\\d+)\\s*F\\b
- **Replacement**: \\1 Fahrenheit
- **Enabled**: true

## Entry: Slang
- **Type**: literal
- **Pattern**: lol
- **Replacement**: laugh out loud
- **Probability**: 50
- **Enabled**: true
"""

# =====================================================================
# Rate Limiting Fixtures
# =====================================================================

@pytest.fixture
def rate_limiter():
    """Create a rate limiter for testing."""
    return CharacterRateLimiter(
        max_requests_per_minute=30,
        max_tokens_per_minute=10000,
        burst_size=5
    )

@pytest.fixture
def rate_limit_config():
    """Rate limiter configuration."""
    return {
        'max_requests_per_minute': 30,
        'max_tokens_per_minute': 10000,
        'burst_size': 5,
        'window_size': 60  # seconds
    }

# =====================================================================
# Message Processing Fixtures
# =====================================================================

@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {'role': 'user', 'content': 'Hello!'},
        {'role': 'assistant', 'content': 'Hi there!'},
        {'role': 'user', 'content': 'How are you?'},
        {'role': 'assistant', 'content': 'I am doing well, thank you!'}
    ]

@pytest.fixture
def context_injection_data():
    """Data for testing context injection."""
    return {
        'world_book_entries': [
            'Dragons are ancient magical creatures.',
            'The kingdom spans three continents.'
        ],
        'dictionary_replacements': {
            'AI': 'Artificial Intelligence',
            'lol': 'laugh out loud'
        },
        'character_context': 'You are a helpful assistant.',
        'scenario': 'Modern day conversation.'
    }

# =====================================================================
# Search and Filter Fixtures
# =====================================================================

@pytest.fixture
def search_queries():
    """Various search query patterns."""
    return {
        'simple': 'test',
        'tags': 'tag:fantasy',
        'creator': 'creator:test_user',
        'combined': 'wizard tag:fantasy',
        'complex': '(science OR tech) AND creator:dev_user'
    }

@pytest.fixture
def filter_criteria():
    """Filter criteria for characters."""
    return {
        'by_tags': {'tags': ['fantasy', 'roleplay']},
        'by_creator': {'creator': 'test_user'},
        'by_date': {
            'created_after': '2024-01-01',
            'created_before': '2024-12-31'
        },
        'combined': {
            'tags': ['test'],
            'creator': 'test_user',
            'active': True
        }
    }

# =====================================================================
# Import/Export Fixtures
# =====================================================================

@pytest.fixture
def export_data():
    """Sample export data structure."""
    return {
        'version': '1.0',
        'exported_at': datetime.utcnow().isoformat(),
        'characters': [
            {
                'name': 'Exported Character 1',
                'description': 'First exported character',
                'personality': 'Friendly',
                'first_message': 'Hello!',
                'tags': ['export', 'test']
            },
            {
                'name': 'Exported Character 2',
                'description': 'Second exported character',
                'personality': 'Mysterious',
                'first_message': 'Greetings...',
                'tags': ['export', 'fantasy']
            }
        ],
        'world_books': [
            {
                'name': 'Exported World',
                'entries': [
                    {'keywords': ['test'], 'content': 'Test entry'}
                ]
            }
        ]
    }

@pytest.fixture
def import_file(export_data, tmp_path):
    """Create a temporary import file."""
    file_path = tmp_path / "character_import.json"
    file_path.write_text(json.dumps(export_data))
    return file_path

# =====================================================================
# Token Counting Fixtures
# =====================================================================

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode = Mock(side_effect=lambda text: list(text.split()))
    tokenizer.decode = Mock(side_effect=lambda tokens: ' '.join(tokens))
    tokenizer.count_tokens = Mock(side_effect=lambda text: len(text.split()))
    return tokenizer

# =====================================================================
# Performance Testing Fixtures
# =====================================================================

@pytest.fixture
def large_character_collection():
    """Generate a large collection of characters for performance testing."""
    characters = []
    for i in range(100):
        characters.append({
            'name': f'Character {i}',
            'description': f'Description for character {i}',
            'personality': f'Personality {i % 10}',
            'first_message': f'Hello from character {i}!',
            'creator': f'user_{i % 5}',
            'tags': [f'tag_{i % 3}', f'group_{i % 5}']
        })
    return characters

@pytest.fixture
def performance_metrics():
    """Track performance metrics during tests."""
    return {
        'create_times': [],
        'read_times': [],
        'search_times': [],
        'message_times': [],
        'context_times': []
    }

# Relax Character-Chat rate limits for this package to avoid flakiness.
@pytest.fixture(autouse=True)
def _override_character_chat_rate_limits_for_character_chat_new(monkeypatch):
    monkeypatch.setenv("CHARACTER_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("CHARACTER_RATE_LIMIT_OPS", "1000000")
    monkeypatch.setenv("CHARACTER_RATE_LIMIT_WINDOW", "60")
    monkeypatch.setenv("MAX_CHARACTERS_PER_USER", "1000000")
    monkeypatch.setenv("MAX_CHATS_PER_USER", "1000000")
    monkeypatch.setenv("MAX_MESSAGES_PER_CHAT", "1000000")
    monkeypatch.setenv("MAX_CHAT_COMPLETIONS_PER_MINUTE", "1000000")
    monkeypatch.setenv("MAX_MESSAGE_SENDS_PER_MINUTE", "1000000")
    try:
        from tldw_Server_API.app.core.Character_Chat import character_rate_limiter as _crl
        _crl._rate_limiter = None  # type: ignore[attr-defined]
    except Exception:
        _ = None
    yield
    try:
        from tldw_Server_API.app.core.Character_Chat import character_rate_limiter as _crl
        _crl._rate_limiter = None  # type: ignore[attr-defined]
    except Exception:
        _ = None

# =====================================================================
# API Client Fixtures
# =====================================================================

@pytest.fixture
def test_client(test_env_vars, character_db):
    """Create a test client for the FastAPI app with proper dependency overrides."""
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_user
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.config import settings as global_settings

    # Create test user
    test_user = User(
        id=1,
        username="test_user",
        email="test@example.com",
        is_admin=True,
        is_active=True
    )
    original_client_id = character_db.client_id
    character_db.client_id = str(test_user.id)

    # Override database dependency
    def override_get_chacha_db_for_user():
        """Override to return test database."""
        return character_db

    # Override user authentication for testing - bypass all auth
    async def override_get_request_user():
        """Override to return a test user."""
        return test_user

    async def override_get_current_user():
        """Override get_current_user to bypass authentication."""
        return test_user

    # Set up dependency overrides
    app.dependency_overrides[get_chacha_db_for_user] = override_get_chacha_db_for_user
    app.dependency_overrides[get_request_user] = override_get_request_user
    app.dependency_overrides[get_current_user] = override_get_current_user

    # Also try to override the settings to ensure consistent API key
    settings_instance = get_settings()
    original_api_key = settings_instance.SINGLE_USER_API_KEY
    settings_instance.SINGLE_USER_API_KEY = 'test-api-key'

    with TestClient(app) as client:
        yield client

    # Cleanup
    app.dependency_overrides.clear()

    # Restore API key
    settings_instance.SINGLE_USER_API_KEY = original_api_key
    character_db.client_id = original_client_id

@pytest.fixture
def auth_headers():
    """Authentication headers for API requests."""
    return {
        "X-API-KEY": "test-api-key",
        "Content-Type": "application/json"
    }

# =====================================================================
# Cleanup Fixtures
# =====================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Cleanup any temporary files or resources
    import gc
    gc.collect()

# =====================================================================
# Helper Functions
# =====================================================================

def create_test_character(db, **kwargs):

    """Helper to create a test character card."""
    character_data = {
        'name': 'Test Character',
        'description': 'Test description',
        'personality': 'Test personality',
        'first_message': 'Hello!',
        'creator': 'test_user'
    }
    character_data.update(kwargs)
    return db.add_character_card(character_data)

def create_test_chat(db, character_id, user_id='test_user', **kwargs):

    """Helper to create a test chat session."""
    import uuid
    chat_id = str(uuid.uuid4())

    db.add_conversation({
        'id': chat_id,
        'character_id': character_id,
        'title': kwargs.get('title', 'Test Chat'),
        'root_id': chat_id,
        'parent_id': None,
        'active': 1,
        'deleted': 0,
        'client_id': 'test_client',
        'version': 1
    })

    return chat_id

def create_test_world_book(service, **kwargs):

    """Helper to create a test world book."""
    world_book_data = {
        'name': 'Test World Book',
        'description': 'Test world book'
    }
    world_book_data.update(kwargs)
    return service.create_world_book(**world_book_data)

# character_chat_db.py
# Database functions for managing character cards and chat histories.
# #
# Imports
import configparser
import sqlite3
import json
import os
import sys
from typing import List, Dict, Optional, Tuple, Any, Union

from PoC_Version.App_Function_Libraries.Utils.Utils import get_database_dir, get_project_relative_path, get_database_path
from PoC_Version.Tests.Chat_APIs.Chat_APIs_Integration_test import logging

#
#######################################################################################################################
#
#

def ensure_database_directory():
    os.makedirs(get_database_dir(), exist_ok=True)

ensure_database_directory()


# Construct the path to the config file
config_path = get_project_relative_path('Config_Files/config.txt')

# Read the config file
config = configparser.ConfigParser()
config.read(config_path)

# Get the chat db path from the config, or use the default if not specified
chat_DB_PATH = config.get('Database', 'chatDB_path', fallback=get_database_path('chatDB.db'))
print(f"Chat Database path: {chat_DB_PATH}")

########################################################################################################
#
# Functions

# FIXME - Setup properly and test/add documentation for its existence...
def initialize_database():
    """Initialize the SQLite database with required tables and FTS5 virtual tables."""
    conn = None
    try:
        conn = sqlite3.connect(chat_DB_PATH)
        cursor = conn.cursor()

        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Create CharacterCards table with V2 fields
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS CharacterCards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            personality TEXT,
            scenario TEXT,
            image BLOB,
            post_history_instructions TEXT,
            first_mes TEXT,
            mes_example TEXT,
            creator_notes TEXT,
            system_prompt TEXT,
            alternate_greetings TEXT,
            tags TEXT,
            creator TEXT,
            character_version TEXT,
            extensions TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create FTS5 virtual table for CharacterCards
        cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS CharacterCards_fts USING fts5(
            name,
            description,
            personality,
            scenario,
            system_prompt,
            content='CharacterCards',
            content_rowid='id'
        );
        """)

        # Create triggers to keep FTS5 table in sync with CharacterCards
        cursor.executescript("""
        CREATE TRIGGER IF NOT EXISTS CharacterCards_ai AFTER INSERT ON CharacterCards BEGIN
            INSERT INTO CharacterCards_fts(
                rowid,
                name,
                description,
                personality,
                scenario,
                system_prompt
            ) VALUES (
                new.id,
                new.name,
                new.description,
                new.personality,
                new.scenario,
                new.system_prompt
            );
        END;

        CREATE TRIGGER IF NOT EXISTS CharacterCards_ad AFTER DELETE ON CharacterCards BEGIN
            DELETE FROM CharacterCards_fts WHERE rowid = old.id;
        END;

        CREATE TRIGGER IF NOT EXISTS CharacterCards_au AFTER UPDATE ON CharacterCards BEGIN
            UPDATE CharacterCards_fts SET
                name = new.name,
                description = new.description,
                personality = new.personality,
                scenario = new.scenario,
                system_prompt = new.system_prompt
            WHERE rowid = new.id;
        END;
        """)

        # Create CharacterChats table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS CharacterChats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            character_id INTEGER NOT NULL,
            conversation_name TEXT,
            chat_history TEXT,
            is_snapshot BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (character_id) REFERENCES CharacterCards(id) ON DELETE CASCADE
        );
        """)

        # Create FTS5 virtual table for CharacterChats
        cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS CharacterChats_fts USING fts5(
            conversation_name,
            chat_history,
            content='CharacterChats',
            content_rowid='id'
        );
        """)

        # Create triggers to keep FTS5 table in sync with CharacterChats
        cursor.executescript("""
        CREATE TRIGGER IF NOT EXISTS CharacterChats_ai AFTER INSERT ON CharacterChats BEGIN
            INSERT INTO CharacterChats_fts(rowid, conversation_name, chat_history)
            VALUES (new.id, new.conversation_name, new.chat_history);
        END;

        CREATE TRIGGER IF NOT EXISTS CharacterChats_ad AFTER DELETE ON CharacterChats BEGIN
            DELETE FROM CharacterChats_fts WHERE rowid = old.id;
        END;

        CREATE TRIGGER IF NOT EXISTS CharacterChats_au AFTER UPDATE ON CharacterChats BEGIN
            UPDATE CharacterChats_fts SET conversation_name = new.conversation_name, chat_history = new.chat_history
            WHERE rowid = new.id;
        END;
        """)

        # Create ChatKeywords table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ChatKeywords (
            chat_id INTEGER NOT NULL,
            keyword TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES CharacterChats(id) ON DELETE CASCADE
        );
        """)

        # Create indexes for faster searches
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chatkeywords_keyword ON ChatKeywords(keyword);
        """)
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chatkeywords_chat_id ON ChatKeywords(chat_id);
        """)

        conn.commit()
        logging.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logging.error(f"SQLite error occurred during database initialization: {e}")
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        logging.error(f"Unexpected error occurred during database initialization: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

# Call initialize_database() at the start of your application
def setup_chat_database():
    try:
        initialize_database()
    except Exception as e:
        logging.critical(f"Failed to initialize database: {e}")
        sys.exit(1)

setup_chat_database()


########################################################################################################
#
# Character Card handling

def parse_character_card(card_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate a character card according to V2 specification."""
    v2_data = {
        'name': card_data.get('name', ''),
        'description': card_data.get('description', ''),
        'personality': card_data.get('personality', ''),
        'scenario': card_data.get('scenario', ''),
        'first_mes': card_data.get('first_mes', ''),
        'mes_example': card_data.get('mes_example', ''),
        'creator_notes': card_data.get('creator_notes', ''),
        'system_prompt': card_data.get('system_prompt', ''),
        'post_history_instructions': card_data.get('post_history_instructions', ''),
        'alternate_greetings': json.dumps(card_data.get('alternate_greetings', [])),
        'tags': json.dumps(card_data.get('tags', [])),
        'creator': card_data.get('creator', ''),
        'character_version': card_data.get('character_version', ''),
        'extensions': json.dumps(card_data.get('extensions', {}))
    }

    # Handle 'image' separately as it might be binary data
    if 'image' in card_data:
        v2_data['image'] = card_data['image']

    return v2_data


def add_character_card(card_data: Dict[str, Any]) -> Optional[int]:
    """Add or update a character card in the database."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        parsed_card = parse_character_card(card_data)

        # Check if character already exists
        cursor.execute("SELECT id FROM CharacterCards WHERE name = ?", (parsed_card['name'],))
        row = cursor.fetchone()

        if row:
            # Update existing character
            character_id = row[0]
            update_query = """
                UPDATE CharacterCards
                SET description = ?, personality = ?, scenario = ?, image = ?, 
                    post_history_instructions = ?, first_mes = ?, mes_example = ?,
                    creator_notes = ?, system_prompt = ?, alternate_greetings = ?,
                    tags = ?, creator = ?, character_version = ?, extensions = ?
                WHERE id = ?
            """
            cursor.execute(update_query, (
                parsed_card['description'], parsed_card['personality'], parsed_card['scenario'],
                parsed_card['image'], parsed_card['post_history_instructions'], parsed_card['first_mes'],
                parsed_card['mes_example'], parsed_card['creator_notes'], parsed_card['system_prompt'],
                parsed_card['alternate_greetings'], parsed_card['tags'], parsed_card['creator'],
                parsed_card['character_version'], parsed_card['extensions'], character_id
            ))
        else:
            # Insert new character
            insert_query = """
                INSERT INTO CharacterCards (name, description, personality, scenario, image, 
                post_history_instructions, first_mes, mes_example, creator_notes, system_prompt, 
                alternate_greetings, tags, creator, character_version, extensions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(insert_query, (
                parsed_card['name'], parsed_card['description'], parsed_card['personality'],
                parsed_card['scenario'], parsed_card['image'], parsed_card['post_history_instructions'],
                parsed_card['first_mes'], parsed_card['mes_example'], parsed_card['creator_notes'],
                parsed_card['system_prompt'], parsed_card['alternate_greetings'], parsed_card['tags'],
                parsed_card['creator'], parsed_card['character_version'], parsed_card['extensions']
            ))
            character_id = cursor.lastrowid

        conn.commit()
        return character_id
    except sqlite3.IntegrityError as e:
        logging.error(f"Error adding character card: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error adding character card: {e}")
        return None
    finally:
        conn.close()

# def add_character_card(card_data: Dict) -> Optional[int]:
#     """Add or update a character card in the database.
#
#     Returns the ID of the inserted character or None if failed.
#     """
#     conn = sqlite3.connect(chat_DB_PATH)
#     cursor = conn.cursor()
#     try:
#         # Ensure all required fields are present
#         required_fields = ['name', 'description', 'personality', 'scenario', 'image', 'post_history_instructions', 'first_message']
#         for field in required_fields:
#             if field not in card_data:
#                 card_data[field] = ''  # Assign empty string if field is missing
#
#         # Check if character already exists
#         cursor.execute("SELECT id FROM CharacterCards WHERE name = ?", (card_data['name'],))
#         row = cursor.fetchone()
#
#         if row:
#             # Update existing character
#             character_id = row[0]
#             cursor.execute("""
#                 UPDATE CharacterCards
#                 SET description = ?, personality = ?, scenario = ?, image = ?, post_history_instructions = ?, first_message = ?
#                 WHERE id = ?
#             """, (
#                 card_data['description'],
#                 card_data['personality'],
#                 card_data['scenario'],
#                 card_data['image'],
#                 card_data['post_history_instructions'],
#                 card_data['first_message'],
#                 character_id
#             ))
#         else:
#             # Insert new character
#             cursor.execute("""
#                 INSERT INTO CharacterCards (name, description, personality, scenario, image, post_history_instructions, first_message)
#                 VALUES (?, ?, ?, ?, ?, ?, ?)
#             """, (
#                 card_data['name'],
#                 card_data['description'],
#                 card_data['personality'],
#                 card_data['scenario'],
#                 card_data['image'],
#                 card_data['post_history_instructions'],
#                 card_data['first_message']
#             ))
#             character_id = cursor.lastrowid
#
#         conn.commit()
#         return cursor.lastrowid
#     except sqlite3.IntegrityError as e:
#         logging.error(f"Error adding character card: {e}")
#         return None
#     except Exception as e:
#         logging.error(f"Unexpected error adding character card: {e}")
#         return None
#     finally:
#         conn.close()


def get_character_cards() -> List[Dict]:
    """Retrieve all character cards from the database and log their name and a preview of the description."""
    try:
        logging.debug(f"Fetching characters from DB: {chat_DB_PATH}")
        conn = sqlite3.connect(chat_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM CharacterCards")
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()

        # Convert rows to dictionaries
        characters = [dict(zip(columns, row)) for row in rows]

        # For each card, extract the card data, then log the name and first 50 characters of the description.
        for card in characters:
            # Assume that the V2 card details are stored in the 'data' field.
            # If stored as JSON, parse it.
            data = card.get('data')
            if data:
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except Exception as e:
                        logging.error(f"Error parsing card 'data' as JSON: {e}")
                        continue

                # Extract 'name' and 'description' from the card's data.
                name = data.get('name', 'Unnamed')
                description = data.get('description', 'No description provided.')
                # Log the name and the first 50 characters of the description.
                logging.info("Card: {name} - {description[:50]}")
            else:
                logging.warning("Card record missing 'data' field.")

        return characters
    except Exception as e:
        logging.error(f"Error fetching character cards: {e}")
        return []

def get_character_card_by_id(character_id: Union[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Retrieve a single character card by its ID.

    Args:
        character_id: Can be either an integer ID or a dictionary containing character data.

    Returns:
        A dictionary containing the character card data, or None if not found.
    """
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        if isinstance(character_id, dict):
            # If a dictionary is passed, assume it's already a character card
            return character_id
        elif isinstance(character_id, int):
            # If an integer is passed, fetch the character from the database
            cursor.execute("SELECT * FROM CharacterCards WHERE id = ?", (character_id,))
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
        else:
            logging.warning(f"Invalid type for character_id: {type(character_id)}")
        return None
    except Exception as e:
        logging.error(f"Error in get_character_card_by_id: {e}")
        return None
    finally:
        conn.close()


def update_character_card(character_id: int, card_data: Dict) -> bool:
    """Update an existing character card."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE CharacterCards
            SET name = ?, description = ?, personality = ?, scenario = ?, image = ?, post_history_instructions = ?, first_message = ?
            WHERE id = ?
        """, (
            card_data.get('name'),
            card_data.get('description'),
            card_data.get('personality'),
            card_data.get('scenario'),
            card_data.get('image'),
            card_data.get('post_history_instructions', ''),
            card_data.get('first_message', "Hello! I'm ready to chat."),
            character_id
        ))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.IntegrityError as e:
        logging.error(f"Error updating character card: {e}")
        return False
    finally:
        conn.close()


def delete_character_card(character_id: int) -> bool:
    """Delete a character card and its associated chats."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        # Delete associated chats first due to foreign key constraint
        cursor.execute("DELETE FROM CharacterChats WHERE character_id = ?", (character_id,))
        cursor.execute("DELETE FROM CharacterCards WHERE id = ?", (character_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logging.error(f"Error deleting character card: {e}")
        return False
    finally:
        conn.close()


def add_character_chat(character_id: int, conversation_name: str, chat_history: List[Tuple[str, str]], keywords: Optional[List[str]] = None, is_snapshot: bool = False) -> Optional[int]:
    """
    Add a new chat history for a character, optionally associating keywords.

    Args:
        character_id (int): The ID of the character.
        conversation_name (str): Name of the conversation.
        chat_history (List[Tuple[str, str]]): List of (user, bot) message tuples.
        keywords (Optional[List[str]]): List of keywords to associate with this chat.
        is_snapshot (bool, optional): Whether this chat is a snapshot.

    Returns:
        Optional[int]: The ID of the inserted chat or None if failed.
    """
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        chat_history_json = json.dumps(chat_history)
        cursor.execute("""
            INSERT INTO CharacterChats (character_id, conversation_name, chat_history, is_snapshot)
            VALUES (?, ?, ?, ?)
        """, (
            character_id,
            conversation_name,
            chat_history_json,
            is_snapshot
        ))
        chat_id = cursor.lastrowid

        if keywords:
            # Insert keywords into ChatKeywords table
            keyword_records = [(chat_id, keyword.strip().lower()) for keyword in keywords]
            cursor.executemany("""
                INSERT INTO ChatKeywords (chat_id, keyword)
                VALUES (?, ?)
            """, keyword_records)

        conn.commit()
        return chat_id
    except sqlite3.Error as e:
        logging.error(f"Error adding character chat: {e}")
        return None
    finally:
        conn.close()


def get_character_chats(character_id: Optional[int] = None) -> List[Dict]:
    """Retrieve all chats, or chats for a specific character if character_id is provided."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    if character_id is not None:
        cursor.execute("SELECT * FROM CharacterChats WHERE character_id = ?", (character_id,))
    else:
        cursor.execute("SELECT * FROM CharacterChats")
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


def get_character_chat_by_id(chat_id: int) -> Optional[Dict]:
    """Retrieve a single chat by its ID."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM CharacterChats WHERE id = ?", (chat_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        columns = [description[0] for description in cursor.description]
        chat = dict(zip(columns, row))
        chat['chat_history'] = json.loads(chat['chat_history'])
        return chat
    return None


def search_character_chats(query: str, character_id: Optional[int] = None) -> Tuple[List[Dict], str]:
    """
    Search for character chats using FTS5, optionally filtered by character_id.

    Args:
        query (str): The search query.
        character_id (Optional[int]): The ID of the character to filter chats by.

    Returns:
        Tuple[List[Dict], str]: A list of matching chats and a status message.
    """
    if not query.strip():
        return [], "Please enter a search query."

    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        if character_id is not None:
            # Search with character_id filter
            cursor.execute("""
                SELECT CharacterChats.id, CharacterChats.conversation_name, CharacterChats.chat_history
                FROM CharacterChats_fts
                JOIN CharacterChats ON CharacterChats_fts.rowid = CharacterChats.id
                WHERE CharacterChats_fts MATCH ? AND CharacterChats.character_id = ?
                ORDER BY rank
            """, (query, character_id))
        else:
            # Search without character_id filter
            cursor.execute("""
                SELECT CharacterChats.id, CharacterChats.conversation_name, CharacterChats.chat_history
                FROM CharacterChats_fts
                JOIN CharacterChats ON CharacterChats_fts.rowid = CharacterChats.id
                WHERE CharacterChats_fts MATCH ?
                ORDER BY rank
            """, (query,))

        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]

        if character_id is not None:
            status_message = f"Found {len(results)} chat(s) matching '{query}' for the selected character."
        else:
            status_message = f"Found {len(results)} chat(s) matching '{query}' across all characters."

        return results, status_message
    except Exception as e:
        logging.error(f"Error searching chats with FTS5: {e}")
        return [], f"Error occurred during search: {e}"
    finally:
        conn.close()

def update_character_chat(chat_id: int, chat_history: List[Tuple[str, str]]) -> bool:
    """Update an existing chat history."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        chat_history_json = json.dumps(chat_history)
        cursor.execute("""
            UPDATE CharacterChats
            SET chat_history = ?
            WHERE id = ?
        """, (
            chat_history_json,
            chat_id
        ))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logging.error(f"Error updating character chat: {e}")
        return False
    finally:
        conn.close()


def delete_character_chat(chat_id: int) -> bool:
    """Delete a specific chat."""
    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM CharacterChats WHERE id = ?", (chat_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logging.error(f"Error deleting character chat: {e}")
        return False
    finally:
        conn.close()


def fetch_keywords_for_chats(keywords: List[str]) -> List[int]:
    """
    Fetch chat IDs associated with any of the specified keywords.

    Args:
        keywords (List[str]): List of keywords to search for.

    Returns:
        List[int]: List of chat IDs associated with the keywords.
    """
    if not keywords:
        return []

    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        # Construct the WHERE clause to search for each keyword
        keyword_clauses = " OR ".join(["keyword = ?"] * len(keywords))
        sql_query = f"SELECT DISTINCT chat_id FROM ChatKeywords WHERE {keyword_clauses}"
        cursor.execute(sql_query, keywords)
        rows = cursor.fetchall()
        chat_ids = [row[0] for row in rows]
        return chat_ids
    except Exception as e:
        logging.error(f"Error in fetch_keywords_for_chats: {e}")
        return []
    finally:
        conn.close()


def save_chat_history_to_character_db(character_id: int, conversation_name: str, chat_history: List[Tuple[str, str]]) -> Optional[int]:
    """Save chat history to the CharacterChats table.

    Returns the ID of the inserted chat or None if failed.
    """
    return add_character_chat(character_id, conversation_name, chat_history)


def search_db(query: str, fields: List[str], where_clause: str = "", page: int = 1, results_per_page: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a full-text search on specified fields with optional filtering and pagination.

    Args:
        query (str): The search query.
        fields (List[str]): List of fields to search in.
        where_clause (str, optional): Additional SQL WHERE clause to filter results.
        page (int, optional): Page number for pagination.
        results_per_page (int, optional): Number of results per page.

    Returns:
        List[Dict[str, Any]]: List of matching chat records with content and metadata.
    """
    if not query.strip():
        return []

    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        # Construct the MATCH query for FTS5
        match_query = " AND ".join(fields) + f" MATCH ?"
        # Adjust the query with the fields
        fts_query = f"""
            SELECT CharacterChats.id, CharacterChats.conversation_name, CharacterChats.chat_history
            FROM CharacterChats_fts
            JOIN CharacterChats ON CharacterChats_fts.rowid = CharacterChats.id
            WHERE {match_query}
        """
        if where_clause:
            fts_query += f" AND ({where_clause})"
        fts_query += " ORDER BY rank LIMIT ? OFFSET ?"
        offset = (page - 1) * results_per_page
        cursor.execute(fts_query, (query, results_per_page, offset))
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]
        return results
    except Exception as e:
        logging.error(f"Error in search_db: {e}")
        return []
    finally:
        conn.close()


def perform_full_text_search_chat(query: str, relevant_chat_ids: List[int], page: int = 1, results_per_page: int = 5) -> \
List[Dict[str, Any]]:
    """
    Perform a full-text search within the specified chat IDs using FTS5.

    Args:
        query (str): The user's query.
        relevant_chat_ids (List[int]): List of chat IDs to search within.
        page (int): Pagination page number.
        results_per_page (int): Number of results per page.

    Returns:
        List[Dict[str, Any]]: List of search results with content and metadata.
    """
    try:
        # Construct a WHERE clause to limit the search to relevant chat IDs
        where_clause = " OR ".join([f"media_id = {chat_id}" for chat_id in relevant_chat_ids])
        if not where_clause:
            where_clause = "1"  # No restriction if no chat IDs

        # Perform full-text search using FTS5
        fts_results = search_db(query, ["content"], where_clause, page=page, results_per_page=results_per_page)

        filtered_fts_results = [
            {
                "content": result['content'],
                "metadata": {"media_id": result['id']}
            }
            for result in fts_results
            if result['id'] in relevant_chat_ids
        ]
        return filtered_fts_results
    except Exception as e:
        logging.error(f"Error in perform_full_text_search_chat: {str(e)}")
        return []


def fetch_all_chats() -> List[Dict[str, Any]]:
    """
    Fetch all chat messages from the database.

    Returns:
        List[Dict[str, Any]]: List of chat messages with relevant metadata.
    """
    try:
        chats = get_character_chats()  # Modify this function to retrieve all chats
        return chats
    except Exception as e:
        logging.error(f"Error fetching all chats: {str(e)}")
        return []


def search_character_chat(query: str, fts_top_k: int = 10, relevant_media_ids: List[str] = None) -> List[Dict[str, Any]]:
    """
    Perform a full-text search on the Character Chat database.

    Args:
        query: Search query string.
        fts_top_k: Maximum number of results to return.
        relevant_media_ids: Optional list of character IDs to filter results.

    Returns:
        List of search results with content and metadata.
    """
    if not query.strip():
        return []

    try:
        # Construct a WHERE clause to limit the search to relevant character IDs
        where_clause = ""
        if relevant_media_ids:
            placeholders = ','.join(['?'] * len(relevant_media_ids))
            where_clause = f"CharacterChats.character_id IN ({placeholders})"

        # Perform full-text search using existing search_db function
        results = search_db(query, ["conversation_name", "chat_history"], where_clause, results_per_page=fts_top_k)

        # Format results
        formatted_results = []
        for r in results:
            formatted_results.append({
                "content": r['chat_history'],
                "metadata": {
                    "chat_id": r['id'],
                    "conversation_name": r['conversation_name'],
                    "character_id": r['character_id']
                }
            })

        return formatted_results

    except Exception as e:
        logging.error(f"Error in search_character_chat: {e}")
        return []


def search_character_cards(query: str, fts_top_k: int = 10, relevant_media_ids: List[str] = None) -> List[Dict[str, Any]]:
    """
    Perform a full-text search on the Character Cards database.

    Args:
        query: Search query string.
        fts_top_k: Maximum number of results to return.
        relevant_media_ids: Optional list of character IDs to filter results.

    Returns:
        List of search results with content and metadata.
    """
    if not query.strip():
        return []

    try:
        conn = sqlite3.connect(chat_DB_PATH)
        cursor = conn.cursor()

        # Construct the query
        sql_query = """
            SELECT CharacterCards.id, CharacterCards.name, CharacterCards.description, CharacterCards.personality, CharacterCards.scenario
            FROM CharacterCards_fts
            JOIN CharacterCards ON CharacterCards_fts.rowid = CharacterCards.id
            WHERE CharacterCards_fts MATCH ?
        """

        params = [query]

        # Add filtering by character IDs if provided
        if relevant_media_ids:
            placeholders = ','.join(['?'] * len(relevant_media_ids))
            sql_query += f" AND CharacterCards.id IN ({placeholders})"
            params.extend(relevant_media_ids)

        sql_query += " LIMIT ?"
        params.append(fts_top_k)

        cursor.execute(sql_query, params)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]

        results = [dict(zip(columns, row)) for row in rows]

        # Format results
        formatted_results = []
        for r in results:
            content = f"Name: {r['name']}\nDescription: {r['description']}\nPersonality: {r['personality']}\nScenario: {r['scenario']}"
            formatted_results.append({
                "content": content,
                "metadata": {
                    "character_id": r['id'],
                    "name": r['name']
                }
            })

        return formatted_results

    except Exception as e:
        logging.error(f"Error in search_character_cards: {e}")
        return []
    finally:
        conn.close()


def fetch_character_ids_by_keywords(keywords: List[str]) -> List[int]:
    """
    Fetch character IDs associated with any of the specified keywords.

    Args:
        keywords (List[str]): List of keywords to search for.

    Returns:
        List[int]: List of character IDs associated with the keywords.
    """
    if not keywords:
        return []

    conn = sqlite3.connect(chat_DB_PATH)
    cursor = conn.cursor()
    try:
        # Assuming 'tags' column in CharacterCards table stores tags as JSON array
        placeholders = ','.join(['?'] * len(keywords))
        sql_query = f"""
            SELECT DISTINCT id FROM CharacterCards
            WHERE EXISTS (
                SELECT 1 FROM json_each(tags)
                WHERE json_each.value IN ({placeholders})
            )
        """
        cursor.execute(sql_query, keywords)
        rows = cursor.fetchall()
        character_ids = [row[0] for row in rows]
        return character_ids
    except Exception as e:
        logging.error(f"Error in fetch_character_ids_by_keywords: {e}")
        return []
    finally:
        conn.close()


###################################################################
#
# Character Keywords

def view_char_keywords():
    try:
        with sqlite3.connect(chat_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT keyword 
                FROM CharacterCards 
                CROSS JOIN json_each(tags) 
                WHERE json_valid(tags)
                ORDER BY keyword
            """)
            keywords = cursor.fetchall()
            if keywords:
                keyword_list = [k[0] for k in keywords]
                return "### Current Character Keywords:\n" + "\n".join(
                    [f"- {k}" for k in keyword_list])
            return "No keywords found."
    except Exception as e:
        return f"Error retrieving keywords: {str(e)}"


def add_char_keywords(name: str, keywords: str):
    try:
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        with sqlite3.connect('character_chat.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT tags FROM CharacterCards WHERE name = ?",
                (name,)
            )
            result = cursor.fetchone()
            if not result:
                return "Character not found."

            current_tags = result[0] if result[0] else "[]"
            current_keywords = set(current_tags[1:-1].split(',')) if current_tags != "[]" else set()
            updated_keywords = current_keywords.union(set(keywords_list))

            cursor.execute(
                "UPDATE CharacterCards SET tags = ? WHERE name = ?",
                (str(list(updated_keywords)), name)
            )
            conn.commit()
            return f"Successfully added keywords to character {name}"
    except Exception as e:
        return f"Error adding keywords: {str(e)}"


def delete_char_keyword(char_name: str, keyword: str) -> str:
    """
    Delete a keyword from a character's tags.

    Args:
        char_name (str): The name of the character
        keyword (str): The keyword to delete

    Returns:
        str: Success/failure message
    """
    try:
        with sqlite3.connect(chat_DB_PATH) as conn:
            cursor = conn.cursor()

            # First, check if the character exists
            cursor.execute("SELECT tags FROM CharacterCards WHERE name = ?", (char_name,))
            result = cursor.fetchone()

            if not result:
                return f"Character '{char_name}' not found."

            # Parse existing tags
            current_tags = json.loads(result[0]) if result[0] else []

            if keyword not in current_tags:
                return f"Keyword '{keyword}' not found in character '{char_name}' tags."

            # Remove the keyword
            updated_tags = [tag for tag in current_tags if tag != keyword]

            # Update the character's tags
            cursor.execute(
                "UPDATE CharacterCards SET tags = ? WHERE name = ?",
                (json.dumps(updated_tags), char_name)
            )
            conn.commit()

            logging.info(f"Keyword '{keyword}' deleted from character '{char_name}'")
            return f"Successfully deleted keyword '{keyword}' from character '{char_name}'."

    except Exception as e:
        error_msg = f"Error deleting keyword: {str(e)}"
        logging.error(error_msg)
        return error_msg


def export_char_keywords_to_csv() -> Tuple[str, str]:
    """
    Export all character keywords to a CSV file with associated metadata.

    Returns:
        Tuple[str, str]: (status_message, file_path)
    """
    import csv
    from tempfile import NamedTemporaryFile
    from datetime import datetime

    try:
        # Create a temporary CSV file
        temp_file = NamedTemporaryFile(mode='w+', delete=False, suffix='.csv', newline='')

        with sqlite3.connect(chat_DB_PATH) as conn:
            cursor = conn.cursor()

            # Get all characters and their tags
            cursor.execute("""
                SELECT 
                    name,
                    tags,
                    (SELECT COUNT(*) FROM CharacterChats WHERE CharacterChats.character_id = CharacterCards.id) as chat_count
                FROM CharacterCards
                WHERE json_valid(tags)
                ORDER BY name
            """)

            results = cursor.fetchall()

            # Process the results to create rows for the CSV
            csv_rows = []
            for name, tags_json, chat_count in results:
                tags = json.loads(tags_json) if tags_json else []
                for tag in tags:
                    csv_rows.append([
                        tag,  # keyword
                        name,  # character name
                        chat_count  # number of chats
                    ])

            # Write to CSV
            writer = csv.writer(temp_file)
            writer.writerow(['Keyword', 'Character Name', 'Number of Chats'])
            writer.writerows(csv_rows)

        temp_file.close()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status_msg = f"Successfully exported {len(csv_rows)} character keyword entries to CSV."
        logging.info(status_msg)

        return status_msg, temp_file.name

    except Exception as e:
        error_msg = f"Error exporting keywords: {str(e)}"
        logging.error(error_msg)
        return error_msg, ""

#
# End of Character chat keyword functions
######################################################


#
# End of Character_Chat_DB.py
#######################################################################################################################

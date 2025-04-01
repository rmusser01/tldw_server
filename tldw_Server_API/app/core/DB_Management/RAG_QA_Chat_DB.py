# RAG_QA_Chat_DB.py
# Description: This file contains the database operations for the RAG QA Chat + Notes system.
#
# Imports
import configparser
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Tuple
#
# External Imports
# (No external imports)
#
# Local Imports
from tldw_Server_API.app.core.Utils.Utils import get_project_relative_path, get_project_root, logger, logging
#
########################################################################################################################
#
# Functions:

def get_rag_qa_db_path():
    config_path = os.path.join(get_project_root(), 'Config_Files', 'config.txt')
    config = configparser.ConfigParser()
    config.read(config_path)
    if config.has_section('Database') and config.has_option('Database', 'rag_qa_db_path'):
        rag_qa_db_path = config.get('Database', 'rag_qa_db_path')
        if not os.path.isabs(rag_qa_db_path):
            rag_qa_db_path = get_project_relative_path(rag_qa_db_path)
        return rag_qa_db_path
    else:
        raise ValueError("Database path not found in config file")

# Database schema
SCHEMA_SQL = '''
-- Table for storing chat messages
CREATE TABLE IF NOT EXISTS rag_qa_chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);

-- Table for storing conversation metadata
CREATE TABLE IF NOT EXISTS conversation_metadata (
    conversation_id TEXT PRIMARY KEY,
    created_at DATETIME NOT NULL,
    last_updated DATETIME NOT NULL,
    title TEXT NOT NULL,
    media_id INTEGER,
    parent_conversation_id TEXT REFERENCES conversation_metadata(conversation_id);
    forked_from_message_id INTEGER REFERENCES rag_qa_chats(id);
    rating INTEGER CHECK(rating BETWEEN 1 AND 3)
);

-- Table for storing keywords
CREATE TABLE IF NOT EXISTS rag_qa_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL UNIQUE
);

-- Table for linking keywords to conversations
CREATE TABLE IF NOT EXISTS rag_qa_conversation_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    keyword_id INTEGER NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversation_metadata(conversation_id),
    FOREIGN KEY (keyword_id) REFERENCES rag_qa_keywords(id)
);

-- Table for storing keyword collections
CREATE TABLE IF NOT EXISTS rag_qa_keyword_collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    parent_id INTEGER,
    FOREIGN KEY (parent_id) REFERENCES rag_qa_keyword_collections(id)
);

-- Table for linking keywords to collections
CREATE TABLE IF NOT EXISTS rag_qa_collection_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL,
    keyword_id INTEGER NOT NULL,
    FOREIGN KEY (collection_id) REFERENCES rag_qa_keyword_collections(id),
    FOREIGN KEY (keyword_id) REFERENCES rag_qa_keywords(id)
);

-- Table for storing notes
CREATE TABLE IF NOT EXISTS rag_qa_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversation_metadata(conversation_id)
);

-- Table for linking notes to keywords
CREATE TABLE IF NOT EXISTS rag_qa_note_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL,
    keyword_id INTEGER NOT NULL,
    FOREIGN KEY (note_id) REFERENCES rag_qa_notes(id),
    FOREIGN KEY (keyword_id) REFERENCES rag_qa_keywords(id)
);

-- Indexes for improved query performance
CREATE INDEX IF NOT EXISTS idx_rag_qa_chats_conversation_id ON rag_qa_chats(conversation_id);
CREATE INDEX IF NOT EXISTS idx_rag_qa_chats_timestamp ON rag_qa_chats(timestamp);
CREATE INDEX IF NOT EXISTS idx_rag_qa_keywords_keyword ON rag_qa_keywords(keyword);
CREATE INDEX IF NOT EXISTS idx_rag_qa_conversation_keywords_conversation_id ON rag_qa_conversation_keywords(conversation_id);
CREATE INDEX IF NOT EXISTS idx_rag_qa_conversation_keywords_keyword_id ON rag_qa_conversation_keywords(keyword_id);
CREATE INDEX IF NOT EXISTS idx_rag_qa_keyword_collections_parent_id ON rag_qa_keyword_collections(parent_id);
CREATE INDEX IF NOT EXISTS idx_rag_qa_collection_keywords_collection_id ON rag_qa_collection_keywords(collection_id);
CREATE INDEX IF NOT EXISTS idx_rag_qa_collection_keywords_keyword_id ON rag_qa_collection_keywords(keyword_id);

-- Full-text search virtual tables
CREATE VIRTUAL TABLE IF NOT EXISTS rag_qa_chats_fts USING fts5(
    content,
    content='rag_qa_chats',
    content_rowid='id'
);

-- FTS table for conversation metadata
CREATE VIRTUAL TABLE IF NOT EXISTS conversation_metadata_fts USING fts5(
    title,
    content='conversation_metadata',
    content_rowid='rowid'
);

-- FTS table for keywords
CREATE VIRTUAL TABLE IF NOT EXISTS rag_qa_keywords_fts USING fts5(
    keyword,
    content='rag_qa_keywords',
    content_rowid='id'
);

-- FTS table for keyword collections
CREATE VIRTUAL TABLE IF NOT EXISTS rag_qa_keyword_collections_fts USING fts5(
    name,
    content='rag_qa_keyword_collections',
    content_rowid='id'
);

-- FTS table for notes
CREATE VIRTUAL TABLE IF NOT EXISTS rag_qa_notes_fts USING fts5(
    title,
    content,
    content='rag_qa_notes',
    content_rowid='id'
);
-- FTS table for notes (modified to include both title and content)
CREATE VIRTUAL TABLE IF NOT EXISTS rag_qa_notes_fts USING fts5(
    title,
    content,
    content='rag_qa_notes',
    content_rowid='id'
);

-- Triggers for maintaining FTS indexes
-- Triggers for rag_qa_chats
CREATE TRIGGER IF NOT EXISTS rag_qa_chats_ai AFTER INSERT ON rag_qa_chats BEGIN
    INSERT INTO rag_qa_chats_fts(rowid, content) 
    VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS rag_qa_chats_au AFTER UPDATE ON rag_qa_chats BEGIN
    UPDATE rag_qa_chats_fts 
    SET content = new.content
    WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS rag_qa_chats_ad AFTER DELETE ON rag_qa_chats BEGIN
    DELETE FROM rag_qa_chats_fts WHERE rowid = old.id;
END;

-- Triggers for conversation_metadata
CREATE TRIGGER IF NOT EXISTS conversation_metadata_ai AFTER INSERT ON conversation_metadata BEGIN
    INSERT INTO conversation_metadata_fts(rowid, title) 
    VALUES (new.rowid, new.title);
END;

CREATE TRIGGER IF NOT EXISTS conversation_metadata_au AFTER UPDATE ON conversation_metadata BEGIN
    UPDATE conversation_metadata_fts 
    SET title = new.title
    WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS conversation_metadata_ad AFTER DELETE ON conversation_metadata BEGIN
    DELETE FROM conversation_metadata_fts WHERE rowid = old.rowid;
END;

-- Triggers for rag_qa_keywords
CREATE TRIGGER IF NOT EXISTS rag_qa_keywords_ai AFTER INSERT ON rag_qa_keywords BEGIN
    INSERT INTO rag_qa_keywords_fts(rowid, keyword) 
    VALUES (new.id, new.keyword);
END;

CREATE TRIGGER IF NOT EXISTS rag_qa_keywords_au AFTER UPDATE ON rag_qa_keywords BEGIN
    UPDATE rag_qa_keywords_fts 
    SET keyword = new.keyword
    WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS rag_qa_keywords_ad AFTER DELETE ON rag_qa_keywords BEGIN
    DELETE FROM rag_qa_keywords_fts WHERE rowid = old.id;
END;

-- Triggers for rag_qa_keyword_collections
CREATE TRIGGER IF NOT EXISTS rag_qa_keyword_collections_ai AFTER INSERT ON rag_qa_keyword_collections BEGIN
    INSERT INTO rag_qa_keyword_collections_fts(rowid, name) 
    VALUES (new.id, new.name);
END;

CREATE TRIGGER IF NOT EXISTS rag_qa_keyword_collections_au AFTER UPDATE ON rag_qa_keyword_collections BEGIN
    UPDATE rag_qa_keyword_collections_fts 
    SET name = new.name
    WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS rag_qa_keyword_collections_ad AFTER DELETE ON rag_qa_keyword_collections BEGIN
    DELETE FROM rag_qa_keyword_collections_fts WHERE rowid = old.id;
END;

-- Triggers for rag_qa_notes
CREATE TRIGGER IF NOT EXISTS rag_qa_notes_ai AFTER INSERT ON rag_qa_notes BEGIN
    INSERT INTO rag_qa_notes_fts(rowid, title, content) 
    VALUES (new.id, new.title, new.content);
END;

CREATE TRIGGER IF NOT EXISTS rag_qa_notes_au AFTER UPDATE ON rag_qa_notes BEGIN
    UPDATE rag_qa_notes_fts 
    SET title = new.title,
        content = new.content
    WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS rag_qa_notes_ad AFTER DELETE ON rag_qa_notes BEGIN
    DELETE FROM rag_qa_notes_fts WHERE rowid = old.id;
END;
'''

# Database connection management
@contextmanager
def get_db_connection():
    db_path = get_rag_qa_db_path()
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def transaction():
    with get_db_connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

def execute_query(query, params=None, conn=None):
    if conn:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.fetchall()
    else:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.fetchall()


def create_tables():
    """Create database tables and initialize FTS indexes."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Execute the SCHEMA_SQL to create tables and triggers
        cursor.executescript(SCHEMA_SQL)

        # Check and populate all FTS tables
        fts_tables = [
            ('rag_qa_notes_fts', 'rag_qa_notes', ['title', 'content']),
            ('rag_qa_chats_fts', 'rag_qa_chats', ['content']),
            ('conversation_metadata_fts', 'conversation_metadata', ['title']),
            ('rag_qa_keywords_fts', 'rag_qa_keywords', ['keyword']),
            ('rag_qa_keyword_collections_fts', 'rag_qa_keyword_collections', ['name'])
        ]

        for fts_table, source_table, columns in fts_tables:
            # Check if FTS table needs population
            cursor.execute(f"SELECT COUNT(*) FROM {fts_table}")
            fts_count = cursor.fetchone()[0]
            cursor.execute(f"SELECT COUNT(*) FROM {source_table}")
            source_count = cursor.fetchone()[0]

            if fts_count != source_count:
                # Repopulate FTS table
                logger.info(f"Repopulating {fts_table}")
                cursor.execute(f"DELETE FROM {fts_table}")
                columns_str = ', '.join(columns)
                source_columns = ', '.join([f"id" if source_table != 'conversation_metadata' else "rowid"] + columns)
                cursor.execute(f"""
                    INSERT INTO {fts_table}(rowid, {columns_str})
                    SELECT {source_columns} FROM {source_table}
                """)

        logger.info("All RAG QA Chat tables and triggers created successfully")


# Initialize the database
create_tables()

#
# End of Setup
############################################################


############################################################
#
# Keyword-related functions

# Input validation
def validate_keyword(keyword):
    if not isinstance(keyword, str):
        raise ValueError("Keyword must be a string")
    if not keyword.strip():
        raise ValueError("Keyword cannot be empty or just whitespace")
    if len(keyword) > 100:
        raise ValueError("Keyword is too long (max 100 characters)")
    if not re.match(r'^[a-zA-Z0-9\s\-_]+$', keyword):
        raise ValueError("Keyword contains invalid characters")
    return keyword.strip()


def validate_collection_name(name):
    if not isinstance(name, str):
        raise ValueError("Collection name must be a string")
    if not name.strip():
        raise ValueError("Collection name cannot be empty or just whitespace")
    if len(name) > 100:
        raise ValueError("Collection name is too long (max 100 characters)")
    if not re.match(r'^[a-zA-Z0-9\s\-_]+$', name):
        raise ValueError("Collection name contains invalid characters")
    return name.strip()


# Core functions
def add_keyword(keyword, conn=None):
    try:
        validated_keyword = validate_keyword(keyword)
        query = "INSERT OR IGNORE INTO rag_qa_keywords (keyword) VALUES (?)"
        execute_query(query, (validated_keyword,), conn)
        logger.info(f"Keyword '{validated_keyword}' added successfully")
    except ValueError as e:
        logger.error(f"Invalid keyword: {e}")
        raise
    except Exception as e:
        logger.error(f"Error adding keyword '{keyword}': {e}")
        raise


def create_keyword_collection(name, parent_id=None):
    try:
        validated_name = validate_collection_name(name)
        query = "INSERT INTO rag_qa_keyword_collections (name, parent_id) VALUES (?, ?)"
        execute_query(query, (validated_name, parent_id))
        logger.info(f"Keyword collection '{validated_name}' created successfully")
    except ValueError as e:
        logger.error(f"Invalid collection name: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating keyword collection '{name}': {e}")
        raise


def add_keyword_to_collection(collection_name, keyword):
    try:
        validated_collection_name = validate_collection_name(collection_name)
        validated_keyword = validate_keyword(keyword)

        with transaction() as conn:
            add_keyword(validated_keyword, conn)

            query = '''
            INSERT INTO rag_qa_collection_keywords (collection_id, keyword_id)
            SELECT c.id, k.id
            FROM rag_qa_keyword_collections c, rag_qa_keywords k
            WHERE c.name = ? AND k.keyword = ?
            '''
            execute_query(query, (validated_collection_name, validated_keyword), conn)

        logger.info(f"Keyword '{validated_keyword}' added to collection '{validated_collection_name}' successfully")
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise
    except Exception as e:
        logger.error(f"Error adding keyword '{keyword}' to collection '{collection_name}': {e}")
        raise


def add_keywords_to_conversation(conversation_id, keywords):
    if not isinstance(keywords, (list, tuple)):
        raise ValueError("Keywords must be a list or tuple")
    try:
        with transaction() as conn:
            for keyword in keywords:
                validated_keyword = validate_keyword(keyword)
                add_keyword(validated_keyword, conn)

                query = '''
                INSERT INTO rag_qa_conversation_keywords (conversation_id, keyword_id)
                SELECT ?, id FROM rag_qa_keywords WHERE keyword = ?
                '''
                execute_query(query, (conversation_id, validated_keyword), conn)

        logger.info(f"Keywords added to conversation '{conversation_id}' successfully")
    except ValueError as e:
        logger.error(f"Invalid keyword: {e}")
        raise
    except Exception as e:
        logger.error(f"Error adding keywords to conversation '{conversation_id}': {e}")
        raise


def view_rag_keywords():
    try:
        rag_db_path = get_rag_qa_db_path()
        with sqlite3.connect(rag_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT keyword FROM rag_qa_keywords ORDER BY keyword")
            keywords = cursor.fetchall()
            if keywords:
                keyword_list = [k[0] for k in keywords]
                return "### Current RAG QA Keywords:\n" + "\n".join(
                    [f"- {k}" for k in keyword_list])
            return "No keywords found."
    except Exception as e:
        return f"Error retrieving keywords: {str(e)}"


def get_keywords_for_conversation(conversation_id):
    try:
        query = '''
        SELECT k.keyword
        FROM rag_qa_keywords k
        JOIN rag_qa_conversation_keywords ck ON k.id = ck.keyword_id
        WHERE ck.conversation_id = ?
        '''
        result = execute_query(query, (conversation_id,))
        keywords = [row[0] for row in result]
        logger.info(f"Retrieved {len(keywords)} keywords for conversation '{conversation_id}'")
        return keywords
    except Exception as e:
        logger.error(f"Error getting keywords for conversation '{conversation_id}': {e}")
        raise


def get_keywords_for_collection(collection_name):
    try:
        query = '''
        SELECT k.keyword
        FROM rag_qa_keywords k
        JOIN rag_qa_collection_keywords ck ON k.id = ck.keyword_id
        JOIN rag_qa_keyword_collections c ON ck.collection_id = c.id
        WHERE c.name = ?
        '''
        result = execute_query(query, (collection_name,))
        keywords = [row[0] for row in result]
        logger.info(f"Retrieved {len(keywords)} keywords for collection '{collection_name}'")
        return keywords
    except Exception as e:
        logger.error(f"Error getting keywords for collection '{collection_name}': {e}")
        raise


def delete_rag_keyword(keyword: str) -> str:
    """
    Delete a keyword from the RAG QA database and all its associations.

    Args:
        keyword (str): The keyword to delete

    Returns:
        str: Success/failure message
    """
    try:
        # Validate the keyword
        validated_keyword = validate_keyword(keyword)

        with transaction() as conn:
            # First, get the keyword ID
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM rag_qa_keywords WHERE keyword = ?", (validated_keyword,))
            result = cursor.fetchone()

            if not result:
                return f"Keyword '{validated_keyword}' not found."

            keyword_id = result[0]

            # Delete from all associated tables
            cursor.execute("DELETE FROM rag_qa_conversation_keywords WHERE keyword_id = ?", (keyword_id,))
            cursor.execute("DELETE FROM rag_qa_collection_keywords WHERE keyword_id = ?", (keyword_id,))
            cursor.execute("DELETE FROM rag_qa_note_keywords WHERE keyword_id = ?", (keyword_id,))

            # Finally, delete the keyword itself
            cursor.execute("DELETE FROM rag_qa_keywords WHERE id = ?", (keyword_id,))

            logger.info(f"Keyword '{validated_keyword}' deleted successfully")
            return f"Successfully deleted keyword '{validated_keyword}' and all its associations."

    except ValueError as e:
        error_msg = f"Invalid keyword: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error deleting keyword: {str(e)}"
        logger.error(error_msg)
        return error_msg


def export_rag_keywords_to_csv() -> Tuple[str, str]:
    """
    Export all RAG QA keywords to a CSV file.

    Returns:
        Tuple[str, str]: (status_message, file_path)
    """
    import csv
    from tempfile import NamedTemporaryFile
    from datetime import datetime

    try:
        # Create a temporary CSV file
        temp_file = NamedTemporaryFile(mode='w+', delete=False, suffix='.csv', newline='')

        with transaction() as conn:
            cursor = conn.cursor()

            # Get all keywords and their associations
            query = """
            SELECT 
                k.keyword,
                GROUP_CONCAT(DISTINCT c.name) as collections,
                COUNT(DISTINCT ck.conversation_id) as num_conversations,
                COUNT(DISTINCT nk.note_id) as num_notes
            FROM rag_qa_keywords k
            LEFT JOIN rag_qa_collection_keywords col_k ON k.id = col_k.keyword_id
            LEFT JOIN rag_qa_keyword_collections c ON col_k.collection_id = c.id
            LEFT JOIN rag_qa_conversation_keywords ck ON k.id = ck.keyword_id
            LEFT JOIN rag_qa_note_keywords nk ON k.id = nk.keyword_id
            GROUP BY k.id, k.keyword
            ORDER BY k.keyword
            """

            cursor.execute(query)
            results = cursor.fetchall()

            # Write to CSV
            writer = csv.writer(temp_file)
            writer.writerow(['Keyword', 'Collections', 'Number of Conversations', 'Number of Notes'])

            for row in results:
                writer.writerow([
                    row[0],  # keyword
                    row[1] if row[1] else '',  # collections (may be None)
                    row[2],  # num_conversations
                    row[3]  # num_notes
                ])

        temp_file.close()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status_msg = f"Successfully exported {len(results)} keywords to CSV."
        logger.info(status_msg)

        return status_msg, temp_file.name

    except Exception as e:
        error_msg = f"Error exporting keywords: {str(e)}"
        logger.error(error_msg)
        return error_msg, ""


#
# End of Keyword-related functions
###################################################


###################################################
#
# Notes and chat-related functions

def save_notes(conversation_id, title, content):
    """Save notes to the database."""
    try:
        query = "INSERT INTO rag_qa_notes (conversation_id, title, content, timestamp) VALUES (?, ?, ?, ?)"
        timestamp = datetime.now().isoformat()
        with transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (conversation_id, title, content, timestamp))
            note_id = cursor.lastrowid
        logger.info(f"Notes saved for conversation '{conversation_id}', note ID '{note_id}'")
        return note_id
    except Exception as e:
        logger.error(f"Error saving notes for conversation '{conversation_id}': {e}")
        raise


def update_note(note_id, title, content):
    try:
        query = "UPDATE rag_qa_notes SET title = ?, content = ?, timestamp = ? WHERE id = ?"
        timestamp = datetime.now().isoformat()
        execute_query(query, (title, content, timestamp, note_id))
        logger.info(f"Note ID '{note_id}' updated successfully")
    except Exception as e:
        logger.error(f"Error updating note ID '{note_id}': {e}")
        raise


def search_notes_titles(search_term: str, page: int = 1, results_per_page: int = 20, connection=None) -> Tuple[
    List[Tuple], int, int]:
    """
    Search note titles using full-text search. Returns all notes if search_term is empty.

    Args:
        search_term (str): The search term for note titles. If empty, returns all notes.
        page (int, optional): Page number for pagination. Defaults to 1.
        results_per_page (int, optional): Number of results per page. Defaults to 20.
        connection (sqlite3.Connection, optional): Database connection. Uses new connection if not provided.

    Returns:
        Tuple[List[Tuple], int, int]: Tuple containing:
            - List of tuples: (note_id, title, content, timestamp, conversation_id)
            - Total number of pages
            - Total count of matching records

    Raises:
        ValueError: If page number is less than 1
        sqlite3.Error: If there's a database error
    """
    if page < 1:
        raise ValueError("Page number must be 1 or greater.")

    offset = (page - 1) * results_per_page

    def execute_search(conn):
        cursor = conn.cursor()

        # Debug: Show table contents
        cursor.execute("SELECT title FROM rag_qa_notes")
        main_titles = cursor.fetchall()
        logger.debug(f"Main table titles: {main_titles}")

        cursor.execute("SELECT title FROM rag_qa_notes_fts")
        fts_titles = cursor.fetchall()
        logger.debug(f"FTS table titles: {fts_titles}")

        if not search_term.strip():
            # Query for all notes
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM rag_qa_notes
                """
            )
            total_count = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT id, title, content, timestamp, conversation_id
                FROM rag_qa_notes
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (results_per_page, offset)
            )
            results = cursor.fetchall()
        else:
            # Search query
            search_term_clean = search_term.strip().lower()

            # Test direct FTS search
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM rag_qa_notes n
                JOIN rag_qa_notes_fts fts ON n.id = fts.rowid
                WHERE fts.title MATCH ?
                """,
                (search_term_clean,)
            )
            total_count = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT 
                    n.id,
                    n.title,
                    n.content,
                    n.timestamp,
                    n.conversation_id
                FROM rag_qa_notes n
                JOIN rag_qa_notes_fts fts ON n.id = fts.rowid
                WHERE fts.title MATCH ?
                ORDER BY rank
                LIMIT ? OFFSET ?
                """,
                (search_term_clean, results_per_page, offset)
            )
            results = cursor.fetchall()

            logger.debug(f"Search term: {search_term_clean}")
            logger.debug(f"Results: {results}")

        total_pages = max(1, (total_count + results_per_page - 1) // results_per_page)
        logger.info(f"Found {total_count} matching notes for search term '{search_term}'")

        return results, total_pages, total_count

    try:
        if connection:
            return execute_search(connection)
        else:
            with get_db_connection() as conn:
                return execute_search(conn)

    except sqlite3.Error as e:
        logger.error(f"Database error in search_notes_titles: {str(e)}")
        logger.error(f"Search term: {search_term}")
        raise sqlite3.Error(f"Error searching notes: {str(e)}")



def get_notes(conversation_id):
    """Retrieve notes for a given conversation."""
    try:
        query = "SELECT content FROM rag_qa_notes WHERE conversation_id = ?"
        result = execute_query(query, (conversation_id,))
        notes = [row[0] for row in result]
        logger.info(f"Retrieved {len(notes)} notes for conversation '{conversation_id}'")
        return notes
    except Exception as e:
        logger.error(f"Error getting notes for conversation '{conversation_id}': {e}")
        raise


def get_note_by_id(note_id):
    try:
        query = "SELECT id, title, content FROM rag_qa_notes WHERE id = ?"
        result = execute_query(query, (note_id,))
        return result
    except Exception as e:
        logger.error(f"Error getting note by ID '{note_id}': {e}")
        raise


def get_notes_by_keywords(keywords, page=1, page_size=20):
    try:
        # Handle empty or invalid keywords
        if not keywords or not isinstance(keywords, (list, tuple)) or len(keywords) == 0:
            return [], 0, 0

        # Convert all keywords to strings and strip them
        clean_keywords = [str(k).strip() for k in keywords if k is not None and str(k).strip()]

        # If no valid keywords after cleaning, return empty result
        if not clean_keywords:
            return [], 0, 0

        placeholders = ','.join(['?'] * len(clean_keywords))
        query = f'''
        SELECT n.id, n.title, n.content, n.timestamp
        FROM rag_qa_notes n
        JOIN rag_qa_note_keywords nk ON n.id = nk.note_id
        JOIN rag_qa_keywords k ON nk.keyword_id = k.id
        WHERE k.keyword IN ({placeholders})
        ORDER BY n.timestamp DESC
        '''
        results, total_pages, total_count = get_paginated_results(query, tuple(clean_keywords), page, page_size)
        logger.info(f"Retrieved {len(results)} notes matching keywords: {', '.join(clean_keywords)} (page {page} of {total_pages})")
        notes = [(row[0], row[1], row[2], row[3]) for row in results]
        return notes, total_pages, total_count
    except Exception as e:
        logger.error(f"Error getting notes by keywords: {e}")
        raise


def get_notes_by_keyword_collection(collection_name, page=1, page_size=20):
    try:
        query = '''
        SELECT n.id, n.title, n.content, n.timestamp
        FROM rag_qa_notes n
        JOIN rag_qa_note_keywords nk ON n.id = nk.note_id
        JOIN rag_qa_keywords k ON nk.keyword_id = k.id
        JOIN rag_qa_collection_keywords ck ON k.id = ck.keyword_id
        JOIN rag_qa_keyword_collections c ON ck.collection_id = c.id
        WHERE c.name = ?
        ORDER BY n.timestamp DESC
        '''
        results, total_pages, total_count = get_paginated_results(query, (collection_name,), page, page_size)
        logger.info(f"Retrieved {len(results)} notes for collection '{collection_name}' (page {page} of {total_pages})")
        notes = [(row[0], row[1], row[2], row[3]) for row in results]
        return notes, total_pages, total_count
    except Exception as e:
        logger.error(f"Error getting notes by keyword collection '{collection_name}': {e}")
        raise

def clear_notes(conversation_id):
    """Clear all notes for a given conversation."""
    try:
        query = "DELETE FROM rag_qa_notes WHERE conversation_id = ?"
        execute_query(query, (conversation_id,))
        logger.info(f"Cleared notes for conversation '{conversation_id}'")
    except Exception as e:
        logger.error(f"Error clearing notes for conversation '{conversation_id}': {e}")
        raise

def add_keywords_to_note(note_id, keywords):
    """Associate keywords with a note."""
    try:
        with transaction() as conn:
            for keyword in keywords:
                validated_keyword = validate_keyword(keyword)
                add_keyword(validated_keyword, conn)

                # Retrieve the keyword ID
                query = "SELECT id FROM rag_qa_keywords WHERE keyword = ?"
                result = execute_query(query, (validated_keyword,), conn)
                if result:
                    keyword_id = result[0][0]
                else:
                    raise Exception(f"Keyword '{validated_keyword}' not found after insertion")

                # Link the note and keyword
                query = "INSERT INTO rag_qa_note_keywords (note_id, keyword_id) VALUES (?, ?)"
                execute_query(query, (note_id, keyword_id), conn)

        logger.info(f"Keywords added to note ID '{note_id}' successfully")
    except Exception as e:
        logger.error(f"Error adding keywords to note ID '{note_id}': {e}")
        raise

def get_keywords_for_note(note_id):
    """Retrieve keywords associated with a given note."""
    try:
        query = '''
        SELECT k.keyword
        FROM rag_qa_keywords k
        JOIN rag_qa_note_keywords nk ON k.id = nk.keyword_id
        WHERE nk.note_id = ?
        '''
        result = execute_query(query, (note_id,))
        keywords = [row[0] for row in result]
        logger.info(f"Retrieved {len(keywords)} keywords for note ID '{note_id}'")
        return keywords
    except Exception as e:
        logger.error(f"Error getting keywords for note ID '{note_id}': {e}")
        raise

def clear_keywords_from_note(note_id):
    """Clear all keywords from a given note."""
    try:
        query = "DELETE FROM rag_qa_note_keywords WHERE note_id = ?"
        execute_query(query, (note_id,))
        logger.info(f"Cleared keywords for note ID '{note_id}'")
    except Exception as e:
        logger.error(f"Error clearing keywords for note ID '{note_id}': {e}")
        raise

def delete_note_by_id(note_id, conn=None):
    """Delete a note and its associated keywords."""
    try:
        # Delete note keywords
        execute_query("DELETE FROM rag_qa_note_keywords WHERE note_id = ?", (note_id,), conn)
        # Delete the note
        execute_query("DELETE FROM rag_qa_notes WHERE id = ?", (note_id,), conn)
        logging.info(f"Note ID '{note_id}' deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting note ID '{note_id}': {e}")
        raise

def delete_note(note_id):
    """Delete a note by ID."""
    try:
        with transaction() as conn:
            delete_note_by_id(note_id, conn)
    except Exception as e:
        logger.error(f"Error deleting note ID '{note_id}': {e}")
        raise

#
# End of Notes related functions
###################################################


###################################################
#
# Chat-related functions

def save_message(conversation_id, role, content, timestamp=None):
    try:
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        query = "INSERT INTO rag_qa_chats (conversation_id, timestamp, role, content) VALUES (?, ?, ?, ?)"
        execute_query(query, (conversation_id, timestamp, role, content))

        # Update last_updated in conversation_metadata
        update_query = "UPDATE conversation_metadata SET last_updated = ? WHERE conversation_id = ?"
        execute_query(update_query, (timestamp, conversation_id))

        logger.info(f"Message saved for conversation '{conversation_id}'")
    except Exception as e:
        logger.error(f"Error saving message for conversation '{conversation_id}': {e}")
        raise


def start_new_conversation(title="Untitled Conversation", media_id=None):
    try:
        conversation_id = str(uuid.uuid4())
        query = """
            INSERT INTO conversation_metadata (
                conversation_id, created_at, last_updated, title, media_id, rating
            ) VALUES (?, ?, ?, ?, ?, ?)
        """
        now = datetime.now().isoformat()
        # Set initial rating to NULL
        execute_query(query, (conversation_id, now, now, title, media_id, None))
        logger.info(f"New conversation '{conversation_id}' started with title '{title}' and media_id '{media_id}'")
        return conversation_id
    except Exception as e:
        logger.error(f"Error starting new conversation: {e}")
        raise


def get_all_conversations(page=1, page_size=20):
    try:
        query = """
            SELECT conversation_id, title, media_id, rating
            FROM conversation_metadata
            ORDER BY last_updated DESC
            LIMIT ? OFFSET ?
        """

        count_query = "SELECT COUNT(*) FROM conversation_metadata"
        db_path = get_rag_qa_db_path()
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Get total count
            cursor.execute(count_query)
            total_count = cursor.fetchone()[0]
            total_pages = (total_count + page_size - 1) // page_size

            # Get page of results
            offset = (page - 1) * page_size
            cursor.execute(query, (page_size, offset))
            results = cursor.fetchall()

            conversations = [{
                'conversation_id': row[0],
                'title': row[1],
                'media_id': row[2],
                'rating': row[3]  # Include rating
            } for row in results]
            return conversations, total_pages, total_count
    except Exception as e:
        logging.error(f"Error getting conversations: {e}")
        raise


def get_all_notes(page=1, page_size=20):
    try:
        query = """
        SELECT n.id, n.conversation_id, n.title, n.content, n.timestamp,
               cm.title as conversation_title, cm.media_id
        FROM rag_qa_notes n
        LEFT JOIN conversation_metadata cm ON n.conversation_id = cm.conversation_id
        ORDER BY n.timestamp DESC
        LIMIT ? OFFSET ?
        """

        count_query = "SELECT COUNT(*) FROM rag_qa_notes"
        db_path = get_rag_qa_db_path()
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Get total count
            cursor.execute(count_query)
            total_count = cursor.fetchone()[0]
            total_pages = (total_count + page_size - 1) // page_size

            # Get page of results
            offset = (page - 1) * page_size
            cursor.execute(query, (page_size, offset))
            results = cursor.fetchall()

            notes = [{
                'id': row[0],
                'conversation_id': row[1],
                'title': row[2],
                'content': row[3],
                'timestamp': row[4],
                'conversation_title': row[5],
                'media_id': row[6]
            } for row in results]

        return notes, total_pages, total_count
    except Exception as e:
        logging.error(f"Error getting notes: {e}")
        raise


# Pagination helper function
def get_paginated_results(query, params=None, page=1, page_size=20):
    try:
        offset = (page - 1) * page_size
        paginated_query = f"{query} LIMIT ? OFFSET ?"
        if params:
            paginated_params = params + (page_size, offset)
        else:
            paginated_params = (page_size, offset)

        result = execute_query(paginated_query, paginated_params)

        count_query = f"SELECT COUNT(*) FROM ({query}) AS total"
        count_params = params if params else ()

        total_count = execute_query(count_query, count_params)[0][0]

        total_pages = (total_count + page_size - 1) // page_size

        logger.info(f"Retrieved page {page} of {total_pages} (total items: {total_count})")
        return result, total_pages, total_count
    except Exception as e:
        logger.error(f"Error retrieving paginated results: {e}")
        raise


def get_all_collections(page=1, page_size=20):
    try:
        query = "SELECT name FROM rag_qa_keyword_collections"
        results, total_pages, total_count = get_paginated_results(query, page=page, page_size=page_size)
        collections = [row[0] for row in results]
        logger.info(f"Retrieved {len(collections)} keyword collections (page {page} of {total_pages})")
        return collections, total_pages, total_count
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        raise


def search_conversations_by_keywords(keywords=None, title_query=None, content_query=None, page=1, page_size=20):
    try:
        # Base query starts with conversation metadata
        query = """
        SELECT DISTINCT cm.conversation_id, cm.title, cm.last_updated
        FROM conversation_metadata cm
        WHERE 1=1
        """
        params = []

        # Add content search if provided
        if content_query and isinstance(content_query, str) and content_query.strip():
            query += """
            AND EXISTS (
                SELECT 1 FROM rag_qa_chats_fts
                WHERE rag_qa_chats_fts.content MATCH ?
                AND rag_qa_chats_fts.rowid IN (
                    SELECT id FROM rag_qa_chats 
                    WHERE conversation_id = cm.conversation_id
                )
            )
            """
            params.append(content_query.strip())

        # Add title search if provided
        if title_query and isinstance(title_query, str) and title_query.strip():
            query += """
            AND EXISTS (
                SELECT 1 FROM conversation_metadata_fts
                WHERE conversation_metadata_fts.title MATCH ?
                AND conversation_metadata_fts.rowid = cm.rowid
            )
            """
            params.append(title_query.strip())

        # Add keyword search if provided
        if keywords and isinstance(keywords, (list, tuple)) and len(keywords) > 0:
            # Convert all keywords to strings and strip them
            clean_keywords = [str(k).strip() for k in keywords if k is not None and str(k).strip()]
            if clean_keywords:  # Only add to query if we have valid keywords
                placeholders = ','.join(['?' for _ in clean_keywords])
                query += f"""
                AND EXISTS (
                    SELECT 1 FROM rag_qa_conversation_keywords ck
                    JOIN rag_qa_keywords k ON ck.keyword_id = k.id
                    WHERE ck.conversation_id = cm.conversation_id
                    AND k.keyword IN ({placeholders})
                )
                """
                params.extend(clean_keywords)

        # Add ordering
        query += " ORDER BY cm.last_updated DESC"

        results, total_pages, total_count = get_paginated_results(query, tuple(params), page, page_size)

        conversations = [
            {
                'conversation_id': row[0],
                'title': row[1],
                'last_updated': row[2]
            }
            for row in results
        ]

        return conversations, total_pages, total_count

    except Exception as e:
        logger.error(f"Error searching conversations: {e}")
        raise


def load_chat_history(conversation_id, page=1, page_size=50):
    try:
        query = "SELECT role, content FROM rag_qa_chats WHERE conversation_id = ? ORDER BY timestamp"
        results, total_pages, total_count = get_paginated_results(query, (conversation_id,), page, page_size)
        logger.info(
            f"Loaded {len(results)} messages for conversation '{conversation_id}' (page {page} of {total_pages})")
        return results, total_pages, total_count
    except Exception as e:
        logger.error(f"Error loading chat history for conversation '{conversation_id}': {e}")
        raise


def update_conversation_title(conversation_id, new_title):
    """Update the title of a conversation."""
    try:
        query = "UPDATE conversation_metadata SET title = ? WHERE conversation_id = ?"
        execute_query(query, (new_title, conversation_id))
        logger.info(f"Conversation '{conversation_id}' title updated to '{new_title}'")
    except Exception as e:
        logger.error(f"Error updating conversation title: {e}")
        raise


def delete_messages_in_conversation(conversation_id):
    """Helper function to delete all messages in a conversation."""
    try:
        execute_query("DELETE FROM rag_qa_chats WHERE conversation_id = ?", (conversation_id,))
        logging.info(f"Messages in conversation '{conversation_id}' deleted successfully.")
    except Exception as e:
        logging.error(f"Error deleting messages in conversation '{conversation_id}': {e}")
        raise


def get_conversation_title(conversation_id):
    """Helper function to get the conversation title."""
    query = "SELECT title FROM conversation_metadata WHERE conversation_id = ?"
    result = execute_query(query, (conversation_id,))
    if result:
        return result[0][0]
    else:
        return "Untitled Conversation"


def get_conversation_text(conversation_id):
    try:
        query = """
        SELECT role, content
        FROM rag_qa_chats
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
        """

        messages = []
        # Use the connection as a context manager
        db_path = get_rag_qa_db_path()
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (conversation_id,))
            messages = cursor.fetchall()

        return "\n\n".join([f"{msg[0]}: {msg[1]}" for msg in messages])
    except Exception as e:
        logger.error(f"Error getting conversation text: {e}")
        raise


def get_conversation_details(conversation_id):
    query = "SELECT title, media_id, rating FROM conversation_metadata WHERE conversation_id = ?"
    result = execute_query(query, (conversation_id,))
    if result:
        return {'title': result[0][0], 'media_id': result[0][1], 'rating': result[0][2]}
    else:
        return {'title': "Untitled Conversation", 'media_id': None, 'rating': None}


def delete_conversation(conversation_id):
    """Delete a conversation and its associated messages and notes."""
    try:
        with transaction() as conn:
            # Delete messages
            execute_query("DELETE FROM rag_qa_chats WHERE conversation_id = ?", (conversation_id,), conn)
            # Delete conversation metadata
            execute_query("DELETE FROM conversation_metadata WHERE conversation_id = ?", (conversation_id,), conn)
            # Delete conversation keywords
            execute_query("DELETE FROM rag_qa_conversation_keywords WHERE conversation_id = ?", (conversation_id,), conn)
            # Delete notes associated with the conversation
            note_ids = execute_query("SELECT id FROM rag_qa_notes WHERE conversation_id = ?", (conversation_id,), conn)
            for (note_id,) in note_ids:
                delete_note_by_id(note_id, conn)
            logging.info(f"Conversation '{conversation_id}' deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting conversation '{conversation_id}': {e}")
        raise

def set_conversation_rating(conversation_id, rating):
    """Set the rating for a conversation."""
    # Validate rating
    if rating not in [1, 2, 3]:
        raise ValueError('Rating must be an integer between 1 and 3.')
    try:
        query = "UPDATE conversation_metadata SET rating = ? WHERE conversation_id = ?"
        execute_query(query, (rating, conversation_id))
        logger.info(f"Rating for conversation '{conversation_id}' set to {rating}")
    except Exception as e:
        logger.error(f"Error setting rating for conversation '{conversation_id}': {e}")
        raise

def get_conversation_rating(conversation_id):
    """Get the rating of a conversation."""
    try:
        query = "SELECT rating FROM conversation_metadata WHERE conversation_id = ?"
        result = execute_query(query, (conversation_id,))
        if result:
            rating = result[0][0]
            logger.info(f"Rating for conversation '{conversation_id}' is {rating}")
            return rating
        else:
            logger.warning(f"Conversation '{conversation_id}' not found.")
            return None
    except Exception as e:
        logger.error(f"Error getting rating for conversation '{conversation_id}': {e}")
        raise


def get_conversation_name(conversation_id: str) -> str:
    """
    Retrieves the title/name of a conversation from the conversation_metadata table.

    Args:
        conversation_id (str): The unique identifier of the conversation

    Returns:
        str: The title of the conversation if found, "Untitled Conversation" if not found

    Raises:
        sqlite3.Error: If there's a database error
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT title FROM conversation_metadata WHERE conversation_id = ?",
                (conversation_id,)
            )
            result = cursor.fetchone()

            if result:
                return result[0]
            else:
                logging.warning(f"No conversation found with ID: {conversation_id}")
                return "Untitled Conversation"

    except sqlite3.Error as e:
        logging.error(f"Database error retrieving conversation name for ID {conversation_id}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error retrieving conversation name for ID {conversation_id}: {e}")
        raise


def search_rag_chat(query: str, fts_top_k: int = 10, relevant_media_ids: List[str] = None) -> List[Dict[str, Any]]:
    """
    Perform a full-text search on the RAG Chat database.

    Args:
        query: Search query string.
        fts_top_k: Maximum number of results to return.
        relevant_media_ids: Optional list of media IDs to filter results.

    Returns:
        List of search results with content and metadata.
    """
    if not query.strip():
        return []

    try:
        db_path = get_rag_qa_db_path()
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Perform the full-text search using the FTS virtual table
            cursor.execute("""
                SELECT rag_qa_chats.id, rag_qa_chats.conversation_id, rag_qa_chats.role, rag_qa_chats.content
                FROM rag_qa_chats_fts
                JOIN rag_qa_chats ON rag_qa_chats_fts.rowid = rag_qa_chats.id
                WHERE rag_qa_chats_fts MATCH ?
                LIMIT ?
            """, (query, fts_top_k))

            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]

            # Filter by relevant_media_ids if provided
            if relevant_media_ids is not None:
                results = [
                    r for r in results
                    if get_conversation_details(r['conversation_id']).get('media_id') in relevant_media_ids
                ]

            # Format results
            formatted_results = [
                {
                    "content": r['content'],
                    "metadata": {
                        "conversation_id": r['conversation_id'],
                        "role": r['role'],
                        "media_id": get_conversation_details(r['conversation_id']).get('media_id')
                    }
                }
                for r in results
            ]
            return formatted_results

    except Exception as e:
        logging.error(f"Error in search_rag_chat: {e}")
        return []


def search_rag_notes(query: str, fts_top_k: int = 10, relevant_media_ids: List[str] = None) -> List[Dict[str, Any]]:
    """
    Perform a full-text search on the RAG Notes database.

    Args:
        query: Search query string.
        fts_top_k: Maximum number of results to return.
        relevant_media_ids: Optional list of media IDs to filter results.

    Returns:
        List of search results with content and metadata.
    """
    if not query.strip():
        return []

    try:
        db_path = get_rag_qa_db_path()
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Perform the full-text search using the FTS virtual table
            cursor.execute("""
                SELECT rag_qa_notes.id, rag_qa_notes.title, rag_qa_notes.content, rag_qa_notes.conversation_id
                FROM rag_qa_notes_fts
                JOIN rag_qa_notes ON rag_qa_notes_fts.rowid = rag_qa_notes.id
                WHERE rag_qa_notes_fts MATCH ?
                LIMIT ?
            """, (query, fts_top_k))

            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]

            # Filter by relevant_media_ids if provided
            if relevant_media_ids is not None:
                results = [
                    r for r in results
                    if get_conversation_details(r['conversation_id']).get('media_id') in relevant_media_ids
                ]

            # Format results
            formatted_results = [
                {
                    "content": r['content'],
                    "metadata": {
                        "note_id": r['id'],
                        "title": r['title'],
                        "conversation_id": r['conversation_id'],
                        "media_id": get_conversation_details(r['conversation_id']).get('media_id')
                    }
                }
                for r in results
            ]
            return formatted_results

    except Exception as e:
        logging.error(f"Error in search_rag_notes: {e}")
        return []

#
# End of Chat-related functions
###################################################


###################################################
#
# Import functions


#
# End of Import functions
###################################################


###################################################
#
# Functions to export DB data

def fetch_all_conversations():
    try:
        # Fetch all conversation IDs and titles
        query = "SELECT conversation_id, title FROM conversation_metadata ORDER BY last_updated DESC"
        results = execute_query(query)
        conversations = []
        for row in results:
            conversation_id, title = row
            # Fetch all messages for this conversation
            messages = load_all_chat_history(conversation_id)
            conversations.append((conversation_id, title, messages))
        logger.info(f"Fetched all conversations: {len(conversations)} found.")
        return conversations
    except Exception as e:
        logger.error(f"Error fetching all conversations: {e}")
        raise

def load_all_chat_history(conversation_id):
    try:
        query = "SELECT role, content FROM rag_qa_chats WHERE conversation_id = ? ORDER BY timestamp"
        results = execute_query(query, (conversation_id,))
        messages = [(row[0], row[1]) for row in results]
        return messages
    except Exception as e:
        logger.error(f"Error loading chat history for conversation '{conversation_id}': {e}")
        raise

def fetch_all_notes():
    try:
        query = "SELECT id, title, content FROM rag_qa_notes ORDER BY timestamp DESC"
        results = execute_query(query)
        notes = [(row[0], row[1], row[2]) for row in results]
        logger.info(f"Fetched all notes: {len(notes)} found.")
        return notes
    except Exception as e:
        logger.error(f"Error fetching all notes: {e}")
        raise

def fetch_conversations_by_ids(conversation_ids):
    try:
        if not conversation_ids:
            return []
        placeholders = ','.join(['?'] * len(conversation_ids))
        query = f"SELECT conversation_id, title FROM conversation_metadata WHERE conversation_id IN ({placeholders})"
        results = execute_query(query, conversation_ids)
        conversations = []
        for row in results:
            conversation_id, title = row
            # Fetch all messages for this conversation
            messages = load_all_chat_history(conversation_id)
            conversations.append((conversation_id, title, messages))
        logger.info(f"Fetched {len(conversations)} conversations by IDs.")
        return conversations
    except Exception as e:
        logger.error(f"Error fetching conversations by IDs: {e}")
        raise

def fetch_notes_by_ids(note_ids):
    try:
        if not note_ids:
            return []
        placeholders = ','.join(['?'] * len(note_ids))
        query = f"SELECT id, title, content FROM rag_qa_notes WHERE id IN ({placeholders})"
        results = execute_query(query, note_ids)
        notes = [(row[0], row[1], row[2]) for row in results]
        logger.info(f"Fetched {len(notes)} notes by IDs.")
        return notes
    except Exception as e:
        logger.error(f"Error fetching notes by IDs: {e}")
        raise

#
# End of Export functions
###################################################

#
# End of RAG_QA_Chat_DB.py
####################################################################################################

# tldw_Server_API/tests/conftest.py
#
import pytest
import sqlite3
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import Database
from tldw_Server_API.app.core.Utils.Utils import logging  # Ensure logging is available


@pytest.fixture(scope="session")  # The factory itself can be session-scoped for efficiency
def memory_db_factory():
    """
    Pytest factory fixture that provides a function to create
    new, isolated in-memory Database instances.
    """
    created_db_instances = []  # To manage cleanup if necessary

    def _create_db(client_id_prefix: str = "mem_db_client"):
        """
        Creates and returns a new Database instance connected to ':memory:'.
        Ensures necessary schema, including FTS tables, is created.
        """
        # Generate a unique client_id for each DB instance to prevent potential collisions
        # if client_id is used for any singleton patterns or global state (though unlikely for :memory: DBs).
        unique_client_id = f"{client_id_prefix}_{Database._generate_uuid()}"
        db_instance = Database(db_path=":memory:", client_id=unique_client_id)

        # Ensure schema, including FTS tables, is created.
        # This mirrors the logic in TestDocumentVersioningV2.db_instance.
        # Ideally, the Database class would handle its full schema creation on init.
        try:
            with db_instance.get_connection() as conn:  # Use context manager for connection
                cursor = conn.cursor()
                try:
                    # Check if FTS tables exist
                    cursor.execute("SELECT 1 FROM media_fts LIMIT 1")
                except sqlite3.OperationalError as e:
                    if "no such table" in str(e).lower():  # Make check case-insensitive
                        logging.warning(
                            f"FTS tables not found for {db_instance.db_path_str} (client: {unique_client_id}), attempting to create.")
                        if hasattr(Database, '_FTS_TABLES_SQL') and Database._FTS_TABLES_SQL:
                            # Execute script to create tables
                            conn.executescript(Database._FTS_TABLES_SQL)
                            # No explicit commit needed for executescript with DDL in sqlite3 usually,
                            # but ensure your Database.execute_query or connection handling does it if required.
                            # If db.execute_query(Database._FTS_TABLES_SQL, commit=True) was used,
                            # and execute_query handles commits, that's fine.
                            # Here, using conn.executescript directly.
                            logging.info(
                                f"FTS tables created for {db_instance.db_path_str} (client: {unique_client_id}).")
                        else:
                            logging.error(
                                "Database._FTS_TABLES_SQL is not defined or is empty. Cannot create FTS tables.")
                            # Consider failing the test if FTS is critical
                            # pytest.fail("Cannot create FTS tables: _FTS_TABLES_SQL is missing or empty.")
                    else:
                        # Re-raise other sqlite3.OperationalError exceptions
                        raise
        except Exception as schema_exc:
            logging.error(
                f"Failed to ensure schema (including FTS tables) for {db_instance.db_path_str} (client: {unique_client_id}): {schema_exc}",
                exc_info=True)
            # Depending on how critical FTS tables are, you might want to fail tests here.
            # pytest.fail(f"Schema creation failed for in-memory DB: {schema_exc}")

        created_db_instances.append(db_instance)
        return db_instance

    yield _create_db  # The fixture yields the creator function

    # Optional: Cleanup for DBs created by this factory,
    # though :memory: databases are destroyed when their last connection is closed.
    # The fixture using this factory (db_instance) should manage its own DB closure.
    for db in created_db_instances:
        if hasattr(db, 'close_all_connections'):
            db.close_all_connections()
            logging.debug(f"Closed all connections for DB: {db.client_id}")
        elif hasattr(db, 'close_connection'):
            db.close_connection()
            logging.debug(f"Closed connection for DB: {db.client_id}")
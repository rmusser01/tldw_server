import hashlib
import logging
import sqlite3
from typing import Optional, Dict, Any, List

from loguru import logger

from tldw_Server_API.app.core.DB_Management.DB_Manager import DatabaseError
from tldw_Server_API.app.core.DB_Management.Media_DB import Database


def get_full_media_details2(media_id: int, db_instance: Database = None): # Use TypedDict in return hint
    """
    Get complete media details including keywords and all versions.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("A valid Database instance must be provided.")

    logger.debug(f"Attempting to get full details for ID: {media_id} on DB: {db_instance.db_path}")
    try:
        with db_instance.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 1. Get basic media info
            cursor.execute('''
                SELECT
                    id, url, title, type, content, author, ingestion_date,
                    transcription_model, is_trash, trash_date,
                    vector_embedding, chunking_status, vector_processing, content_hash
                FROM Media WHERE id = ?
            ''', (media_id,))
            media_row = cursor.fetchone()

            if not media_row:
                logger.warning(f"No media found for ID {media_id} in DB {db_instance.db_path}.")
                return None

            # 2. Populate the dictionary, ensuring types
            media_dict = {
                "id": media_row['id'],
                "url": media_row['url'],
                "title": media_row['title'],
                "type": media_row['type'],
                "content": media_row['content'],
                "author": media_row['author'],
                "ingestion_date": media_row['ingestion_date'],
                "transcription_model": media_row['transcription_model'],
                "is_trash": bool(media_row['is_trash']), # Ensure bool
                "trash_date": media_row['trash_date'],
                "vector_embedding": media_row['vector_embedding'],
                "chunking_status": media_row['chunking_status'],
                "vector_processing": media_row['vector_processing'],
                "content_hash": media_row['content_hash'],
                "keywords": [], # Initialize as empty list
                "versions": []  # Initialize as empty list
            }

            # 3. Get keywords
            cursor.execute('''
                SELECT k.keyword FROM Keywords k JOIN MediaKeywords mk ON k.id = mk.keyword_id
                WHERE mk.media_id = ? ORDER BY k.keyword COLLATE NOCASE
            ''', (media_id,))
            # Assign directly to the key
            media_dict["keywords"] = [row['keyword'] for row in cursor.fetchall()]
            logger.debug(f"Keywords fetched: {media_dict['keywords']}")

        # 4. Get versions (outside the 'with' block for the connection)
        # Assign directly to the key
        media_dict["versions"] = get_all_document_versions(
            media_id=media_id,
            include_content=False,
            db_instance=db_instance
        )
        logger.debug(f"Versions fetched: {len(media_dict['versions'])} versions found.")

        # Cast the final dictionary to the TypedDict type before returning
        # This helps the type checker verify the structure.
        return media_dict # Pylance should understand this structure now

    except sqlite3.Error as e:
        logger.error(f"Database error getting full media details for ID {media_id} on {db_instance.db_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting full media details for ID {media_id} on {db_instance.db_path}: {e}", exc_info=True)
        return None


def create_document_version(
        media_id: int,
        content: str,
        prompt: Optional[str] = None,
        analysis_content: Optional[str] = None,
        db_instance: Database = None,
        # Add conn parameter for use within existing transactions
        conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Any]:
    """
    Creates a new document version record in the DocumentVersions table.

    Args:
        media_id: The ID of the associated media item.
        content: The content snapshot for this version.
        prompt: The prompt associated with this version's analysis (optional).
        analysis_content: The analysis associated with this version's content (optional).
        db_instance: The Database instance (required if conn is not provided).
        conn: An existing sqlite3.Connection (optional, used for transactions).

    Returns:
        A dictionary containing the new version_number and media_id.

    Raises:
        DatabaseError: If the database operation fails.
        ValueError: If neither db_instance nor conn is provided.
    """
    if conn is None and db_instance is None:
        raise ValueError("Either db_instance or conn must be provided.")
    if conn and not isinstance(conn, sqlite3.Connection):
         raise TypeError("Provided conn must be a valid sqlite3.Connection.")
    if db_instance and not isinstance(db_instance, Database):
         raise TypeError("Provided db_instance must be a valid Database object.")

    # Prefer using the passed connection if available (for transactions)
    db_to_use = db_instance if db_instance else None # Primarily for logging path
    log_path = db_to_use.db_path if db_to_use else "existing connection"
    logging.debug(f"Creating document version for media_id={media_id} on DB: {log_path}")

    # Define the operation as a function to run within transaction or directly
    def _create_version_operation(connection: sqlite3.Connection) -> Dict[str, Any]:
        try:
            cursor = connection.cursor()

            # --- Get the next version number ---
            cursor.execute('''
                SELECT COALESCE(MAX(version_number), 0) + 1
                FROM DocumentVersions
                WHERE media_id = ?
            ''', (media_id,))
            version_number_result = cursor.fetchone()
            # Ensure we handle the case where fetchone might return None (though COALESCE should prevent it)
            version_number = version_number_result[0] if version_number_result else 1

            logging.debug(f"Determined next version number: {version_number} for media_id={media_id}")

            # --- Insert the new version record ---
            # Note: prompt and analysis_content can be NULL in the table
            cursor.execute('''
                INSERT INTO DocumentVersions
                (media_id, version_number, content, prompt, analysis_content, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (media_id, version_number, content, prompt, analysis_content))

            logging.info(f"Successfully created version {version_number} for media_id={media_id}")

            # Return essential info
            return {
                'media_id': media_id,
                'version_number': version_number,
                # Add content length if needed, but usually not required
                # 'content_length': len(content)
            }
        except sqlite3.IntegrityError as ie:
             # This might happen if (media_id, version_number) UNIQUE constraint is violated (race condition?)
             logging.error(f"Integrity error creating version for media_id={media_id}: {ie}", exc_info=True)
             raise DatabaseError(f"Failed to create document version due to integrity constraint: {ie}") from ie
        except sqlite3.Error as e:
            logging.error(f"SQLite error creating version {version_number} for media_id={media_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to create document version: {e}") from e

    # Execute the operation
    try:
        if conn:
            # Run directly using the provided connection (part of an outer transaction)
            return _create_version_operation(conn)
        else:
            # Run within its own transaction using the db_instance
            with db_to_use.transaction() as new_conn:
                return _create_version_operation(new_conn)
    except DatabaseError: # Re-raise DatabaseErrors caught inside
        raise
    except Exception as e:
        logging.error(f"Unexpected error wrapper creating version for media_id={media_id}: {e}", exc_info=True)
        raise DatabaseError(f"An unexpected error occurred: {e}") from e


def get_document_version(
        media_id: int,
        version_number: Optional[int] = None,
        include_content: bool = True,
        db_instance: Database = None
) -> Optional[Dict[str, Any]]: # Return Optional[Dict] or raise error
    """
    Get a specific document version or the latest version for a media item.

    Args:
        media_id: The ID of the media item.
        version_number: The specific version number to retrieve. If None, retrieves the latest.
        include_content: Whether to include the full 'content' field.
        db_instance: The Database instance to use.

    Returns:
        A dictionary representing the document version, or None if not found.
        The dictionary contains 'id', 'media_id', 'version_number', 'created_at',
        'prompt', 'analysis_content', and optionally 'content'.

    Raises:
        DatabaseError: If a database query fails.
        TypeError: If db_instance is not a valid Database object.
        ValueError: If version_number is provided but is not a positive integer.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("A valid Database instance must be provided.")
    if version_number is not None and (not isinstance(version_number, int) or version_number < 1):
        raise ValueError("Version number must be a positive integer.")

    log_msg = f"Getting {'latest' if version_number is None else f'version {version_number}'} for media_id={media_id}"
    logging.debug(f"{log_msg} from DB: {db_instance.db_path} (Include content: {include_content})")

    try:
        # Use transaction context for connection management
        with db_instance.transaction() as conn:
            cursor = conn.cursor()

            # Construct the SELECT clause dynamically
            select_cols = "id, version_number, created_at, prompt, analysis_content"
            if include_content:
                select_cols += ", content"

            params = [media_id]
            if version_number is None:
                # Get latest version
                query = f'''
                    SELECT {select_cols}
                    FROM DocumentVersions
                    WHERE media_id = ?
                    ORDER BY version_number DESC
                    LIMIT 1
                '''
            else:
                # Get specific version
                query = f'''
                    SELECT {select_cols}
                    FROM DocumentVersions
                    WHERE media_id = ? AND version_number = ?
                '''
                params.append(version_number)

            cursor.execute(query, tuple(params))
            result = cursor.fetchone() # Fetch using the row factory

            if not result:
                logging.warning(f"Version {'latest' if version_number is None else version_number} not found for media_id {media_id}")
                return None # Return None instead of {'error': ...}

            # Convert Row object to dictionary
            version_data = dict(result) # Convert Row to dict
            version_data['media_id'] = media_id # Ensure media_id is present

            return version_data

    except sqlite3.Error as e:
        logging.error(f"SQLite error retrieving {log_msg}: {e}", exc_info=True)
        raise DatabaseError(f"Database error retrieving version: {e}") from e
    except Exception as e:
         logging.error(f"Unexpected error retrieving {log_msg}: {e}", exc_info=True)
         raise DatabaseError(f"Unexpected error retrieving version: {e}") from e


def get_all_document_versions(
        media_id: int,
        include_content: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = 0, # Default offset to 0
        db_instance: Database = None
) -> List[Dict[str, Any]]:
    """
    Get all versions for a media item with pagination, including prompt and analysis_content.

    Args:
        media_id: The ID of the media item.
        include_content: Whether to include the full content of each version.
        limit: Maximum number of versions to return. None for no limit.
        offset: Number of versions to skip (for pagination). Defaults to 0.
        db_instance: The Database instance.

    Returns:
        A list of dictionaries, each representing a document version. Returns empty list if none found or on error.

    Raises:
        DatabaseError: If a database query fails.
        TypeError: If db_instance is not a valid Database object.
        ValueError: If limit or offset are invalid.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("A valid Database instance must be provided.")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("Limit must be a positive integer.")
    if offset is not None and (not isinstance(offset, int) or offset < 0):
         raise ValueError("Offset must be a non-negative integer.")

    logging.debug(f"Getting all versions for media_id={media_id} (Limit={limit}, Offset={offset}, Content={include_content}) from DB: {db_instance.db_path}")

    try:
        # Use transaction context for connection management
        with db_instance.transaction() as conn:
            cursor = conn.cursor()

            # Include prompt and analysis_content in the selection
            select_clause = 'id, version_number, created_at, prompt, analysis_content'
            if include_content:
                select_clause += ', content'

            query = f'''
                    SELECT {select_clause}
                    FROM DocumentVersions
                    WHERE media_id = ?
                    ORDER BY version_number DESC
                '''

            params = [media_id]

            # Apply limit and offset if specified
            # Note: SQLite requires LIMIT before OFFSET if both are used.
            if limit is not None:
                query += ' LIMIT ?'
                params.append(limit)
                # OFFSET only makes sense if LIMIT is also applied
                if offset is not None and offset > 0:
                    query += ' OFFSET ?'
                    params.append(offset)

            logging.debug(f"Executing get_all_document_versions query | Params: {params}")
            cursor.execute(query, tuple(params))
            results_raw = cursor.fetchall()

            # Convert rows to dictionaries using the Row factory's dict conversion
            versions_list = [dict(row) for row in results_raw]
            # Add media_id for context if needed elsewhere (optional)
            for v in versions_list:
                v['media_id'] = media_id

            logging.debug(f"Found {len(versions_list)} versions for media_id={media_id}")
            return versions_list

    except sqlite3.Error as e:
        logging.error(f"SQLite error retrieving versions for media_id {media_id} from {db_instance.db_path}: {e}",
                     exc_info=True)
        # Return empty list on error as per original docstring
        return []
    except Exception as e:
        logging.error(f"Unexpected error retrieving versions for media_id {media_id} from {db_instance.db_path}: {e}",
                     exc_info=True)
        return []


def delete_document_version(media_id: int, version_number: int, db_instance: Database) -> Dict[str, Any]:
    """
    Delete a specific document version.

    Returns {'error': message} if:
      - The version doesn't exist.
      - It's the last existing version for the media item.
    Returns {'success': message} on successful deletion.

    Args:
        media_id: The ID of the media item.
        version_number: The specific version number to delete.
        db_instance: The Database instance.

    Raises:
        DatabaseError: If a database query fails unexpectedly.
        TypeError: If db_instance is not a valid Database object.
        ValueError: If version_number is invalid.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("A valid Database instance must be provided.")
    if not isinstance(version_number, int) or version_number < 1:
        raise ValueError("Version number must be a positive integer.")

    logging.debug(f"Attempting to delete version {version_number} for media_id={media_id} from DB: {db_instance.db_path}")

    try:
        # Use a transaction to ensure atomicity of checks and delete
        with db_instance.transaction() as conn:
            cursor = conn.cursor()

            # Check how many total versions exist for this media item
            cursor.execute('''
                SELECT COUNT(*) FROM DocumentVersions
                WHERE media_id = ?
            ''', (media_id,))
            count_result = cursor.fetchone()
            total_versions = count_result[0] if count_result else 0

            if total_versions <= 1:
                logging.warning(f"Attempted to delete the last version ({version_number}) for media_id={media_id}")
                return {'error': 'Cannot delete the last version'}

            # Check if the target version exists before attempting delete
            cursor.execute('''
                SELECT 1 FROM DocumentVersions
                WHERE media_id = ? AND version_number = ?
            ''', (media_id, version_number))
            exists = cursor.fetchone()

            if not exists:
                logging.warning(f"Version {version_number} not found for deletion for media_id={media_id}")
                return {'error': 'Version not found'}

            # Perform the delete operation
            cursor.execute('''
                DELETE FROM DocumentVersions
                WHERE media_id = ? AND version_number = ?
            ''', (media_id, version_number))
            rows_affected = cursor.rowcount

            if rows_affected > 0:
                 logging.info(f"Successfully deleted version {version_number} for media_id={media_id}")
                 return {'success': f'Version {version_number} deleted successfully'}
            else:
                 # Should not happen if exists check passed, but handle defensively
                 logging.error(f"Version {version_number} found but delete affected 0 rows for media_id={media_id}")
                 return {'error': 'Deletion failed unexpectedly after existence check'}

    except sqlite3.Error as e:
        logging.error(f"SQLite error deleting version {version_number} for media_id={media_id}: {e}", exc_info=True)
        # Don't return {'error': str(e)} here, raise a proper exception
        raise DatabaseError(f"Database error deleting version: {e}") from e
    except Exception as e:
        logging.error(f"Unexpected error deleting version {version_number} for media_id={media_id}: {e}", exc_info=True)
        raise DatabaseError(f"Unexpected error deleting version: {e}") from e


def rollback_to_version(
        media_id: int,
        version_number: int,
        db_instance: Database
) -> Dict[str, Any]:
    """
    Rolls back the main Media record to a previous version's state by:
    1. Fetching the content, prompt, and analysis from the target version.
    2. Creating a NEW version record in DocumentVersions with this fetched data.
    3. Updating the main Media table's 'content' and 'content_hash' fields
       to match the rolled-back content.

    Args:
        media_id: The ID of the media item.
        version_number: The version number to roll back to.
        db_instance: The Database instance.

    Returns:
        A dictionary indicating success and the new version number created,
        or an error dictionary. Example: {'success': msg, 'new_version_number': num} or {'error': msg}

    Raises:
        DatabaseError: If database operations fail.
        TypeError: If db_instance is not valid.
        ValueError: If version_number is invalid.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("A valid Database instance must be provided.")
    if not isinstance(version_number, int) or version_number < 1:
        raise ValueError("Version number must be a positive integer.")

    logging.debug(f"Attempting rollback to version {version_number} for media_id={media_id} on DB: {db_instance.db_path}")

    try:
        # Use a single transaction for all operations
        with db_instance.transaction() as conn:
            cursor = conn.cursor()

            # --- 1. Get the target version data ---
            # Use the updated get_document_version
            target_version_data = get_document_version(
                media_id=media_id,
                version_number=version_number,
                include_content=True,
                db_instance=db_instance # Pass instance, get_document_version will use the transaction's conn
            )

            if target_version_data is None:
                logging.warning(f"Rollback failed: Target version {version_number} not found for media_id={media_id}")
                return {'error': f'Version {version_number} not found'}

            target_content = target_version_data.get('content')
            target_prompt = target_version_data.get('prompt')
            target_analysis = target_version_data.get('analysis_content')

            # Ensure content exists before proceeding
            if target_content is None:
                 logging.error(f"Rollback failed: Target version {version_number} for media_id={media_id} has NULL content.")
                 return {'error': f'Version {version_number} has no content to roll back to.'}

            # --- 2. Create a *new* version reflecting the rollback state ---
            # Pass the connection 'conn' to run within the current transaction
            new_version_info = create_document_version(
                media_id=media_id,
                content=target_content,
                prompt=target_prompt,
                analysis_content=target_analysis,
                db_instance=db_instance, # Still pass instance for logging etc.
                conn=conn # Pass the active connection!
            )
            new_version_number = new_version_info.get('version_number')
            if not new_version_number:
                 # This shouldn't happen if create_document_version is correct
                 logging.error(f"Rollback failed: create_document_version did not return a version number for media_id={media_id}")
                 raise DatabaseError("Failed to get new version number during rollback.")

            logging.debug(f"Created new version {new_version_number} during rollback for media_id={media_id}")

            # --- 3. Update the main Media table ---
            # Calculate the hash of the rolled-back content
            new_content_hash = hashlib.sha256(target_content.encode()).hexdigest()

            # Update Media.content and Media.content_hash
            # Also update transcription_model if relevant? Maybe copy from original rolled-back version?
            # For now, just update content and hash.
            cursor.execute('''
                UPDATE Media
                SET content = ?,
                    content_hash = ?
                WHERE id = ?
            ''', (target_content, new_content_hash, media_id))
            rows_affected = cursor.rowcount

            if rows_affected == 0:
                 # This indicates the media_id doesn't exist in the Media table, a major inconsistency.
                 logging.error(f"Rollback warning: Media record for ID {media_id} not found during update.")
                 # Rollback might still be considered partially successful as version was created,
                 # but raise an error because the main record wasn't updated.
                 raise DatabaseError(f"Media record {media_id} not found for final rollback update.")

            logging.info(f"Successfully rolled back media_id={media_id} to state of version {version_number} (New version: {new_version_number})")

            # Commit happens automatically when 'with' block exits without error

            return {
                'success': f'Successfully rolled back to version {version_number}. State saved as new version {new_version_number}.',
                'new_version_number': new_version_number
            }

    except sqlite3.Error as e:
        logging.error(f"SQLite error during rollback for media_id={media_id} to version {version_number}: {e}", exc_info=True)
        # Re-raise as specific error
        raise DatabaseError(f"Database error during rollback: {e}") from e
    except DatabaseError as de: # Catch errors raised by helpers
        logging.error(f"DatabaseError during rollback for media_id={media_id} to version {version_number}: {de}", exc_info=True)
        raise # Re-raise
    except Exception as e:
        logging.error(f"Unexpected error during rollback for media_id={media_id} to version {version_number}: {e}", exc_info=True)
        raise DatabaseError(f"Unexpected error during rollback: {e}") from e

# transaction_utils.py
# Description: Database transaction utilities for ensuring ACID properties
#
# Imports
import asyncio
import functools
from contextlib import asynccontextmanager
from typing import Callable, Optional, TypeVar

from loguru import logger

from tldw_Server_API.app.core.DB_Management.async_db_wrapper import AsyncDatabaseWrapper
from tldw_Server_API.app.core.Utils.backoff import capped_exponential_delay
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

#######################################################################################################################
#
# Type definitions:

T = TypeVar('T')

#######################################################################################################################
#
# Functions:

@asynccontextmanager
async def db_transaction(db: CharactersRAGDB, max_retries: int = 3):
    """
    Async context manager for database transactions with automatic retry logic.

    Note: This wraps the synchronous transaction manager on the calling thread.
    Use with care in async code: execute blocking DB work via run_in_executor
    or prefer `run_transaction` below to execute a function inside a single
    executor thread that holds the transaction for its entire duration.

    Args:
        db: Database instance
        max_retries: Maximum number of retries for transient failures

    Yields:
        Transaction context

    Raises:
        CharactersRAGDBError: On database errors after retries exhausted
    """
    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        entered_context = False
        try:
            # Retry is only safe before entering/yielding the transactional block.
            # After the caller's block runs, we must not attempt a second yield.
            with db.transaction():
                entered_context = True
                yield db
                return

        except ConflictError as e:
            if entered_context:
                # Conflict occurred after the caller's block started; an async
                # context manager cannot safely re-yield to replay user code.
                raise
            retry_count += 1
            last_error = e
            if retry_count < max_retries:
                # Shared short capped-exponential schedule for in-process contention.
                # jitter=False preserves this call site's existing timing; enabling it
                # is a deliberate follow-up, not part of the consolidation.
                wait_time = capped_exponential_delay(
                    retry_count, base_s=0.1, jitter=False
                )
                logger.warning(f"Transaction conflict, retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Transaction failed after {max_retries} retries due to conflicts")
                raise

        except InputError as e:
            # Input validation error - don't retry
            logger.error(f"Transaction failed due to input error: {e}")
            raise

        except CharactersRAGDBError as e:
            # Other database errors - don't retry
            logger.error(f"Transaction failed due to database error: {e}")
            raise

        except Exception as e:
            # Unexpected error - log and re-raise
            logger.error(f"Unexpected error in transaction: {e}", exc_info=True)
            raise CharactersRAGDBError(f"Transaction failed: {str(e)}") from e

    # If we get here, all retries exhausted
    if last_error:
        raise last_error
    else:
        raise CharactersRAGDBError(f"Transaction failed after {max_retries} retries")


async def run_transaction(
    adb: AsyncDatabaseWrapper,
    fn: Callable[[CharactersRAGDB], T],
    *,
    max_retries: int = 3,
) -> T:
    """
    Execute a synchronous function within a single-threaded transaction safely.

    This ensures that the transaction, and all DB calls inside `fn`, run on the
    same worker thread (using the wrapper's executor). This is the recommended
    pattern for multi-step operations from async code when you need true
    transactional guarantees.

    Args:
        adb: AsyncDatabaseWrapper for a CharactersRAGDB instance
        fn: A callable that receives the underlying CharactersRAGDB and returns a value
        max_retries: Number of retries on ConflictError

    Returns:
        The value returned by `fn`.
    """
    loop = asyncio.get_running_loop()
    retries = 0
    last_err: Optional[Exception] = None

    def _run_once() -> T:  # type: ignore[override]
        with adb.db.transaction():
            return fn(adb.db)

    while retries < max_retries:
        try:
            return await loop.run_in_executor(adb._executor, _run_once)  # type: ignore[attr-defined]
        except ConflictError as ce:
            last_err = ce
            retries += 1
            if retries < max_retries:
                wait_time = 0.1 * (2 ** retries)
                logger.warning(
                    f"Transaction conflict in run_transaction, retrying in {wait_time}s (attempt {retries}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
            else:
                raise
        except InputError:
            raise
        except CharactersRAGDBError:
            # Non-conflict DB errors: surface to caller
            raise
        except Exception as e:  # noqa: BLE001
            # Unexpected error
            raise CharactersRAGDBError(f"Transaction failed: {e}") from e

    # Exhausted retries
    if last_err:
        raise last_err
    raise CharactersRAGDBError("Transaction failed after retries")


def transactional(max_retries: int = 3):
    """
    Decorator to make a function transactional with automatic retry logic.

    Args:
        max_retries: Maximum number of retries for transient failures

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to find the db parameter
            db = None
            for arg in args:
                if isinstance(arg, CharactersRAGDB):
                    db = arg
                    break
            if not db:
                db = kwargs.get('db')

            if not db:
                # No database parameter found, run without transaction
                return await func(*args, **kwargs)

            async with db_transaction(db, max_retries):
                return await func(*args, **kwargs)

        return wrapper
    return decorator


async def save_conversation_with_messages(
    db: CharactersRAGDB | AsyncDatabaseWrapper,
    conversation_data: dict,
    messages: list[dict],
    max_retries: int = 3
) -> tuple[str, list[str]]:
    """
    Save a conversation and its messages in a single transaction.

    Args:
        db: Database instance
        conversation_data: Conversation data dictionary
        messages: List of message dictionaries
        max_retries: Maximum retry attempts

    Returns:
        Tuple of (conversation_id, list of message_ids)

    Raises:
        CharactersRAGDBError: On database errors
    """
    # If provided an AsyncDatabaseWrapper, execute within a single executor-held transaction
    if isinstance(db, AsyncDatabaseWrapper):
        def _work(sync_db: CharactersRAGDB) -> tuple[str, list[str]]:
            conversation_id = sync_db.add_conversation(conversation_data)
            if not conversation_id:
                raise CharactersRAGDBError("Failed to create conversation")
            message_ids: list[str] = []
            for message in messages:
                message_payload = dict(message)
                message_payload['conversation_id'] = conversation_id
                message_id = sync_db.add_message(message_payload)
                if not message_id:
                    raise CharactersRAGDBError(
                        f"Failed to add message to conversation {conversation_id}"
                    )
                message_ids.append(message_id)
            return conversation_id, message_ids

        return await run_transaction(db, _work, max_retries=max_retries)

    # Fallback: use async transaction wrapper that may block event loop if used directly in async contexts
    async with db_transaction(db, max_retries):
        conversation_id = db.add_conversation(conversation_data)  # type: ignore[attr-defined]
        if not conversation_id:
            raise CharactersRAGDBError("Failed to create conversation")
        message_ids = []
        for message in messages:
            message_payload = dict(message)
            message_payload['conversation_id'] = conversation_id
            message_id = db.add_message(message_payload)  # type: ignore[attr-defined]
            if not message_id:
                raise CharactersRAGDBError(
                    f"Failed to add message to conversation {conversation_id}"
                )
            message_ids.append(message_id)
        return conversation_id, message_ids


async def update_conversation_with_rollback(
    db: CharactersRAGDB | AsyncDatabaseWrapper,
    conversation_id: str,
    updates: dict,
    new_messages: Optional[list[dict]] = None
) -> bool:
    """
    Update a conversation and optionally add new messages, with rollback on failure.

    Args:
        db: Database instance
        conversation_id: ID of the conversation to update
        updates: Dictionary of updates to apply
        new_messages: Optional list of new messages to add

    Returns:
        True if successful, False otherwise
    """
    try:
        if isinstance(db, AsyncDatabaseWrapper):
            def _work(sync_db: CharactersRAGDB) -> bool:
                if updates:
                    success = sync_db.update_conversation(conversation_id, updates)
                    if not success:
                        raise CharactersRAGDBError(
                            f"Failed to update conversation {conversation_id}"
                        )
                if new_messages:
                    for message in new_messages:
                        message_payload = dict(message)
                        message_payload['conversation_id'] = conversation_id
                        message_id = sync_db.add_message(message_payload)
                        if not message_id:
                            raise CharactersRAGDBError(
                                f"Failed to add message to conversation {conversation_id}"
                            )
                return True

            return await run_transaction(db, _work)

        async with db_transaction(db):
            if updates:
                success = db.update_conversation(conversation_id, updates)  # type: ignore[attr-defined]
                if not success:
                    raise CharactersRAGDBError(
                        f"Failed to update conversation {conversation_id}"
                    )
            if new_messages:
                for message in new_messages:
                    message_payload = dict(message)
                    message_payload['conversation_id'] = conversation_id
                    message_id = db.add_message(message_payload)  # type: ignore[attr-defined]
                    if not message_id:
                        raise CharactersRAGDBError(
                            f"Failed to add message to conversation {conversation_id}"
                        )
            return True
    except Exception as e:
        logger.error(f"Failed to update conversation {conversation_id}: {e}", exc_info=True)
        return False


#
# End of transaction_utils.py
#######################################################################################################################

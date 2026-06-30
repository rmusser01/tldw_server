# VoiceAssistant/db_helpers.py
# Database helper functions for persisting voice commands and sessions
#
#######################################################################################################################
import json
import uuid
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from .schemas import ActionType, VoiceCommand, VoiceSessionContext, VoiceSessionState

VOICE_EVENT_RESOLUTION_DIRECT = "direct_command"
VOICE_EVENT_RESOLUTION_FALLBACK = "planner_fallback"
PERSONA_LIVE_VOICE_EVENT_COMMIT = "commit"
PERSONA_LIVE_VOICE_EVENT_MANUAL_MODE_REQUIRED = "manual_mode_required"


def save_voice_command(
    db,
    command: VoiceCommand,
) -> str:
    """
    Save a voice command to the database.

    Args:
        db: CharactersRAGDB instance
        command: VoiceCommand to save

    Returns:
        The command ID
    """
    command_id = command.id or str(uuid.uuid4())
    db.upsert_voice_command(
        command_id=command_id,
        user_id=command.user_id,
        persona_id=command.persona_id,
        connection_id=command.connection_id,
        name=command.name,
        phrases=command.phrases,
        action_type=command.action_type.value,
        action_config=command.action_config,
        priority=command.priority,
        enabled=command.enabled,
        requires_confirmation=command.requires_confirmation,
        description=command.description,
        created_at=(
            command.created_at.isoformat()
            if command.created_at is not None
            else datetime.utcnow().isoformat()
        ),
        updated_at=datetime.utcnow().isoformat(),
    )

    logger.debug(f"Saved voice command: {command.name} (id={command_id})")
    return command_id


def get_voice_command(
    db,
    command_id: str,
    user_id: Optional[int] = None,
    persona_id: Optional[str] = None,
) -> Optional[VoiceCommand]:
    """
    Get a voice command by ID.

    Args:
        db: CharactersRAGDB instance
        command_id: Command ID
        user_id: Optional user ID filter

    Returns:
        VoiceCommand if found, None otherwise
    """
    query = "SELECT * FROM voice_commands WHERE id = ? AND deleted = 0"
    params = [command_id]

    if user_id is not None:
        query += " AND user_id = ?"
        params.append(user_id)

    if persona_id is not None:
        query += " AND persona_id = ?"
        params.append(persona_id)

    result = db.execute_query(query, tuple(params))
    rows = result.fetchall() if hasattr(result, 'fetchall') else list(result)

    if not rows:
        return None

    return _row_to_voice_command(rows[0])


def get_user_voice_commands(
    db,
    user_id: int,
    include_system: bool = True,
    enabled_only: bool = True,
    persona_id: Optional[str] = None,
) -> list[VoiceCommand]:
    """
    Get all voice commands for a user.

    Args:
        db: CharactersRAGDB instance
        user_id: User ID
        include_system: Include system commands (user_id=0)
        enabled_only: Only return enabled commands

    Returns:
        List of VoiceCommand objects
    """
    conditions = ["deleted = 0"]
    params = []

    if include_system:
        if persona_id is not None:
            conditions.append("((user_id = ? AND persona_id = ?) OR user_id = 0)")
            params.extend([user_id, persona_id])
        else:
            conditions.append("(user_id = ? OR user_id = 0)")
            params.append(user_id)
    else:
        conditions.append("user_id = ?")
        params.append(user_id)
        if persona_id is not None:
            conditions.append("persona_id = ?")
            params.append(persona_id)

    if enabled_only:
        conditions.append("enabled = 1")

    where_clause = " AND ".join(conditions)
    query_template = """
        SELECT * FROM voice_commands
        WHERE {where_clause}
        ORDER BY priority DESC, name ASC
    """
    query = query_template.format_map(locals())  # nosec B608

    result = db.execute_query(query, tuple(params))
    rows = result.fetchall() if hasattr(result, 'fetchall') else list(result)

    return [_row_to_voice_command(row) for row in rows]


def delete_voice_command(
    db,
    command_id: str,
    user_id: int,
    hard_delete: bool = False,
) -> bool:
    """
    Delete a voice command.

    Args:
        db: CharactersRAGDB instance
        command_id: Command ID
        user_id: User ID (must match for non-system commands)
        hard_delete: If True, permanently delete; otherwise soft delete

    Returns:
        True if deleted, False if not found or not authorized
    """
    # Check command exists and belongs to user
    command = get_voice_command(db, command_id)
    if not command:
        return False

    # Can't delete system commands (user_id=0) unless admin
    if command.user_id == 0:
        return False

    if command.user_id != user_id:
        return False

    with db.transaction():
        if hard_delete:
            db.execute_query(
                "DELETE FROM voice_commands WHERE id = ?",
                (command_id,),
            )
        else:
            db.execute_query(
                "UPDATE voice_commands SET deleted = 1, updated_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), command_id),
            )

    logger.debug(f"Deleted voice command: {command_id}")
    return True


def save_voice_session(
    db,
    session: VoiceSessionContext,
) -> str:
    """
    Save a voice session to the database.

    Args:
        db: CharactersRAGDB instance
        session: VoiceSessionContext to save

    Returns:
        The session ID
    """
    with db.transaction():
        db.execute_query(
            """
            INSERT INTO voice_sessions (
                session_id, user_id, state, context,
                conversation_history, pending_intent, last_action_result,
                created_at, last_activity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                state = excluded.state,
                context = excluded.context,
                conversation_history = excluded.conversation_history,
                pending_intent = excluded.pending_intent,
                last_action_result = excluded.last_action_result,
                last_activity = excluded.last_activity
            """,
            (
                session.session_id,
                session.user_id,
                session.state.value,
                json.dumps(session.metadata),
                json.dumps(session.conversation_history),
                json.dumps(session.pending_intent.model_dump()) if session.pending_intent else None,
                json.dumps(session.last_action_result) if session.last_action_result else None,
                session.created_at.isoformat(),
                session.last_activity.isoformat(),
            ),
        )

    logger.debug(f"Saved voice session: {session.session_id}")
    return session.session_id


def get_voice_session(
    db,
    session_id: str,
) -> Optional[VoiceSessionContext]:
    """
    Get a voice session by ID.

    Args:
        db: CharactersRAGDB instance
        session_id: Session ID

    Returns:
        VoiceSessionContext if found, None otherwise
    """
    result = db.execute_query(
        "SELECT * FROM voice_sessions WHERE session_id = ?",
        (session_id,),
    )
    rows = result.fetchall() if hasattr(result, 'fetchall') else list(result)

    if not rows:
        return None

    return _row_to_voice_session(rows[0])


def get_user_voice_sessions(
    db,
    user_id: int,
    limit: int = 10,
    offset: int = 0,
    active_after: Optional[datetime] = None,
) -> list[VoiceSessionContext]:
    """
    Get recent voice sessions for a user.

    Args:
        db: CharactersRAGDB instance
        user_id: User ID
        limit: Maximum sessions to return
        offset: Zero-based pagination offset
        active_after: Optional lower bound for active session last activity

    Returns:
        List of VoiceSessionContext objects
    """
    query = """
        SELECT * FROM voice_sessions
        WHERE user_id = ?
    """
    params: list[Any] = [user_id]

    if active_after is not None:
        query += " AND last_activity >= ?"
        params.append(active_after.isoformat())

    query += " ORDER BY last_activity DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    result = db.execute_query(query, tuple(params))
    rows = result.fetchall() if hasattr(result, 'fetchall') else list(result)

    return [_row_to_voice_session(row) for row in rows]


def count_user_voice_sessions(
    db,
    user_id: int,
    active_after: Optional[datetime] = None,
) -> int:
    """
    Count voice sessions for a user.

    Args:
        db: CharactersRAGDB instance
        user_id: User ID
        active_after: Optional lower bound for active session last activity

    Returns:
        Number of matching voice sessions
    """
    query = """
        SELECT COUNT(*) FROM voice_sessions
        WHERE user_id = ?
    """
    params: list[Any] = [user_id]

    if active_after is not None:
        query += " AND last_activity >= ?"
        params.append(active_after.isoformat())

    result = db.execute_query(query, tuple(params))
    rows = result.fetchall() if hasattr(result, 'fetchall') else list(result)
    if not rows:
        return 0
    row = rows[0]
    if isinstance(row, dict):
        return int(next(iter(row.values())))
    return int(row[0])


def delete_voice_session(
    db,
    session_id: str,
) -> bool:
    """
    Delete a voice session.

    Args:
        db: CharactersRAGDB instance
        session_id: Session ID

    Returns:
        True if deleted, False if not found
    """
    with db.transaction():
        result = db.execute_query(
            "DELETE FROM voice_sessions WHERE session_id = ?",
            (session_id,),
        )

    deleted = result.rowcount > 0 if hasattr(result, 'rowcount') else True
    if deleted:
        logger.debug(f"Deleted voice session: {session_id}")
    return deleted


def cleanup_old_sessions(
    db,
    max_age_hours: int = 24,
) -> int:
    """
    Clean up old voice sessions.

    Args:
        db: CharactersRAGDB instance
        max_age_hours: Maximum session age in hours

    Returns:
        Number of sessions deleted
    """
    with db.transaction():
        result = db.execute_query(
            """
            DELETE FROM voice_sessions
            WHERE last_activity < datetime('now', ?)
            """,
            (f"-{max_age_hours} hours",),
        )

    count = result.rowcount if hasattr(result, 'rowcount') else 0
    if count > 0:
        logger.info(f"Cleaned up {count} old voice sessions")
    return count


def record_voice_command_event(
    db,
    *,
    command_id: Optional[str],
    command_name: Optional[str],
    user_id: int,
    action_type: ActionType,
    success: bool,
    response_time_ms: Optional[float] = None,
    session_id: Optional[str] = None,
    persona_id: Optional[str] = None,
    resolution_type: str = VOICE_EVENT_RESOLUTION_DIRECT,
) -> None:
    """
    Record a voice command execution event for analytics.

    Args:
        db: CharactersRAGDB instance
        command_id: Command ID (nullable for fallback commands)
        command_name: Command name (nullable)
        user_id: User ID
        action_type: Action type for the command
        success: Whether the action succeeded
        response_time_ms: Optional response time
        session_id: Optional session ID
    """
    with db.transaction():
        db.execute_query(
            """
            INSERT INTO voice_command_events (
                command_id, command_name, user_id, persona_id, action_type,
                success, response_time_ms, session_id, resolution_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                command_id,
                command_name,
                user_id,
                persona_id,
                action_type.value,
                1 if success else 0,
                response_time_ms,
                session_id,
                resolution_type,
            ),
        )


def record_persona_live_voice_event(
    db,
    *,
    user_id: int,
    persona_id: Optional[str],
    session_id: Optional[str],
    event_type: str,
    commit_source: Optional[str] = None,
) -> None:
    """Persist a persona live-voice analytics event for the active session."""
    with db.transaction():
        db.execute_query(
            """
            INSERT INTO persona_live_voice_events (
                user_id, persona_id, session_id, event_type, commit_source
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                persona_id,
                session_id,
                event_type,
                commit_source,
            ),
        )


def _build_voice_event_filters(
    *,
    user_id: Optional[int] = None,
    command_id: Optional[str] = None,
    days: Optional[int] = None,
    persona_id: Optional[str] = None,
    resolution_type: Optional[str] = None,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    if command_id is not None:
        clauses.append("command_id = ?")
        params.append(command_id)
    if user_id is not None:
        clauses.append("user_id = ?")
        params.append(user_id)
    if days is not None:
        clauses.append("created_at >= datetime('now', ?)")
        params.append(f"-{days} days")
    if persona_id is not None:
        clauses.append("persona_id = ?")
        params.append(persona_id)
    if resolution_type is not None:
        clauses.append("resolution_type = ?")
        params.append(resolution_type)

    if not clauses:
        return "", params
    return " WHERE " + " AND ".join(clauses), params


def get_voice_command_usage_stats(
    db,
    *,
    command_id: str,
    user_id: int,
    days: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """
    Get usage statistics for a specific voice command.

    Args:
        db: CharactersRAGDB instance
        command_id: Command ID
        user_id: User ID
        days: Optional lookback window in days

    Returns:
        Dict with usage stats or None if no data
    """
    params: list[Any] = [command_id, user_id]
    date_filter = ""
    if days is not None:
        date_filter = " AND created_at >= datetime('now', ?)"
        params.append(f"-{days} days")

    command_usage_sql_template = """
        SELECT
            command_id,
            MAX(command_name) AS command_name,
            COUNT(*) AS total_invocations,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS success_count,
            SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS error_count,
            AVG(response_time_ms) AS avg_response_time_ms,
            MAX(created_at) AS last_used
        FROM voice_command_events
        WHERE command_id = ? AND user_id = ?{date_filter}
        """
    command_usage_sql = command_usage_sql_template.format_map(locals())  # nosec B608
    result = db.execute_query(
        command_usage_sql,
        tuple(params),
    )
    rows = result.fetchall() if hasattr(result, 'fetchall') else list(result)
    if not rows:
        return None
    row = rows[0]
    if not isinstance(row, dict):
        row = dict(row)
    if not row or row.get("total_invocations") in (None, 0):
        return None

    return {
        "command_id": row.get("command_id"),
        "command_name": row.get("command_name"),
        "total_invocations": row.get("total_invocations") or 0,
        "success_count": row.get("success_count") or 0,
        "error_count": row.get("error_count") or 0,
        "avg_response_time_ms": row.get("avg_response_time_ms") or 0.0,
        "last_used": row.get("last_used"),
    }


def get_voice_top_commands(
    db,
    *,
    user_id: int,
    days: Optional[int] = None,
    limit: int = 10,
    persona_id: Optional[str] = None,
    resolution_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Get top voice commands by usage.

    Args:
        db: CharactersRAGDB instance
        user_id: User ID
        days: Optional lookback window in days
        limit: Maximum commands to return

    Returns:
        List of usage stats for top commands
    """
    where_clause, params = _build_voice_event_filters(
        user_id=user_id,
        days=days,
        persona_id=persona_id,
        resolution_type=resolution_type,
    )
    params.append(limit)

    top_commands_sql_template = """
        SELECT
            command_id,
            MAX(command_name) AS command_name,
            COUNT(*) AS total_invocations,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS success_count,
            SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS error_count,
            AVG(response_time_ms) AS avg_response_time_ms,
            MAX(created_at) AS last_used
        FROM voice_command_events
        {where_clause}
        GROUP BY command_id
        ORDER BY total_invocations DESC
        LIMIT ?
        """
    top_commands_sql = top_commands_sql_template.format_map(locals())  # nosec B608
    result = db.execute_query(
        top_commands_sql,
        tuple(params),
    )
    rows = result.fetchall() if hasattr(result, 'fetchall') else list(result)

    top_commands = []
    for row in rows:
        if not isinstance(row, dict):
            row = dict(row)
        top_commands.append(
            {
                "command_id": row.get("command_id"),
                "command_name": row.get("command_name"),
                "total_invocations": row.get("total_invocations") or 0,
                "success_count": row.get("success_count") or 0,
                "error_count": row.get("error_count") or 0,
                "avg_response_time_ms": row.get("avg_response_time_ms") or 0.0,
                "last_used": row.get("last_used"),
            }
        )
    return top_commands


def get_voice_usage_by_day(
    db,
    *,
    user_id: Optional[int] = None,
    days: int = 7,
) -> list[dict[str, Any]]:
    """
    Get daily voice usage metrics.

    Args:
        db: CharactersRAGDB instance
        user_id: Optional user ID filter
        days: Lookback window in days

    Returns:
        List of daily analytics dicts
    """
    params: list[Any] = [f"-{days} days"]
    user_filter = ""
    if user_id is not None:
        user_filter = " AND user_id = ?"
        params.append(user_id)

    usage_by_day_sql_template = """
        SELECT
            date(created_at) AS day,
            COUNT(*) AS total_commands,
            COUNT(DISTINCT user_id) AS unique_users,
            COALESCE(SUM(success) * 1.0 / NULLIF(COUNT(*), 0), 0.0) AS success_rate,
            AVG(response_time_ms) AS avg_response_time_ms
        FROM voice_command_events
        WHERE created_at >= datetime('now', ?){user_filter}
        GROUP BY day
        ORDER BY day ASC
        """
    usage_by_day_sql = usage_by_day_sql_template.format_map(locals())  # nosec B608
    result = db.execute_query(
        usage_by_day_sql,
        tuple(params),
    )
    rows = result.fetchall() if hasattr(result, 'fetchall') else list(result)

    usage_by_day = []
    for row in rows:
        if not isinstance(row, dict):
            row = dict(row)
        usage_by_day.append(
            {
                "date": row.get("day"),
                "total_commands": row.get("total_commands") or 0,
                "unique_users": row.get("unique_users") or 0,
                "success_rate": row.get("success_rate") or 0.0,
                "avg_response_time_ms": row.get("avg_response_time_ms") or 0.0,
            }
        )
    return usage_by_day


def get_voice_analytics_summary_stats(
    db,
    *,
    user_id: Optional[int] = None,
    days: int = 7,
    persona_id: Optional[str] = None,
    resolution_type: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get aggregate voice analytics stats.

    Args:
        db: CharactersRAGDB instance
        user_id: Optional user ID filter
        days: Lookback window in days

    Returns:
        Dict with total, success_rate, avg_response_time_ms
    """
    where_clause, params = _build_voice_event_filters(
        user_id=user_id,
        days=days,
        persona_id=persona_id,
        resolution_type=resolution_type,
    )

    aggregate_stats_sql_template = """
        SELECT
            COUNT(*) AS total_commands,
            COALESCE(SUM(success) * 1.0 / NULLIF(COUNT(*), 0), 0.0) AS success_rate,
            AVG(response_time_ms) AS avg_response_time_ms
        FROM voice_command_events
        {where_clause}
        """
    aggregate_stats_sql = aggregate_stats_sql_template.format_map(locals())  # nosec B608
    result = db.execute_query(
        aggregate_stats_sql,
        tuple(params),
    )
    rows = result.fetchall() if hasattr(result, 'fetchall') else list(result)
    if not rows:
        return {"total_commands": 0, "success_rate": 0.0, "avg_response_time_ms": 0.0}

    row = rows[0]
    if not isinstance(row, dict):
        row = dict(row)
    return {
        "total_commands": row.get("total_commands") or 0,
        "success_rate": row.get("success_rate") or 0.0,
        "avg_response_time_ms": row.get("avg_response_time_ms") or 0.0,
    }


def get_voice_resolution_stats(
    db,
    *,
    user_id: Optional[int] = None,
    days: int = 7,
    persona_id: Optional[str] = None,
    resolution_type: str,
) -> dict[str, Any]:
    where_clause, params = _build_voice_event_filters(
        user_id=user_id,
        days=days,
        persona_id=persona_id,
        resolution_type=resolution_type,
    )
    resolution_stats_sql_template = """
        SELECT
            COUNT(*) AS total_invocations,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS success_count,
            SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS error_count,
            AVG(response_time_ms) AS avg_response_time_ms,
            MAX(created_at) AS last_used
        FROM voice_command_events
        {where_clause}
        """
    resolution_stats_sql = resolution_stats_sql_template.format_map(locals())  # nosec B608
    result = db.execute_query(
        resolution_stats_sql,
        tuple(params),
    )
    rows = result.fetchall() if hasattr(result, 'fetchall') else list(result)
    if not rows:
        return {
            "total_invocations": 0,
            "success_count": 0,
            "error_count": 0,
            "avg_response_time_ms": 0.0,
            "last_used": None,
        }

    row = rows[0]
    if not isinstance(row, dict):
        row = dict(row)
    return {
        "total_invocations": row.get("total_invocations") or 0,
        "success_count": row.get("success_count") or 0,
        "error_count": row.get("error_count") or 0,
        "avg_response_time_ms": row.get("avg_response_time_ms") or 0.0,
        "last_used": row.get("last_used"),
    }


def get_persona_live_voice_summary(
    db,
    *,
    user_id: Optional[int] = None,
    days: int = 7,
    persona_id: Optional[str] = None,
) -> dict[str, Any]:
    clauses = ["created_at >= datetime('now', ?)"]
    params: list[Any] = [f"-{days} days"]
    if user_id is not None:
        clauses.append("user_id = ?")
        params.append(user_id)
    if persona_id is not None:
        clauses.append("persona_id = ?")
        params.append(persona_id)

    where_clause = " WHERE " + " AND ".join(clauses)

    commit_sql_template = """
        SELECT
            COUNT(*) AS total_committed_turns,
            SUM(CASE WHEN commit_source = 'vad_auto' THEN 1 ELSE 0 END) AS vad_auto_commit_count,
            SUM(CASE WHEN commit_source = 'manual' THEN 1 ELSE 0 END) AS manual_commit_count
        FROM persona_live_voice_events
        {where_clause}
          AND event_type = ?
        """
    commit_sql = commit_sql_template.format_map(locals())  # nosec B608
    commit_rows = db.execute_query(
        commit_sql,
        tuple([*params, PERSONA_LIVE_VOICE_EVENT_COMMIT]),
    ).fetchall()
    commit_row = dict(commit_rows[0]) if commit_rows else {}

    degraded_sql_template = """
        SELECT
            COUNT(DISTINCT session_id) AS degraded_session_count
        FROM persona_live_voice_events
        {where_clause}
          AND event_type = ?
        """
    degraded_sql = degraded_sql_template.format_map(locals())  # nosec B608
    degraded_rows = db.execute_query(
        degraded_sql,
        tuple([*params, PERSONA_LIVE_VOICE_EVENT_MANUAL_MODE_REQUIRED]),
    ).fetchall()
    degraded_row = dict(degraded_rows[0]) if degraded_rows else {}

    total_committed_turns = int(commit_row.get("total_committed_turns") or 0)
    vad_auto_commit_count = int(commit_row.get("vad_auto_commit_count") or 0)
    manual_commit_count = int(commit_row.get("manual_commit_count") or 0)

    return {
        "total_committed_turns": total_committed_turns,
        "vad_auto_commit_count": vad_auto_commit_count,
        "manual_commit_count": manual_commit_count,
        "vad_auto_rate": (
            float(vad_auto_commit_count) / float(total_committed_turns)
            if total_committed_turns
            else 0.0
        ),
        "manual_commit_rate": (
            float(manual_commit_count) / float(total_committed_turns)
            if total_committed_turns
            else 0.0
        ),
        "degraded_session_count": int(degraded_row.get("degraded_session_count") or 0),
    }


def get_active_voice_session_count(
    db,
    *,
    user_id: int,
    activity_window_seconds: int,
) -> int:
    """
    Count active voice sessions within a recent activity window.

    Args:
        db: CharactersRAGDB instance
        user_id: User ID
        activity_window_seconds: Window in seconds to consider active

    Returns:
        Active session count
    """
    result = db.execute_query(
        """
        SELECT COUNT(*) AS count
        FROM voice_sessions
        WHERE user_id = ?
          AND last_activity >= datetime('now', ?)
        """,
        (user_id, f"-{activity_window_seconds} seconds"),
    )
    rows = result.fetchall() if hasattr(result, 'fetchall') else list(result)
    if not rows:
        return 0
    row = rows[0]
    if not isinstance(row, dict):
        row = dict(row)
    return int(row.get("count") or 0)


def get_voice_command_counts(
    db,
    *,
    user_id: int,
) -> dict[str, int]:
    """
    Count total and enabled voice commands for a user (excluding system commands).

    Args:
        db: CharactersRAGDB instance
        user_id: User ID

    Returns:
        Dict with total and enabled counts
    """
    result = db.execute_query(
        """
        SELECT
            COUNT(*) AS total_commands,
            SUM(CASE WHEN enabled = 1 THEN 1 ELSE 0 END) AS enabled_commands
        FROM voice_commands
        WHERE user_id = ? AND deleted = 0
        """,
        (user_id,),
    )
    rows = result.fetchall() if hasattr(result, 'fetchall') else list(result)
    if not rows:
        return {"total": 0, "enabled": 0}
    row = rows[0]
    if not isinstance(row, dict):
        row = dict(row)
    return {"total": row.get("total_commands") or 0, "enabled": row.get("enabled_commands") or 0}


def _row_to_voice_command(row: dict[str, Any]) -> VoiceCommand:
    """Convert a database row to a VoiceCommand."""
    if not isinstance(row, dict):
        row = dict(row)
    phrases = row.get("phrases", "[]")
    if isinstance(phrases, str):
        phrases = json.loads(phrases)

    action_config = row.get("action_config", "{}")
    if isinstance(action_config, str):
        action_config = json.loads(action_config)

    created_at = row.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)

    updated_at = row.get("updated_at")
    if isinstance(updated_at, str):
        updated_at = datetime.fromisoformat(updated_at)

    return VoiceCommand(
        id=row["id"],
        user_id=row["user_id"],
        persona_id=row.get("persona_id"),
        connection_id=row.get("connection_id"),
        name=row["name"],
        phrases=phrases,
        action_type=ActionType(row["action_type"]),
        action_config=action_config,
        priority=row.get("priority", 0),
        enabled=bool(row.get("enabled", 1)),
        requires_confirmation=bool(row.get("requires_confirmation", 0)),
        description=row.get("description"),
        created_at=created_at,
        updated_at=updated_at,
    )


def _row_to_voice_session(row: dict[str, Any]) -> VoiceSessionContext:
    """Convert a database row to a VoiceSessionContext."""
    from .schemas import VoiceIntent

    if not isinstance(row, dict):
        row = dict(row)
    metadata = row.get("context", "{}")
    if isinstance(metadata, str):
        metadata = json.loads(metadata) if metadata else {}

    conversation_history = row.get("conversation_history", "[]")
    if isinstance(conversation_history, str):
        conversation_history = json.loads(conversation_history) if conversation_history else []

    pending_intent_data = row.get("pending_intent")
    pending_intent = None
    if pending_intent_data:
        if isinstance(pending_intent_data, str):
            pending_intent_data = json.loads(pending_intent_data)
        if pending_intent_data:
            pending_intent = VoiceIntent(**pending_intent_data)

    last_action_result = row.get("last_action_result")
    if isinstance(last_action_result, str):
        last_action_result = json.loads(last_action_result) if last_action_result else None

    created_at = row.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)

    last_activity = row.get("last_activity")
    if isinstance(last_activity, str):
        last_activity = datetime.fromisoformat(last_activity)

    return VoiceSessionContext(
        session_id=row["session_id"],
        user_id=row["user_id"],
        state=VoiceSessionState(row.get("state", "idle")),
        conversation_history=conversation_history,
        pending_intent=pending_intent,
        last_action_result=last_action_result,
        metadata=metadata,
        created_at=created_at or datetime.utcnow(),
        last_activity=last_activity or datetime.utcnow(),
    )


#
# End of VoiceAssistant/db_helpers.py
#######################################################################################################################

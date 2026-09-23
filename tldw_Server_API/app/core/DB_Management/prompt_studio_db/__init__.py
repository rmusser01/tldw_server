"""Backend-neutral Prompt Studio persistence (TASK-13318, ADR-051).

Aggregates move here one at a time out of the two parallel implementations in
PromptStudioDatabase.py. Each repository runs over the legacy database object as its
session -- both implementations expose transaction(), _cursor_exec() with `?`
placeholders, _row_to_dict(), _write_lock and client_id -- as media_db/ does.
"""

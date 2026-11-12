"""Migration-only, snapshot-free storage for permanent Workspace startup keys."""

from __future__ import annotations


def workspace_chat_startup_schema_statements(*, postgres: bool) -> tuple[str, ...]:
    """Return bounded receipt DDL without creating tables outside migrations."""
    invalid_hex = "{column} !~ '[^0-9a-f]'" if postgres else "{column} NOT GLOB '*[^0-9a-f]*'"
    # SQLite text functions stop at NUL, so both character and stored-byte lengths matter.
    length_check = (
        "length({column}) = 64"
        if postgres
        else "typeof({column}) = 'text' AND length({column}) = 64 AND length(CAST({column} AS BLOB)) = 64"
    )
    checks = ",\n".join(
        f"CHECK({length_check.format(column=column)} AND {invalid_hex.format(column=column)})"
        for column in ("key_digest", "request_fingerprint", "binding_digest")
    )
    return (
        f"""
        CREATE TABLE workspace_chat_startup_receipts (
            owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
            key_digest TEXT NOT NULL,
            request_fingerprint TEXT NOT NULL,
            binding_digest TEXT NOT NULL,
            workspace_id TEXT NOT NULL CHECK(length(trim(workspace_id)) > 0),
            conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            invalidated_at TEXT,
            PRIMARY KEY (owner_user_id, key_digest),
            {checks}
        )
        """,
        "CREATE INDEX workspace_chat_startup_conversation "
        "ON workspace_chat_startup_receipts(conversation_id, owner_user_id)",
    )

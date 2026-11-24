"""Schema for thin Research Studio output receipts in the ChaCha database."""

RESEARCH_STUDY_SCHEMA = (
    """
    CREATE TABLE research_study_outputs (
        owner_user_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
        operation_id TEXT NOT NULL CHECK(length(operation_id) BETWEEN 1 AND 256),
        request_fingerprint TEXT NOT NULL CHECK(length(request_fingerprint) = 64),
        output_kind TEXT NOT NULL CHECK(output_kind IN ('flashcards', 'quiz')),
        owner_id BIGINT NOT NULL CHECK(owner_id > 0),
        generation_json TEXT NOT NULL CHECK(length(generation_json) <= 32768),
        created_at TEXT NOT NULL,
        PRIMARY KEY (owner_user_id, workspace_id, operation_id),
        UNIQUE (owner_user_id, workspace_id, output_kind, owner_id)
    )
    """,
    "CREATE INDEX research_study_output_page "
    "ON research_study_outputs(owner_user_id, workspace_id, created_at DESC, operation_id)",
)

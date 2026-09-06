-- version: 27
-- description: Original-output history incarnation and monotonic disposal receiver
-- idempotent: true
BEGIN TRANSACTION;

-- TTS history was formerly an optional bootstrap ensure, not a numbered migration.
CREATE TABLE IF NOT EXISTS tts_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    text TEXT,
    text_hash TEXT NOT NULL,
    text_length INTEGER,
    provider TEXT,
    model TEXT,
    voice_id TEXT,
    voice_name TEXT,
    voice_info TEXT,
    format TEXT,
    duration_ms INTEGER,
    generation_time_ms INTEGER,
    params_json TEXT,
    status TEXT,
    segments_json TEXT,
    favorite BOOLEAN NOT NULL DEFAULT 0,
    job_id INTEGER,
    output_id INTEGER,
    artifact_ids TEXT,
    artifact_deleted_at TEXT,
    error_message TEXT,
    deleted BOOLEAN NOT NULL DEFAULT 0,
    deleted_at TEXT
);
ALTER TABLE tts_history ADD COLUMN output_incarnation TEXT;

CREATE TABLE IF NOT EXISTS tts_output_instances (
    user_id TEXT NOT NULL CHECK (length(trim(user_id)) > 0),
    output_incarnation TEXT NOT NULL CHECK (length(output_incarnation) = 32),
    state TEXT NOT NULL CHECK (state IN ('live', 'disposed')),
    disposal_token TEXT,
    disposed_at TEXT,
    PRIMARY KEY (user_id, output_incarnation),
    CHECK (
        (state = 'live' AND disposal_token IS NULL AND disposed_at IS NULL)
        OR (state = 'disposed' AND disposal_token IS NOT NULL
            AND length(disposal_token) = 32 AND disposed_at IS NOT NULL)
    )
);
CREATE INDEX IF NOT EXISTS idx_tts_history_incarnation ON tts_history(user_id, output_incarnation);
CREATE INDEX IF NOT EXISTS idx_tts_history_user_created ON tts_history(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tts_history_user_favorite ON tts_history(user_id, favorite);
CREATE INDEX IF NOT EXISTS idx_tts_history_user_provider ON tts_history(user_id, provider);
CREATE INDEX IF NOT EXISTS idx_tts_history_user_model ON tts_history(user_id, model);
CREATE INDEX IF NOT EXISTS idx_tts_history_user_voice_id ON tts_history(user_id, voice_id);
CREATE INDEX IF NOT EXISTS idx_tts_history_user_text_hash ON tts_history(user_id, text_hash);

UPDATE schema_version SET version = 27;
COMMIT;

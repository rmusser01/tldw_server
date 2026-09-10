-- version: 23
-- description: Add transcript run-history columns and backfill existing transcript/media state

PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

ALTER TABLE Media ADD COLUMN latest_transcription_run_id INTEGER;
ALTER TABLE Media ADD COLUMN next_transcription_run_id INTEGER NOT NULL DEFAULT 1;

ALTER TABLE Transcripts RENAME TO Transcripts_v22_legacy;

CREATE TABLE Transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    whisper_model TEXT,
    transcription TEXT,
    created_at DATETIME,
    transcription_run_id INTEGER,
    supersedes_run_id INTEGER,
    idempotency_key TEXT,
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0,
    prev_version INTEGER,
    merge_parent_uuid TEXT,
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
);

INSERT INTO Transcripts (
    id,
    media_id,
    whisper_model,
    transcription,
    created_at,
    transcription_run_id,
    supersedes_run_id,
    idempotency_key,
    uuid,
    last_modified,
    version,
    client_id,
    deleted,
    prev_version,
    merge_parent_uuid
)
SELECT
    legacy.id,
    legacy.media_id,
    legacy.whisper_model,
    legacy.transcription,
    legacy.created_at,
    ROW_NUMBER() OVER (
        PARTITION BY legacy.media_id
        ORDER BY legacy.created_at ASC, legacy.id ASC
    ) AS transcription_run_id,
    NULL AS supersedes_run_id,
    NULL AS idempotency_key,
    legacy.uuid,
    legacy.last_modified,
    legacy.version,
    legacy.client_id,
    legacy.deleted,
    legacy.prev_version,
    legacy.merge_parent_uuid
FROM Transcripts_v22_legacy AS legacy
ORDER BY legacy.id;

DROP TABLE Transcripts_v22_legacy;

CREATE INDEX IF NOT EXISTS idx_transcripts_media_id ON Transcripts(media_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_uuid ON Transcripts(uuid);
CREATE INDEX IF NOT EXISTS idx_transcripts_last_modified ON Transcripts(last_modified);
CREATE INDEX IF NOT EXISTS idx_transcripts_deleted ON Transcripts(deleted);
CREATE INDEX IF NOT EXISTS idx_transcripts_prev_version ON Transcripts(prev_version);
CREATE INDEX IF NOT EXISTS idx_transcripts_merge_parent_uuid ON Transcripts(merge_parent_uuid);
CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_media_run_id
    ON Transcripts(media_id, transcription_run_id DESC)
    WHERE transcription_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_transcripts_supersedes_run_id ON Transcripts(supersedes_run_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_media_idempotency_key
    ON Transcripts(media_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

DROP TRIGGER IF EXISTS media_validate_sync_update;

UPDATE Media
SET latest_transcription_run_id = (
        SELECT t.transcription_run_id
        FROM Transcripts AS t
        WHERE t.media_id = Media.id
          AND t.deleted = 0
        ORDER BY t.transcription_run_id DESC
        LIMIT 1
    ),
    next_transcription_run_id = COALESCE((
        SELECT MAX(t.transcription_run_id) + 1
        FROM Transcripts AS t
        WHERE t.media_id = Media.id
    ), 1);

CREATE TRIGGER media_validate_sync_update BEFORE UPDATE ON Media
BEGIN
    SELECT RAISE(ABORT, 'Sync Error (Media): Version must increment by exactly 1.')
    WHERE NEW.version IS NOT OLD.version + 1;
    SELECT RAISE(ABORT, 'Sync Error (Media): Client ID cannot be NULL or empty.')
    WHERE NEW.client_id IS NULL OR NEW.client_id = '';
    SELECT RAISE(ABORT, 'Sync Error (Media): UUID cannot be changed.')
    WHERE NEW.uuid IS NOT OLD.uuid;
END;

CREATE INDEX IF NOT EXISTS idx_media_latest_transcription_run_id ON Media(latest_transcription_run_id);
CREATE INDEX IF NOT EXISTS idx_media_next_transcription_run_id ON Media(next_transcription_run_id);

DROP TRIGGER IF EXISTS transcripts_validate_sync_update;
CREATE TRIGGER transcripts_validate_sync_update BEFORE UPDATE ON Transcripts
BEGIN
    SELECT RAISE(ABORT, 'Sync Error (Transcripts): Version must increment by exactly 1.')
    WHERE NEW.version IS NOT OLD.version + 1;
    SELECT RAISE(ABORT, 'Sync Error (Transcripts): Client ID cannot be NULL or empty.')
    WHERE NEW.client_id IS NULL OR NEW.client_id = '';
    SELECT RAISE(ABORT, 'Sync Error (Transcripts): UUID cannot be changed.')
    WHERE NEW.uuid IS NOT OLD.uuid;
END;

COMMIT;
PRAGMA foreign_keys = ON;

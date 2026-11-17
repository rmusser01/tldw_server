-- version: 27
-- description: Owner-scoped cluster publication fences and lifetime replay receipts
-- idempotent: true
BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS claims_cluster_publication_heads (
    owner_user_id TEXT PRIMARY KEY NOT NULL CHECK (length(owner_user_id) BETWEEN 1 AND 255),
    authority_uuid TEXT,
    generation BIGINT NOT NULL DEFAULT 0 CHECK (generation >= 0),
    job_uuid TEXT,
    CHECK ((generation = 0 AND authority_uuid IS NULL AND job_uuid IS NULL)
        OR (generation > 0 AND authority_uuid IS NOT NULL AND length(authority_uuid) = 36
            AND job_uuid IS NOT NULL AND length(job_uuid) = 36))
);
CREATE TABLE IF NOT EXISTS claims_cluster_publication_receipts (
    owner_user_id TEXT NOT NULL CHECK (length(owner_user_id) BETWEEN 1 AND 255),
    job_uuid TEXT NOT NULL CHECK (length(job_uuid) = 36),
    authority_uuid TEXT NOT NULL CHECK (length(authority_uuid) = 36),
    generation BIGINT NOT NULL CHECK (generation > 0),
    settings_fingerprint TEXT NOT NULL CHECK (length(settings_fingerprint) = 64),
    method TEXT NOT NULL CHECK (method IN ('exact', 'embeddings')),
    similarity_threshold REAL,
    clusters_created INTEGER NOT NULL CHECK (clusters_created BETWEEN 0 AND 50000),
    claims_assigned INTEGER NOT NULL CHECK (claims_assigned BETWEEN 0 AND 50000),
    claims_skipped INTEGER NOT NULL CHECK (claims_skipped BETWEEN 0 AND 50000),
    outcome TEXT NOT NULL CHECK (outcome IN ('ok', 'skipped')),
    reason TEXT CHECK (reason IS NULL OR length(reason) BETWEEN 1 AND 128),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (owner_user_id, job_uuid),
    UNIQUE (owner_user_id, authority_uuid, generation),
    CHECK ((method = 'exact' AND similarity_threshold IS NULL)
        OR (method = 'embeddings' AND similarity_threshold IS NOT NULL AND similarity_threshold BETWEEN 0 AND 1)),
    CHECK ((outcome = 'ok' AND reason IS NULL) OR (outcome = 'skipped' AND reason IS NOT NULL))
);
UPDATE schema_version SET version = 27;
COMMIT;

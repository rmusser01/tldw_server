-- version: 24
-- description: Add Jobs linkage and snapshot metadata to Claims analytics exports
-- idempotent: true

PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

ALTER TABLE claims_analytics_exports ADD COLUMN job_id INTEGER;
ALTER TABLE claims_analytics_exports ADD COLUMN error_code TEXT;
ALTER TABLE claims_analytics_exports ADD COLUMN snapshot_at TEXT;
ALTER TABLE claims_analytics_exports ADD COLUMN snapshot_event_id INTEGER;

CREATE INDEX IF NOT EXISTS idx_claims_analytics_exports_job_id
    ON claims_analytics_exports(job_id);
CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_user_created_id
    ON claims_monitoring_events(user_id, created_at, id);

UPDATE schema_version SET version = 24;

COMMIT;
PRAGMA foreign_keys = ON;

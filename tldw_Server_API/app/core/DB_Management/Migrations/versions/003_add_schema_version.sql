--This file adds the table to track schema version and initializes it.
-- migrations/versions/003_add_schema_version.sql
    -- Schema Version Table --
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY NOT NULL
    );

    -- Initialize version to 0, indicating the base schema before migrations ran
    INSERT INTO schema_version (version) VALUES (0);
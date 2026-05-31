CREATE TABLE IF NOT EXISTS schema_versions (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

INSERT OR IGNORE INTO schema_versions(version) VALUES (1);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    language TEXT NOT NULL,
    size INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    modified_at REAL NOT NULL,
    indexed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    node_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'indexed',
    errors TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    identity_key TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT,
    file_path TEXT NOT NULL,
    language TEXT,
    start_line INTEGER,
    end_line INTEGER,
    start_column INTEGER,
    end_column INTEGER,
    signature TEXT,
    docstring TEXT,
    visibility TEXT,
    flags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_nodes_file_path ON nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target TEXT REFERENCES nodes(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line INTEGER,
    column INTEGER,
    metadata TEXT NOT NULL DEFAULT '{}',
    provenance TEXT
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
CREATE INDEX IF NOT EXISTS idx_edges_file_path ON edges(file_path);

CREATE TABLE IF NOT EXISTS unresolved_refs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    reference_name TEXT NOT NULL,
    reference_kind TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line INTEGER,
    column INTEGER,
    candidates TEXT NOT NULL DEFAULT '[]',
    language TEXT,
    resolved_target TEXT,
    resolved_edge TEXT,
    resolution_kind TEXT,
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_unresolved_refs_file_path ON unresolved_refs(file_path);
CREATE INDEX IF NOT EXISTS idx_unresolved_refs_from_node ON unresolved_refs(from_node_id);

CREATE TABLE IF NOT EXISTS index_runs (
    run_id TEXT PRIMARY KEY,
    workspace_key TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    counters TEXT NOT NULL DEFAULT '{}',
    error_summary TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_index_runs_workspace_started ON index_runs(workspace_key, started_at);

CREATE TABLE IF NOT EXISTS project_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

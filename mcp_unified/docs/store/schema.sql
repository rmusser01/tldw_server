PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS docs_schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO docs_schema_migrations (version) VALUES (1);

CREATE TABLE IF NOT EXISTS docs_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_scope TEXT NOT NULL DEFAULT '',
    profile_scope TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL,
    document_type TEXT NOT NULL,
    canonical_uri TEXT NOT NULL,
    source_path TEXT,
    source_url TEXT,
    content_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    package_name TEXT,
    package_version TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (owner_scope, profile_scope, canonical_uri)
);

CREATE INDEX IF NOT EXISTS docs_documents_scope_idx
    ON docs_documents (owner_scope, profile_scope, id);

CREATE INDEX IF NOT EXISTS docs_documents_scope_type_idx
    ON docs_documents (owner_scope, profile_scope, document_type);

CREATE INDEX IF NOT EXISTS docs_documents_scope_package_idx
    ON docs_documents (owner_scope, profile_scope, package_name, package_version);

CREATE TABLE IF NOT EXISTS docs_collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_scope TEXT NOT NULL DEFAULT '',
    profile_scope TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (owner_scope, profile_scope, name)
);

CREATE INDEX IF NOT EXISTS docs_collections_scope_idx
    ON docs_collections (owner_scope, profile_scope, name);

CREATE TABLE IF NOT EXISTS docs_collection_members (
    collection_id INTEGER NOT NULL REFERENCES docs_collections(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES docs_documents(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (collection_id, document_id)
);

CREATE INDEX IF NOT EXISTS docs_collection_members_document_idx
    ON docs_collection_members (document_id);

CREATE TABLE IF NOT EXISTS docs_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_scope TEXT NOT NULL DEFAULT '',
    profile_scope TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (owner_scope, profile_scope, name)
);

CREATE INDEX IF NOT EXISTS docs_keywords_scope_idx
    ON docs_keywords (owner_scope, profile_scope, name);

CREATE TABLE IF NOT EXISTS docs_document_keywords (
    keyword_id INTEGER NOT NULL REFERENCES docs_keywords(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES docs_documents(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (keyword_id, document_id)
);

CREATE INDEX IF NOT EXISTS docs_document_keywords_document_idx
    ON docs_document_keywords (document_id);

CREATE TABLE IF NOT EXISTS docs_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES docs_documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    heading TEXT,
    level INTEGER,
    start_char INTEGER,
    end_char INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS docs_sections_document_idx
    ON docs_sections (document_id, ordinal);

CREATE TABLE IF NOT EXISTS docs_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES docs_documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    citation TEXT NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS docs_chunks_document_idx
    ON docs_chunks (document_id, ordinal);

CREATE VIRTUAL TABLE IF NOT EXISTS docs_chunks_fts USING fts5(
    title,
    body,
    citation,
    chunk_id UNINDEXED,
    document_id UNINDEXED
);

CREATE TABLE IF NOT EXISTS docs_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_scope TEXT NOT NULL DEFAULT '',
    profile_scope TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL,
    document_id INTEGER NOT NULL REFERENCES docs_documents(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (owner_scope, profile_scope, name)
);

CREATE INDEX IF NOT EXISTS docs_aliases_scope_idx
    ON docs_aliases (owner_scope, profile_scope, name);

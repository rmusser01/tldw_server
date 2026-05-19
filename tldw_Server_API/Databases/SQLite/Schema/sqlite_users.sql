-- SQLite Users Database Schema
-- This schema defines the core authentication and user management tables

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    is_verified BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME,
    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until DATETIME,
    role TEXT NOT NULL DEFAULT 'user',
    metadata TEXT -- JSON field for additional user metadata
);

-- Sessions table for session management
CREATE TABLE IF NOT EXISTS user_sessions (
    id TEXT PRIMARY KEY, -- Session token/ID
    user_id INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    user_agent TEXT,
    ip_address TEXT,
    last_activity DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,
    key_prefix TEXT,
    name TEXT,
    description TEXT,
    scope TEXT,
    status TEXT DEFAULT 'active',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,
    last_used_at DATETIME,
    last_used_ip TEXT,
    usage_count INTEGER DEFAULT 0,
    rate_limit INTEGER,
    allowed_ips TEXT,
    metadata TEXT,
    rotated_from INTEGER REFERENCES api_keys(id),
    rotated_to INTEGER REFERENCES api_keys(id),
    revoked_at DATETIME,
    revoked_by INTEGER,
    revoke_reason TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Per-user provider secrets (BYOK)
CREATE TABLE IF NOT EXISTS user_provider_secrets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    encrypted_blob TEXT NOT NULL,
    key_hint TEXT,
    metadata TEXT,
    created_by INTEGER,
    updated_by INTEGER,
    revoked_by INTEGER,
    revoked_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_used_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, provider)
);

-- Org/team shared provider secrets (BYOK)
CREATE TABLE IF NOT EXISTS org_provider_secrets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_type TEXT NOT NULL CHECK (scope_type IN ('org', 'team')),
    scope_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    encrypted_blob TEXT NOT NULL,
    key_hint TEXT,
    metadata TEXT,
    created_by INTEGER,
    updated_by INTEGER,
    revoked_by INTEGER,
    revoked_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_used_at DATETIME,
    UNIQUE (scope_type, scope_id, provider)
);

-- Organizations table for multi-tenant support
CREATE TABLE IF NOT EXISTS organizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE,
    name TEXT UNIQUE NOT NULL,
    slug TEXT UNIQUE,
    owner_user_id INTEGER,
    is_active INTEGER DEFAULT 1,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (owner_user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Organization membership table
CREATE TABLE IF NOT EXISTS org_members (
    org_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    role TEXT DEFAULT 'member',
    status TEXT DEFAULT 'active',
    added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (org_id, user_id),
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    slug TEXT,
    description TEXT,
    is_active INTEGER DEFAULT 1,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (org_id, name),
    FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
);

-- Team membership table
CREATE TABLE IF NOT EXISTS team_members (
    team_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    role TEXT DEFAULT 'member',
    status TEXT DEFAULT 'active',
    added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (team_id, user_id),
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Team invites table
CREATE TABLE IF NOT EXISTS team_invites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    email TEXT NOT NULL,
    token TEXT UNIQUE NOT NULL,
    status TEXT DEFAULT 'pending',
    invited_by INTEGER,
    invited_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    accepted_at DATETIME,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
    FOREIGN KEY (invited_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Password reset tokens
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token TEXT NOT NULL UNIQUE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL,
    used_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Email verification tokens
CREATE TABLE IF NOT EXISTS email_verification_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token TEXT NOT NULL UNIQUE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL,
    used_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_uuid ON users(uuid);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);
CREATE INDEX IF NOT EXISTS idx_orgs_owner ON organizations(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_org_members_user ON org_members(user_id);
CREATE INDEX IF NOT EXISTS idx_teams_org ON teams(org_id);
CREATE INDEX IF NOT EXISTS idx_team_members_user ON team_members(user_id);
CREATE INDEX IF NOT EXISTS idx_team_invites_team ON team_invites(team_id);
CREATE INDEX IF NOT EXISTS idx_team_invites_status ON team_invites(status);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status);
CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at);
CREATE INDEX IF NOT EXISTS idx_user_provider_secrets_user_id ON user_provider_secrets(user_id);
CREATE INDEX IF NOT EXISTS idx_user_provider_secrets_provider ON user_provider_secrets(provider);
CREATE INDEX IF NOT EXISTS idx_org_provider_secrets_scope ON org_provider_secrets(scope_type, scope_id);
CREATE INDEX IF NOT EXISTS idx_org_provider_secrets_provider ON org_provider_secrets(provider);
CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_token ON password_reset_tokens(token);
CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_expires_at ON password_reset_tokens(expires_at);
CREATE INDEX IF NOT EXISTS idx_email_verification_tokens_token ON email_verification_tokens(token);
CREATE INDEX IF NOT EXISTS idx_email_verification_tokens_expires_at ON email_verification_tokens(expires_at);

-- Update trigger for users table
CREATE TRIGGER IF NOT EXISTS update_users_timestamp
    AFTER UPDATE ON users
    FOR EACH ROW
BEGIN
    UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Update trigger for user_provider_secrets table
CREATE TRIGGER IF NOT EXISTS update_user_provider_secrets_timestamp
    AFTER UPDATE ON user_provider_secrets
    FOR EACH ROW
BEGIN
    UPDATE user_provider_secrets SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Update trigger for org_provider_secrets table
CREATE TRIGGER IF NOT EXISTS update_org_provider_secrets_timestamp
    AFTER UPDATE ON org_provider_secrets
    FOR EACH ROW
BEGIN
    UPDATE org_provider_secrets SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Cleanup org_provider_secrets when organizations are deleted
CREATE TRIGGER IF NOT EXISTS delete_org_provider_secrets_on_org_delete
    AFTER DELETE ON organizations
    FOR EACH ROW
BEGIN
    DELETE FROM org_provider_secrets WHERE scope_type = 'org' AND scope_id = OLD.id;
END;

-- Cleanup org_provider_secrets when teams are deleted
CREATE TRIGGER IF NOT EXISTS delete_org_provider_secrets_on_team_delete
    AFTER DELETE ON teams
    FOR EACH ROW
BEGIN
    DELETE FROM org_provider_secrets WHERE scope_type = 'team' AND scope_id = OLD.id;
END;

-- Update trigger for session activity
CREATE TRIGGER IF NOT EXISTS update_session_activity
    AFTER UPDATE ON user_sessions
    FOR EACH ROW
BEGIN
    UPDATE user_sessions SET last_activity = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

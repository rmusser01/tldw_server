-- PostgreSQL AuthNZ bootstrap schema (users core table)
-- This file is consumed by app/core/AuthNZ/database.py during PG startup.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS public.users (
    id SERIAL PRIMARY KEY,
    uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
    is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TIMESTAMPTZ,
    storage_quota_mb INTEGER NOT NULL DEFAULT 5120,
    storage_used_mb DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    profile_version TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMPTZ,
    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    email_verified_at TIMESTAMPTZ,
    two_factor_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    two_factor_secret TEXT,
    totp_secret TEXT,
    backup_codes TEXT,
    created_by INTEGER REFERENCES public.users(id) ON DELETE SET NULL,
    password_changed_at TIMESTAMPTZ
);

-- Organizations/teams bootstrap so pg_migrations_extra can safely add
-- org/team-referencing tables during first startup.
CREATE TABLE IF NOT EXISTS public.organizations (
    id SERIAL PRIMARY KEY,
    uuid VARCHAR(64) UNIQUE,
    name VARCHAR(255) UNIQUE NOT NULL,
    slug VARCHAR(255) UNIQUE,
    owner_user_id INTEGER REFERENCES public.users(id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS public.teams (
    id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255),
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (org_id, name)
);

CREATE TABLE IF NOT EXISTS public.org_members (
    org_id INTEGER NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    role VARCHAR(32) DEFAULT 'member',
    status VARCHAR(32) DEFAULT 'active',
    added_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (org_id, user_id)
);

CREATE TABLE IF NOT EXISTS public.team_members (
    team_id INTEGER NOT NULL REFERENCES public.teams(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    role VARCHAR(32) DEFAULT 'member',
    status VARCHAR(32) DEFAULT 'active',
    added_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (team_id, user_id)
);

CREATE TABLE IF NOT EXISTS public.user_config_overrides (
    user_id INTEGER NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value_json TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER,
    updated_by INTEGER,
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS public.org_config_overrides (
    org_id INTEGER NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value_json TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER,
    updated_by INTEGER,
    PRIMARY KEY (org_id, key)
);

CREATE TABLE IF NOT EXISTS public.team_config_overrides (
    team_id INTEGER NOT NULL REFERENCES public.teams(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value_json TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER,
    updated_by INTEGER,
    PRIMARY KEY (team_id, key)
);

CREATE TABLE IF NOT EXISTS audio_studio_media_tickets (
    id BIGSERIAL PRIMARY KEY,
    token_hash TEXT UNIQUE NOT NULL,
    user_id BIGINT NOT NULL,
    project_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    purpose TEXT NOT NULL CHECK (purpose IN ('playback', 'download')),
    expires_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by_auth_mode TEXT,
    last_redeemed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_users_username ON public.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);
CREATE INDEX IF NOT EXISTS idx_users_uuid ON public.users(uuid);
CREATE INDEX IF NOT EXISTS idx_users_role ON public.users(role);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON public.users(is_active);
CREATE INDEX IF NOT EXISTS idx_orgs_owner ON public.organizations(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_teams_org ON public.teams(org_id);
CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_hash ON audio_studio_media_tickets(token_hash);
CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_expiry ON audio_studio_media_tickets(expires_at);
CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_artifact ON audio_studio_media_tickets(user_id, project_id, artifact_id);

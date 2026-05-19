"""
PostgreSQL additive migrations (runtime ensure) for AuthNZ-related extras.

This module centralizes small, idempotent DDL helpers that are safe to call
at startup on Postgres backends. They complement the SQLite migrations in
``migrations.py`` and help keep inline runtime SQL out of business logic.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from .database import DatabasePool, get_db_pool

_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    json.JSONDecodeError,
)

_BUDGET_FIELD_KEYS = {
    "budget_day_usd",
    "budget_month_usd",
    "budget_day_tokens",
    "budget_month_tokens",
}


_CREATE_TOOL_CATALOGS = [
    # tool_catalogs
    (
        """
        CREATE TABLE IF NOT EXISTS tool_catalogs (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NULL,
            org_id INTEGER NULL REFERENCES organizations(id) ON DELETE SET NULL,
            team_id INTEGER NULL REFERENCES teams(id) ON DELETE SET NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_tool_catalogs_scope UNIQUE (name, org_id, team_id)
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_tool_catalogs_org_team ON tool_catalogs(org_id, team_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_tool_catalogs_name ON tool_catalogs(name)", ()),
    # tool_catalog_entries
    (
        """
        CREATE TABLE IF NOT EXISTS tool_catalog_entries (
            id SERIAL PRIMARY KEY,
            catalog_id INTEGER NOT NULL REFERENCES tool_catalogs(id) ON DELETE CASCADE,
            tool_name TEXT NOT NULL,
            module_id TEXT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_tool_catalog_entries UNIQUE (catalog_id, tool_name)
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_tool_catalog_entries_catalog ON tool_catalog_entries(catalog_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_tool_catalog_entries_tool ON tool_catalog_entries(tool_name)", ()),
]


_CREATE_MCP_HUB_TABLES = [
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_acp_profiles (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NULL,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER NULL,
            profile_json TEXT NOT NULL DEFAULT '{}',
            is_active BOOLEAN DEFAULT TRUE,
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_mcp_acp_profiles_scope UNIQUE (name, owner_scope_type, owner_scope_id)
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_acp_profiles_scope "
        "ON mcp_acp_profiles(owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_external_servers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            enabled BOOLEAN DEFAULT TRUE,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER NULL,
            transport TEXT NOT NULL,
            config_json TEXT NOT NULL DEFAULT '{}',
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_external_servers_scope "
        "ON mcp_external_servers(owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        "ALTER TABLE mcp_external_servers "
        "ADD COLUMN IF NOT EXISTS server_source TEXT NOT NULL DEFAULT 'managed'",
        (),
    ),
    (
        "ALTER TABLE mcp_external_servers "
        "ADD COLUMN IF NOT EXISTS legacy_source_ref TEXT",
        (),
    ),
    (
        "ALTER TABLE mcp_external_servers "
        "ADD COLUMN IF NOT EXISTS superseded_by_server_id TEXT",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_external_server_secrets (
            server_id TEXT PRIMARY KEY REFERENCES mcp_external_servers(id) ON DELETE CASCADE,
            encrypted_blob TEXT NOT NULL,
            key_hint TEXT NULL,
            updated_by INTEGER NULL,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_external_server_secrets_updated_at "
        "ON mcp_external_server_secrets(updated_at)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_external_server_credential_slots (
            id SERIAL PRIMARY KEY,
            server_id TEXT NOT NULL REFERENCES mcp_external_servers(id) ON DELETE CASCADE,
            slot_name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            secret_kind TEXT NOT NULL,
            privilege_class TEXT NOT NULL DEFAULT 'default',
            is_required BOOLEAN DEFAULT FALSE,
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_external_server_slots_server_slot "
        "ON mcp_external_server_credential_slots(server_id, slot_name)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_external_server_slots_server "
        "ON mcp_external_server_credential_slots(server_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_external_server_slot_secrets (
            slot_id INTEGER PRIMARY KEY REFERENCES mcp_external_server_credential_slots(id) ON DELETE CASCADE,
            encrypted_blob TEXT NOT NULL,
            key_hint TEXT NULL,
            updated_by INTEGER NULL,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_external_server_slot_secrets_updated_at "
        "ON mcp_external_server_slot_secrets(updated_at)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_permission_profiles (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NULL,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER NULL,
            mode TEXT NOT NULL DEFAULT 'custom',
            policy_document_json TEXT NOT NULL DEFAULT '{}',
            is_active BOOLEAN DEFAULT TRUE,
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_mcp_permission_profiles_scope UNIQUE (name, owner_scope_type, owner_scope_id)
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_permission_profiles_scope "
        "ON mcp_permission_profiles(owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        "ALTER TABLE mcp_permission_profiles "
        "ADD COLUMN IF NOT EXISTS path_scope_object_id INTEGER",
        (),
    ),
    (
        "ALTER TABLE mcp_permission_profiles "
        "ADD COLUMN IF NOT EXISTS is_immutable BOOLEAN NOT NULL DEFAULT FALSE",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_permission_profiles_path_scope_object "
        "ON mcp_permission_profiles(path_scope_object_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_path_scope_objects (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NULL,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER NULL,
            path_scope_document_json TEXT NOT NULL DEFAULT '{}',
            is_active BOOLEAN DEFAULT TRUE,
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_mcp_path_scope_objects_scope UNIQUE (name, owner_scope_type, owner_scope_id)
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_path_scope_objects_scope "
        "ON mcp_path_scope_objects(owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_policy_assignments (
            id SERIAL PRIMARY KEY,
            target_type TEXT NOT NULL,
            target_id TEXT NULL,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER NULL,
            profile_id INTEGER NULL REFERENCES mcp_permission_profiles(id) ON DELETE SET NULL,
            inline_policy_document_json TEXT NOT NULL DEFAULT '{}',
            approval_policy_id INTEGER NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_policy_assignments_scope "
        "ON mcp_policy_assignments(owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_policy_assignments_target "
        "ON mcp_policy_assignments(target_type, target_id)",
        (),
    ),
    (
        "ALTER TABLE mcp_policy_assignments "
        "ADD COLUMN IF NOT EXISTS path_scope_object_id INTEGER",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_policy_assignments_path_scope_object "
        "ON mcp_policy_assignments(path_scope_object_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_policy_assignment_workspaces (
            assignment_id INTEGER NOT NULL REFERENCES mcp_policy_assignments(id) ON DELETE CASCADE,
            workspace_id TEXT NOT NULL,
            created_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_mcp_policy_assignment_workspaces UNIQUE (assignment_id, workspace_id)
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_policy_assignment_workspaces_assignment "
        "ON mcp_policy_assignment_workspaces(assignment_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_workspace_set_objects (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NULL,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_mcp_workspace_set_objects_scope UNIQUE (name, owner_scope_type, owner_scope_id)
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_workspace_set_objects_scope "
        "ON mcp_workspace_set_objects(owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_shared_workspaces (
            id SERIAL PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            display_name TEXT NOT NULL,
            absolute_root TEXT NOT NULL,
            owner_scope_type TEXT NOT NULL DEFAULT 'team',
            owner_scope_id INTEGER NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_mcp_shared_workspaces_scope_workspace UNIQUE (owner_scope_type, owner_scope_id, workspace_id)
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_shared_workspaces_scope "
        "ON mcp_shared_workspaces(owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_workspace_set_object_members (
            workspace_set_object_id INTEGER NOT NULL REFERENCES mcp_workspace_set_objects(id) ON DELETE CASCADE,
            workspace_id TEXT NOT NULL,
            created_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_mcp_workspace_set_object_members UNIQUE (workspace_set_object_id, workspace_id)
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_workspace_set_object_members_object "
        "ON mcp_workspace_set_object_members(workspace_set_object_id)",
        (),
    ),
    (
        "ALTER TABLE mcp_policy_assignments "
        "ADD COLUMN IF NOT EXISTS workspace_source_mode TEXT",
        (),
    ),
    (
        "ALTER TABLE mcp_policy_assignments "
        "ADD COLUMN IF NOT EXISTS workspace_set_object_id INTEGER",
        (),
    ),
    (
        "ALTER TABLE mcp_policy_assignments "
        "ADD COLUMN IF NOT EXISTS is_immutable BOOLEAN NOT NULL DEFAULT FALSE",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_policy_assignments_workspace_set_object "
        "ON mcp_policy_assignments(workspace_set_object_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_policy_overrides (
            id SERIAL PRIMARY KEY,
            assignment_id INTEGER NOT NULL UNIQUE REFERENCES mcp_policy_assignments(id) ON DELETE CASCADE,
            override_document_json TEXT NOT NULL DEFAULT '{}',
            is_active BOOLEAN DEFAULT TRUE,
            broadens_access BOOLEAN DEFAULT FALSE,
            grant_authority_snapshot_json TEXT NOT NULL DEFAULT '{}',
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "ALTER TABLE mcp_policy_overrides "
        "ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
        (),
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_policy_overrides_assignment "
        "ON mcp_policy_overrides(assignment_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_approval_policies (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NULL,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER NULL,
            mode TEXT NOT NULL,
            rules_json TEXT NOT NULL DEFAULT '{}',
            is_active BOOLEAN DEFAULT TRUE,
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_approval_policies_scope "
        "ON mcp_approval_policies(owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        "ALTER TABLE mcp_approval_policies "
        "ADD COLUMN IF NOT EXISTS is_immutable BOOLEAN NOT NULL DEFAULT FALSE",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_capability_adapter_mappings (
            id SERIAL PRIMARY KEY,
            mapping_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NULL,
            owner_scope_type TEXT NOT NULL DEFAULT 'global',
            owner_scope_id INTEGER NULL,
            capability_name TEXT NOT NULL,
            adapter_contract_version INTEGER NOT NULL,
            resolved_policy_document_json TEXT NOT NULL DEFAULT '{}',
            supported_environment_requirements_json TEXT NOT NULL DEFAULT '[]',
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_capability_adapter_mappings_scope "
        "ON mcp_capability_adapter_mappings(owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_capability_adapter_mappings_mapping_id "
        "ON mcp_capability_adapter_mappings(mapping_id)",
        (),
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_capability_adapter_mappings_active_scope_capability "
        "ON mcp_capability_adapter_mappings("
        "owner_scope_type, COALESCE(owner_scope_id, -1), capability_name"
        ") WHERE is_active",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_governance_packs (
            id SERIAL PRIMARY KEY,
            pack_id TEXT NOT NULL,
            pack_version TEXT NOT NULL,
            pack_schema_version INTEGER NOT NULL,
            capability_taxonomy_version INTEGER NOT NULL,
            adapter_contract_version INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NULL,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER NULL,
            bundle_digest TEXT NOT NULL,
            manifest_json TEXT NOT NULL DEFAULT '{}',
            normalized_ir_json TEXT NOT NULL DEFAULT '{}',
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_governance_packs_scope "
        "ON mcp_governance_packs(owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_governance_packs_scope_version "
        "ON mcp_governance_packs(pack_id, pack_version, owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS is_active_install BOOLEAN NOT NULL DEFAULT TRUE",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS superseded_by_governance_pack_id INTEGER NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS installed_from_upgrade_id INTEGER NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS source_type TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS source_location TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS source_ref_requested TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS source_ref_kind TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS source_subpath TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS source_commit_resolved TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS pack_content_digest TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS source_verified BOOLEAN NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS source_verification_mode TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS signer_fingerprint TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS signer_identity TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS verified_object_type TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS verification_result_code TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS verification_warning_code TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS source_fetched_at TIMESTAMPTZ NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_packs "
        "ADD COLUMN IF NOT EXISTS fetched_by INTEGER NULL",
        (),
    ),
    (
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE indexname = 'uq_mcp_governance_packs_active_scope'
                  AND schemaname = ANY (current_schemas(false))
            ) THEN
                UPDATE mcp_governance_packs
                SET is_active_install = FALSE;

                WITH latest AS (
                    SELECT MAX(id) AS id
                    FROM mcp_governance_packs
                    GROUP BY pack_id, owner_scope_type, COALESCE(owner_scope_id, -1)
                )
                UPDATE mcp_governance_packs
                SET is_active_install = TRUE
                WHERE id IN (SELECT id FROM latest);
            END IF;
        END $$;
        """,
        (),
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_governance_packs_active_scope "
        "ON mcp_governance_packs(pack_id, owner_scope_type, COALESCE(owner_scope_id, -1)) "
        "WHERE is_active_install",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_governance_pack_objects (
            id SERIAL PRIMARY KEY,
            governance_pack_id INTEGER NOT NULL REFERENCES mcp_governance_packs(id) ON DELETE CASCADE,
            object_type TEXT NOT NULL,
            object_id TEXT NOT NULL,
            source_object_id TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_governance_pack_objects_pack "
        "ON mcp_governance_pack_objects(governance_pack_id)",
        (),
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_governance_pack_objects_pack_source "
        "ON mcp_governance_pack_objects(governance_pack_id, object_type, source_object_id)",
        (),
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_governance_pack_objects_object "
        "ON mcp_governance_pack_objects(object_type, object_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_governance_pack_upgrades (
            id SERIAL PRIMARY KEY,
            pack_id TEXT NOT NULL,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER NULL,
            from_governance_pack_id INTEGER NOT NULL REFERENCES mcp_governance_packs(id) ON DELETE CASCADE,
            to_governance_pack_id INTEGER NOT NULL REFERENCES mcp_governance_packs(id) ON DELETE CASCADE,
            from_pack_version TEXT NOT NULL,
            to_pack_version TEXT NOT NULL,
            status TEXT NOT NULL,
            planned_by INTEGER NULL,
            executed_by INTEGER NULL,
            planner_inputs_fingerprint TEXT NULL,
            adapter_state_fingerprint TEXT NULL,
            plan_summary_json TEXT NOT NULL DEFAULT '{}',
            accepted_resolutions_json TEXT NOT NULL DEFAULT '{}',
            failure_summary TEXT NULL,
            planned_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            executed_at TIMESTAMPTZ NULL
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_governance_pack_upgrades_scope "
        "ON mcp_governance_pack_upgrades(pack_id, owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_governance_pack_source_candidates (
            id SERIAL PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_location TEXT NOT NULL,
            source_ref_requested TEXT NULL,
            source_ref_kind TEXT NULL,
            source_subpath TEXT NULL,
            source_commit_resolved TEXT NULL,
            pack_content_digest TEXT NOT NULL,
            pack_document_json TEXT NOT NULL DEFAULT '{}',
            source_verified BOOLEAN NULL,
            source_verification_mode TEXT NULL,
            signer_fingerprint TEXT NULL,
            signer_identity TEXT NULL,
            verified_object_type TEXT NULL,
            verification_result_code TEXT NULL,
            verification_warning_code TEXT NULL,
            source_fetched_at TIMESTAMPTZ NULL,
            fetched_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_governance_pack_source_candidates_source "
        "ON mcp_governance_pack_source_candidates(source_type, source_location)",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_pack_source_candidates "
        "ADD COLUMN IF NOT EXISTS source_ref_kind TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_pack_source_candidates "
        "ADD COLUMN IF NOT EXISTS pack_document_json TEXT NOT NULL DEFAULT '{}'",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_pack_source_candidates "
        "ADD COLUMN IF NOT EXISTS signer_fingerprint TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_pack_source_candidates "
        "ADD COLUMN IF NOT EXISTS signer_identity TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_pack_source_candidates "
        "ADD COLUMN IF NOT EXISTS verified_object_type TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_pack_source_candidates "
        "ADD COLUMN IF NOT EXISTS verification_result_code TEXT NULL",
        (),
    ),
    (
        "ALTER TABLE mcp_governance_pack_source_candidates "
        "ADD COLUMN IF NOT EXISTS verification_warning_code TEXT NULL",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_governance_pack_trust_policy (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            policy_document_json TEXT NOT NULL DEFAULT '{}',
            updated_by INTEGER NULL,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_approval_decisions (
            id SERIAL PRIMARY KEY,
            approval_policy_id INTEGER NULL REFERENCES mcp_approval_policies(id) ON DELETE SET NULL,
            context_key TEXT NOT NULL,
            conversation_id TEXT NULL,
            tool_name TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            decision TEXT NOT NULL,
            consume_on_match BOOLEAN DEFAULT FALSE,
            expires_at TIMESTAMPTZ NULL,
            consumed_at TIMESTAMPTZ NULL,
            created_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    ("ALTER TABLE mcp_approval_decisions ADD COLUMN IF NOT EXISTS consume_on_match BOOLEAN DEFAULT FALSE", ()),
    ("ALTER TABLE mcp_approval_decisions ADD COLUMN IF NOT EXISTS consumed_at TIMESTAMPTZ", ()),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_approval_decisions_context "
        "ON mcp_approval_decisions(context_key, conversation_id)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_approval_decisions_active "
        "ON mcp_approval_decisions(context_key, conversation_id, tool_name, scope_key, decision, consumed_at)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_credential_bindings (
            id SERIAL PRIMARY KEY,
            binding_target_type TEXT NOT NULL,
            binding_target_id TEXT NULL,
            external_server_id TEXT NOT NULL REFERENCES mcp_external_servers(id) ON DELETE CASCADE,
            credential_ref TEXT NOT NULL,
            binding_mode TEXT NOT NULL DEFAULT 'grant',
            usage_rules_json TEXT NOT NULL DEFAULT '{}',
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_credential_bindings_target "
        "ON mcp_credential_bindings(binding_target_type, binding_target_id)",
        (),
    ),
    (
        "ALTER TABLE mcp_credential_bindings "
        "ADD COLUMN IF NOT EXISTS binding_mode TEXT NOT NULL DEFAULT 'grant'",
        (),
    ),
    (
        "ALTER TABLE mcp_credential_bindings "
        "ADD COLUMN IF NOT EXISTS slot_name TEXT NOT NULL DEFAULT ''",
        (),
    ),
    (
        """
        UPDATE mcp_credential_bindings
        SET binding_mode = CASE
            WHEN usage_rules_json LIKE '%"binding_mode":"disable"%'
              OR usage_rules_json LIKE '%"binding_mode": "disable"%'
            THEN 'disable'
            ELSE COALESCE(NULLIF(TRIM(binding_mode), ''), 'grant')
        END
        """,
        (),
    ),
    ("DROP INDEX IF EXISTS uq_mcp_credential_bindings_target_server", ()),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_credential_bindings_target_server_slot "
        "ON mcp_credential_bindings(binding_target_type, binding_target_id, external_server_id, slot_name)",
        (),
    ),
    (
        """
        INSERT INTO mcp_external_server_credential_slots (
            server_id, slot_name, display_name, secret_kind, privilege_class, is_required,
            created_by, updated_by, created_at, updated_at
        )
        SELECT
            s.id,
            CASE
                WHEN LOWER(COALESCE(s.config_json::jsonb -> 'auth' ->> 'mode', '')) = 'bearer_token' THEN 'bearer_token'
                WHEN LOWER(COALESCE(s.config_json::jsonb -> 'auth' ->> 'mode', '')) = 'api_key_header' THEN 'api_key'
                ELSE NULL
            END,
            CASE
                WHEN LOWER(COALESCE(s.config_json::jsonb -> 'auth' ->> 'mode', '')) = 'bearer_token' THEN 'Bearer Token'
                WHEN LOWER(COALESCE(s.config_json::jsonb -> 'auth' ->> 'mode', '')) = 'api_key_header' THEN 'Api Key'
                ELSE NULL
            END,
            CASE
                WHEN LOWER(COALESCE(s.config_json::jsonb -> 'auth' ->> 'mode', '')) = 'bearer_token' THEN 'bearer_token'
                WHEN LOWER(COALESCE(s.config_json::jsonb -> 'auth' ->> 'mode', '')) = 'api_key_header' THEN 'api_key'
                ELSE NULL
            END,
            'default',
            TRUE,
            s.created_by,
            s.updated_by,
            s.created_at,
            s.updated_at
        FROM mcp_external_servers s
        WHERE COALESCE(s.server_source, 'managed') = 'managed'
          AND s.superseded_by_server_id IS NULL
          AND LOWER(COALESCE(s.config_json::jsonb -> 'auth' ->> 'mode', '')) IN ('bearer_token', 'api_key_header')
        ON CONFLICT (server_id, slot_name) DO NOTHING
        """,
        (),
    ),
    (
        """
        UPDATE mcp_credential_bindings b
        SET slot_name = CASE
            WHEN LOWER(COALESCE(s.config_json::jsonb -> 'auth' ->> 'mode', '')) = 'bearer_token' THEN 'bearer_token'
            WHEN LOWER(COALESCE(s.config_json::jsonb -> 'auth' ->> 'mode', '')) = 'api_key_header' THEN 'api_key'
            ELSE b.slot_name
        END
        FROM mcp_external_servers s
        WHERE b.external_server_id = s.id
          AND COALESCE(BTRIM(b.slot_name), '') = ''
          AND LOWER(COALESCE(s.config_json::jsonb -> 'auth' ->> 'mode', '')) IN ('bearer_token', 'api_key_header')
        """,
        (),
    ),
    (
        """
        INSERT INTO mcp_external_server_slot_secrets (slot_id, encrypted_blob, key_hint, updated_by, updated_at)
        SELECT slot.id, sec.encrypted_blob, sec.key_hint, sec.updated_by, sec.updated_at
        FROM mcp_external_server_credential_slots slot
        JOIN mcp_external_server_secrets sec ON sec.server_id = slot.server_id
        WHERE slot.slot_name IN ('bearer_token', 'api_key')
        ON CONFLICT (slot_id) DO NOTHING
        """,
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS mcp_policy_audit_history (
            id SERIAL PRIMARY KEY,
            resource_type TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            action TEXT NOT NULL,
            previous_value_json TEXT NOT NULL DEFAULT '{}',
            new_value_json TEXT NOT NULL DEFAULT '{}',
            broadened_access BOOLEAN DEFAULT FALSE,
            actor_id INTEGER NULL,
            grant_authority_snapshot_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_mcp_policy_audit_history_resource "
        "ON mcp_policy_audit_history(resource_type, resource_id)",
        (),
    ),
]


_CREATE_PRIVILEGE_SNAPSHOTS = [
    (
        """
        CREATE TABLE IF NOT EXISTS privilege_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            generated_at TIMESTAMPTZ NOT NULL,
            generated_by TEXT NOT NULL,
            org_id TEXT NULL,
            team_id TEXT NULL,
            catalog_version TEXT NOT NULL,
            summary_json TEXT NULL,
            scope_index TEXT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_priv_snapshots_generated_at ON privilege_snapshots(generated_at)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_priv_snapshots_org ON privilege_snapshots(org_id)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_priv_snapshots_team ON privilege_snapshots(team_id)",
        (),
    ),
]


_CREATE_AUTHNZ_CORE_TABLES = [
    # audit_logs
    (
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            action VARCHAR(255) NOT NULL,
            resource_type VARCHAR(128),
            resource_id INTEGER,
            ip_address VARCHAR(45),
            user_agent TEXT,
            status VARCHAR(32),
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at)", ()),
    # retention policy overrides
    (
        """
        CREATE TABLE IF NOT EXISTS retention_policy_overrides (
            policy_key TEXT PRIMARY KEY,
            days INTEGER NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    # user profile config overrides
    (
        """
        CREATE TABLE IF NOT EXISTS user_config_overrides (
            user_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER,
            updated_by INTEGER,
            PRIMARY KEY (user_id, key),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_user_config_overrides_user_id ON user_config_overrides(user_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_user_config_overrides_key ON user_config_overrides(key)", ()),
    (
        """
        CREATE TABLE IF NOT EXISTS org_config_overrides (
            org_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER,
            updated_by INTEGER,
            PRIMARY KEY (org_id, key),
            FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_org_config_overrides_org_id ON org_config_overrides(org_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_org_config_overrides_key ON org_config_overrides(key)", ()),
    (
        """
        CREATE TABLE IF NOT EXISTS team_config_overrides (
            team_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER,
            updated_by INTEGER,
            PRIMARY KEY (team_id, key),
            FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_team_config_overrides_team_id ON team_config_overrides(team_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_team_config_overrides_key ON team_config_overrides(key)", ()),
    # sessions (core columns + additive columns/indexes)
    (
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            token_hash VARCHAR(64) NOT NULL,
            refresh_token_hash VARCHAR(64),
            encrypted_token TEXT,
            encrypted_refresh TEXT,
            expires_at TIMESTAMP NOT NULL,
            refresh_expires_at TIMESTAMP,
            ip_address VARCHAR(45),
            user_agent TEXT,
            device_id TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            is_revoked BOOLEAN DEFAULT FALSE,
            revoked_at TIMESTAMP,
            revoked_by INTEGER,
            revoke_reason TEXT,
            access_jti VARCHAR(128),
            refresh_jti VARCHAR(128),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """,
        (),
    ),
    ("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS refresh_expires_at TIMESTAMP", ()),
    ("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS is_revoked BOOLEAN DEFAULT FALSE", ()),
    ("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS access_jti VARCHAR(128)", ()),
    ("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS refresh_jti VARCHAR(128)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_sessions_access_jti ON sessions(access_jti)", ()),
    # registration_codes
    (
        """
        CREATE TABLE IF NOT EXISTS registration_codes (
            id SERIAL PRIMARY KEY,
            code VARCHAR(128) UNIQUE NOT NULL,
            role_to_grant VARCHAR(50) DEFAULT 'user',
            max_uses INTEGER DEFAULT 1,
            uses INTEGER DEFAULT 0,
            expires_at TIMESTAMP,
            created_by INTEGER,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    ("ALTER TABLE registration_codes ADD COLUMN IF NOT EXISTS times_used INTEGER DEFAULT 0", ()),
    ("ALTER TABLE registration_codes ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE", ()),
    ("ALTER TABLE registration_codes ADD COLUMN IF NOT EXISTS description TEXT", ()),
    ("ALTER TABLE registration_codes ADD COLUMN IF NOT EXISTS allowed_email_domain TEXT", ()),
    ("ALTER TABLE IF EXISTS org_invites ADD COLUMN IF NOT EXISTS allowed_email_domain TEXT", ()),
    (
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'registration_codes' AND column_name = 'uses'
            ) THEN
                UPDATE registration_codes
                SET times_used = uses
                WHERE times_used IS NULL OR times_used = 0;
            END IF;
        END $$;
        """,
        (),
    ),
    # RBAC core tables
    (
        """
        CREATE TABLE IF NOT EXISTS roles (
            id SERIAL PRIMARY KEY,
            name VARCHAR(64) UNIQUE NOT NULL,
            description TEXT,
            is_system BOOLEAN DEFAULT FALSE
        )
        """,
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS permissions (
            id SERIAL PRIMARY KEY,
            name VARCHAR(128) UNIQUE NOT NULL,
            description TEXT,
            category VARCHAR(64)
        )
        """,
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS role_permissions (
            role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
            permission_id INTEGER NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
            PRIMARY KEY (role_id, permission_id)
        )
        """,
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS user_roles (
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
            granted_by INTEGER,
            expires_at TIMESTAMP,
            PRIMARY KEY (user_id, role_id)
        )
        """,
        (),
    ),
    # Backstop for older/minimal schemas (tests) that created user_roles without these columns.
    ("ALTER TABLE user_roles ADD COLUMN IF NOT EXISTS granted_by INTEGER", ()),
    ("ALTER TABLE user_roles ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP", ()),
    (
        """
        CREATE TABLE IF NOT EXISTS user_permissions (
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            permission_id INTEGER NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
            granted BOOLEAN NOT NULL DEFAULT TRUE,
            expires_at TIMESTAMP,
            PRIMARY KEY (user_id, permission_id)
        )
        """,
        (),
    ),
    # RBAC rate limits
    (
        """
        CREATE TABLE IF NOT EXISTS rbac_role_rate_limits (
            role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
            resource VARCHAR(128) NOT NULL,
            limit_per_min INTEGER,
            burst INTEGER,
            PRIMARY KEY (role_id, resource)
        )
        """,
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS rbac_user_rate_limits (
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            resource VARCHAR(128) NOT NULL,
            limit_per_min INTEGER,
            burst INTEGER,
            PRIMARY KEY (user_id, resource)
        )
        """,
        (),
    ),
    # Organizations and teams hierarchy
    (
        """
        CREATE TABLE IF NOT EXISTS organizations (
            id SERIAL PRIMARY KEY,
            uuid VARCHAR(64) UNIQUE,
            name VARCHAR(255) UNIQUE NOT NULL,
            slug VARCHAR(255) UNIQUE,
            owner_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_orgs_owner ON organizations(owner_user_id)", ()),
    (
        """
        CREATE TABLE IF NOT EXISTS org_members (
            org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role VARCHAR(32) DEFAULT 'member',
            status VARCHAR(32) DEFAULT 'active',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (org_id, user_id)
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_org_members_user ON org_members(user_id)", ()),
    (
        """
        CREATE TABLE IF NOT EXISTS teams (
            id SERIAL PRIMARY KEY,
            org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            slug VARCHAR(255),
            description TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (org_id, name)
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_teams_org ON teams(org_id)", ()),
    (
        """
        CREATE TABLE IF NOT EXISTS team_members (
            team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role VARCHAR(32) DEFAULT 'member',
            status VARCHAR(32) DEFAULT 'active',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (team_id, user_id)
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_team_members_user ON team_members(user_id)", ()),
    (
        """
        CREATE TABLE IF NOT EXISTS org_invites (
            id SERIAL PRIMARY KEY,
            code TEXT UNIQUE NOT NULL,
            org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE,
            role_to_grant TEXT DEFAULT 'member',
            created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            max_uses INTEGER DEFAULT 1,
            uses_count INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT TRUE,
            allowed_email_domain TEXT,
            description TEXT,
            metadata JSONB
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_org_invites_code ON org_invites(code)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_org_invites_org_active ON org_invites(org_id, is_active)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_org_invites_expires ON org_invites(expires_at)", ()),
    (
        """
        CREATE TABLE IF NOT EXISTS org_invite_redemptions (
            id SERIAL PRIMARY KEY,
            invite_id INTEGER NOT NULL REFERENCES org_invites(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            redeemed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT,
            user_agent TEXT,
            UNIQUE(invite_id, user_id)
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_invite_redemptions_invite ON org_invite_redemptions(invite_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_invite_redemptions_user ON org_invite_redemptions(user_id)", ()),
    ("ALTER TABLE registration_codes ADD COLUMN IF NOT EXISTS org_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL", ()),
    ("ALTER TABLE registration_codes ADD COLUMN IF NOT EXISTS org_role VARCHAR(50)", ()),
    ("ALTER TABLE registration_codes ADD COLUMN IF NOT EXISTS team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL", ()),
]


_CREATE_BILLING_TABLES = [
    (
        """
        CREATE TABLE IF NOT EXISTS org_budgets (
            org_id INTEGER PRIMARY KEY REFERENCES organizations(id) ON DELETE CASCADE,
            budgets_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_org_budgets_org ON org_budgets(org_id)", ()),
]


_CREATE_API_KEYS_TABLES = [
    (
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            key_hash TEXT UNIQUE NOT NULL,
            key_id VARCHAR(32),
            key_prefix VARCHAR(16) NOT NULL,
            name VARCHAR(255),
            description TEXT,
            scope VARCHAR(50),
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            last_used_at TIMESTAMP,
            last_used_ip VARCHAR(45),
            usage_count INTEGER DEFAULT 0,
            rate_limit INTEGER,
            allowed_ips TEXT,
            metadata JSONB,
            rotated_from INTEGER REFERENCES api_keys(id),
            rotated_to INTEGER REFERENCES api_keys(id),
            revoked_at TIMESTAMP,
            revoked_by INTEGER,
            revoke_reason TEXT
        )
        """,
        (),
    ),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS scope VARCHAR(50)", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'active'", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS key_id VARCHAR(32)", ()),
    ("ALTER TABLE api_keys ALTER COLUMN key_hash TYPE TEXT", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS metadata JSONB", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS rotated_from INTEGER REFERENCES api_keys(id)", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS rotated_to INTEGER REFERENCES api_keys(id)", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMP", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS revoked_by INTEGER", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS revoke_reason TEXT", ()),
    # Virtual key fields
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS is_virtual BOOLEAN DEFAULT FALSE", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS parent_key_id INTEGER REFERENCES api_keys(id)", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS org_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS llm_budget_day_tokens BIGINT", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS llm_budget_month_tokens BIGINT", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS llm_budget_day_usd DOUBLE PRECISION", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS llm_budget_month_usd DOUBLE PRECISION", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS llm_allowed_endpoints TEXT", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS llm_allowed_providers TEXT", ()),
    ("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS llm_allowed_models TEXT", ()),
    # Helpful indexes
    ("CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash)", ()),
    ("CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_key_id ON api_keys(key_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_api_keys_virtual ON api_keys(is_virtual)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_api_keys_org ON api_keys(org_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_api_keys_team ON api_keys(team_id)", ()),
    # Audit log
    (
        """
        CREATE TABLE IF NOT EXISTS api_key_audit_log (
            id SERIAL PRIMARY KEY,
            api_key_id INTEGER NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
            action VARCHAR(50) NOT NULL,
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            ip_address VARCHAR(45),
            user_agent TEXT,
            details JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_api_key_audit_log_api_key_id ON api_key_audit_log(api_key_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_api_key_audit_log_created_at ON api_key_audit_log(created_at)", ()),
]

_CREATE_USER_PROVIDER_SECRETS = [
    (
        """
        CREATE TABLE IF NOT EXISTS user_provider_secrets (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            encrypted_blob TEXT NOT NULL,
            key_hint TEXT,
            metadata TEXT,
            created_by INTEGER,
            updated_by INTEGER,
            revoked_by INTEGER,
            revoked_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP WITH TIME ZONE,
            CONSTRAINT uq_user_provider_secrets UNIQUE (user_id, provider)
        )
        """,
        (),
    ),
    ("ALTER TABLE user_provider_secrets ADD COLUMN IF NOT EXISTS encrypted_blob TEXT", ()),
    ("ALTER TABLE user_provider_secrets ADD COLUMN IF NOT EXISTS key_hint TEXT", ()),
    ("ALTER TABLE user_provider_secrets ADD COLUMN IF NOT EXISTS metadata TEXT", ()),
    ("ALTER TABLE user_provider_secrets ADD COLUMN IF NOT EXISTS created_by INTEGER", ()),
    ("ALTER TABLE user_provider_secrets ADD COLUMN IF NOT EXISTS updated_by INTEGER", ()),
    ("ALTER TABLE user_provider_secrets ADD COLUMN IF NOT EXISTS revoked_by INTEGER", ()),
    ("ALTER TABLE user_provider_secrets ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMP WITH TIME ZONE", ()),
    ("ALTER TABLE user_provider_secrets ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP", ()),
    ("ALTER TABLE user_provider_secrets ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP", ()),
    ("ALTER TABLE user_provider_secrets ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMP WITH TIME ZONE", ()),
    ("CREATE INDEX IF NOT EXISTS idx_user_provider_secrets_user_id ON user_provider_secrets(user_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_user_provider_secrets_provider ON user_provider_secrets(provider)", ()),
]

_CREATE_ORG_PROVIDER_SECRETS = [
    (
        """
        CREATE TABLE IF NOT EXISTS org_provider_secrets (
            id SERIAL PRIMARY KEY,
            scope_type TEXT NOT NULL,
            scope_id INTEGER NOT NULL,
            provider TEXT NOT NULL,
            encrypted_blob TEXT NOT NULL,
            key_hint TEXT,
            metadata TEXT,
            created_by INTEGER,
            updated_by INTEGER,
            revoked_by INTEGER,
            revoked_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP WITH TIME ZONE,
            CONSTRAINT uq_org_provider_secrets UNIQUE (scope_type, scope_id, provider)
        )
        """,
        (),
    ),
    ("ALTER TABLE org_provider_secrets ADD COLUMN IF NOT EXISTS encrypted_blob TEXT", ()),
    ("ALTER TABLE org_provider_secrets ADD COLUMN IF NOT EXISTS key_hint TEXT", ()),
    ("ALTER TABLE org_provider_secrets ADD COLUMN IF NOT EXISTS metadata TEXT", ()),
    ("ALTER TABLE org_provider_secrets ADD COLUMN IF NOT EXISTS created_by INTEGER", ()),
    ("ALTER TABLE org_provider_secrets ADD COLUMN IF NOT EXISTS updated_by INTEGER", ()),
    ("ALTER TABLE org_provider_secrets ADD COLUMN IF NOT EXISTS revoked_by INTEGER", ()),
    ("ALTER TABLE org_provider_secrets ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMP WITH TIME ZONE", ()),
    ("ALTER TABLE org_provider_secrets ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP", ()),
    ("ALTER TABLE org_provider_secrets ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP", ()),
    ("ALTER TABLE org_provider_secrets ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMP WITH TIME ZONE", ()),
    ("CREATE INDEX IF NOT EXISTS idx_org_provider_secrets_scope ON org_provider_secrets(scope_type, scope_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_org_provider_secrets_provider ON org_provider_secrets(provider)", ()),
]

_CREATE_IDENTITY_PROVIDERS = [
    (
        """
        CREATE TABLE IF NOT EXISTS identity_providers (
            id SERIAL PRIMARY KEY,
            slug TEXT NOT NULL,
            provider_type TEXT NOT NULL DEFAULT 'oidc',
            owner_scope_type TEXT NOT NULL DEFAULT 'global',
            owner_scope_id INTEGER NULL,
            enabled BOOLEAN NOT NULL DEFAULT FALSE,
            display_name TEXT NULL,
            issuer TEXT NOT NULL,
            discovery_url TEXT NULL,
            authorization_url TEXT NULL,
            token_url TEXT NULL,
            jwks_url TEXT NULL,
            client_id TEXT NULL,
            client_secret_ref TEXT NULL,
            claim_mapping_json TEXT NOT NULL DEFAULT '{}',
            provisioning_policy_json TEXT NOT NULL DEFAULT '{}',
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS provider_type TEXT NOT NULL DEFAULT 'oidc'", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS owner_scope_type TEXT NOT NULL DEFAULT 'global'", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS owner_scope_id INTEGER", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS enabled BOOLEAN NOT NULL DEFAULT FALSE", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS display_name TEXT", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS issuer TEXT", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS discovery_url TEXT", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS authorization_url TEXT", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS token_url TEXT", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS jwks_url TEXT", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS client_id TEXT", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS client_secret_ref TEXT", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS claim_mapping_json TEXT NOT NULL DEFAULT '{}'", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS provisioning_policy_json TEXT NOT NULL DEFAULT '{}'", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS created_by INTEGER", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS updated_by INTEGER", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP", ()),
    ("ALTER TABLE identity_providers ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP", ()),
    (
        "CREATE INDEX IF NOT EXISTS idx_identity_providers_scope "
        "ON identity_providers(owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_identity_providers_enabled "
        "ON identity_providers(enabled)",
        (),
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_identity_providers_scope_slug "
        "ON identity_providers(slug, owner_scope_type, COALESCE(owner_scope_id, 0))",
        (),
    ),
]

_CREATE_FEDERATED_IDENTITIES = [
    (
        """
        CREATE TABLE IF NOT EXISTS federated_identities (
            id SERIAL PRIMARY KEY,
            identity_provider_id INTEGER NOT NULL REFERENCES identity_providers(id) ON DELETE CASCADE,
            external_subject TEXT NOT NULL,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            external_username TEXT NULL,
            external_email TEXT NULL,
            last_claims_hash TEXT NULL,
            last_seen_at TIMESTAMPTZ NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    ("ALTER TABLE federated_identities ADD COLUMN IF NOT EXISTS external_username TEXT", ()),
    ("ALTER TABLE federated_identities ADD COLUMN IF NOT EXISTS external_email TEXT", ()),
    ("ALTER TABLE federated_identities ADD COLUMN IF NOT EXISTS last_claims_hash TEXT", ()),
    ("ALTER TABLE federated_identities ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ", ()),
    ("ALTER TABLE federated_identities ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active'", ()),
    ("ALTER TABLE federated_identities ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP", ()),
    ("ALTER TABLE federated_identities ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP", ()),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_federated_identities_provider_subject "
        "ON federated_identities(identity_provider_id, external_subject)",
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_federated_identities_user_id ON federated_identities(user_id)", ()),
    (
        "CREATE INDEX IF NOT EXISTS idx_federated_identities_provider_id "
        "ON federated_identities(identity_provider_id)",
        (),
    ),
]

_CREATE_SECRET_BACKENDS = [
    (
        """
        CREATE TABLE IF NOT EXISTS secret_backends (
            name TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'enabled',
            capabilities_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    ("ALTER TABLE secret_backends ADD COLUMN IF NOT EXISTS display_name TEXT", ()),
    ("ALTER TABLE secret_backends ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'enabled'", ()),
    ("ALTER TABLE secret_backends ADD COLUMN IF NOT EXISTS capabilities_json TEXT NOT NULL DEFAULT '{}'", ()),
    ("ALTER TABLE secret_backends ADD COLUMN IF NOT EXISTS metadata_json TEXT NOT NULL DEFAULT '{}'", ()),
    ("ALTER TABLE secret_backends ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP", ()),
    ("ALTER TABLE secret_backends ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP", ()),
    ("CREATE INDEX IF NOT EXISTS idx_secret_backends_status ON secret_backends(status)", ()),
]

_CREATE_MANAGED_SECRET_REFS = [
    (
        """
        CREATE TABLE IF NOT EXISTS managed_secret_refs (
            id SERIAL PRIMARY KEY,
            backend_name TEXT NOT NULL REFERENCES secret_backends(name) ON DELETE RESTRICT,
            owner_scope_type TEXT NOT NULL,
            owner_scope_id INTEGER NOT NULL,
            provider_key TEXT NOT NULL,
            backend_ref TEXT NULL,
            display_name TEXT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            last_resolved_at TIMESTAMPTZ NULL,
            expires_at TIMESTAMPTZ NULL,
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            revoked_by INTEGER NULL,
            revoked_at TIMESTAMPTZ NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    ("ALTER TABLE managed_secret_refs ADD COLUMN IF NOT EXISTS backend_ref TEXT", ()),
    ("ALTER TABLE managed_secret_refs ADD COLUMN IF NOT EXISTS display_name TEXT", ()),
    ("ALTER TABLE managed_secret_refs ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active'", ()),
    ("ALTER TABLE managed_secret_refs ADD COLUMN IF NOT EXISTS metadata_json TEXT NOT NULL DEFAULT '{}'", ()),
    ("ALTER TABLE managed_secret_refs ADD COLUMN IF NOT EXISTS last_resolved_at TIMESTAMPTZ", ()),
    ("ALTER TABLE managed_secret_refs ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ", ()),
    ("ALTER TABLE managed_secret_refs ADD COLUMN IF NOT EXISTS created_by INTEGER", ()),
    ("ALTER TABLE managed_secret_refs ADD COLUMN IF NOT EXISTS updated_by INTEGER", ()),
    ("ALTER TABLE managed_secret_refs ADD COLUMN IF NOT EXISTS revoked_by INTEGER", ()),
    ("ALTER TABLE managed_secret_refs ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMPTZ", ()),
    ("ALTER TABLE managed_secret_refs ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP", ()),
    ("ALTER TABLE managed_secret_refs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP", ()),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_managed_secret_refs_scope_provider "
        "ON managed_secret_refs(backend_name, owner_scope_type, owner_scope_id, provider_key)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_managed_secret_refs_scope "
        "ON managed_secret_refs(owner_scope_type, owner_scope_id)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_managed_secret_refs_status "
        "ON managed_secret_refs(status)",
        (),
    ),
]

_CREATE_FEDERATED_MANAGED_GRANTS = [
    (
        """
        CREATE TABLE IF NOT EXISTS federated_managed_grants (
            id SERIAL PRIMARY KEY,
            identity_provider_id INTEGER NOT NULL REFERENCES identity_providers(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            grant_kind TEXT NOT NULL,
            target_ref TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    ("ALTER TABLE federated_managed_grants ADD COLUMN IF NOT EXISTS grant_kind TEXT NOT NULL DEFAULT 'org'", ()),
    ("ALTER TABLE federated_managed_grants ADD COLUMN IF NOT EXISTS target_ref TEXT NOT NULL DEFAULT ''", ()),
    ("ALTER TABLE federated_managed_grants ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP", ()),
    ("ALTER TABLE federated_managed_grants ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP", ()),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_federated_managed_grants_target "
        "ON federated_managed_grants(identity_provider_id, user_id, grant_kind, target_ref)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_federated_managed_grants_user_id "
        "ON federated_managed_grants(user_id)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_federated_managed_grants_provider_id "
        "ON federated_managed_grants(identity_provider_id)",
        (),
    ),
]

_CREATE_BYOK_OAUTH_STATE = [
    (
        """
        CREATE TABLE IF NOT EXISTS byok_oauth_state (
            state TEXT NOT NULL,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            auth_session_id TEXT NOT NULL,
            redirect_uri TEXT NOT NULL,
            pkce_verifier_encrypted TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            consumed_at TIMESTAMP WITH TIME ZONE,
            return_path TEXT,
            PRIMARY KEY (state, user_id)
        )
        """,
        (),
    ),
    ("ALTER TABLE byok_oauth_state ADD COLUMN IF NOT EXISTS provider TEXT", ()),
    ("ALTER TABLE byok_oauth_state ADD COLUMN IF NOT EXISTS auth_session_id TEXT", ()),
    ("ALTER TABLE byok_oauth_state ADD COLUMN IF NOT EXISTS redirect_uri TEXT", ()),
    ("ALTER TABLE byok_oauth_state ADD COLUMN IF NOT EXISTS pkce_verifier_encrypted TEXT", ()),
    (
        "ALTER TABLE byok_oauth_state ADD COLUMN IF NOT EXISTS created_at "
        "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP",
        (),
    ),
    ("ALTER TABLE byok_oauth_state ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP WITH TIME ZONE", ()),
    ("ALTER TABLE byok_oauth_state ADD COLUMN IF NOT EXISTS consumed_at TIMESTAMP WITH TIME ZONE", ()),
    ("ALTER TABLE byok_oauth_state ADD COLUMN IF NOT EXISTS return_path TEXT", ()),
    (
        "CREATE INDEX IF NOT EXISTS idx_byok_oauth_state_provider_expires "
        "ON byok_oauth_state(provider, expires_at)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_byok_oauth_state_user_provider_consumed "
        "ON byok_oauth_state(user_id, provider, consumed_at)",
        (),
    ),
]

_CREATE_LLM_PROVIDER_OVERRIDES = [
    (
        """
        CREATE TABLE IF NOT EXISTS llm_provider_overrides (
            provider TEXT PRIMARY KEY,
            is_enabled BOOLEAN,
            allowed_models TEXT,
            config_json TEXT,
            secret_blob TEXT,
            api_key_hint TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    ("ALTER TABLE llm_provider_overrides ADD COLUMN IF NOT EXISTS is_enabled BOOLEAN", ()),
    ("ALTER TABLE llm_provider_overrides ADD COLUMN IF NOT EXISTS allowed_models TEXT", ()),
    ("ALTER TABLE llm_provider_overrides ADD COLUMN IF NOT EXISTS config_json TEXT", ()),
    ("ALTER TABLE llm_provider_overrides ADD COLUMN IF NOT EXISTS secret_blob TEXT", ()),
    ("ALTER TABLE llm_provider_overrides ADD COLUMN IF NOT EXISTS api_key_hint TEXT", ()),
    ("ALTER TABLE llm_provider_overrides ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP", ()),
    ("ALTER TABLE llm_provider_overrides ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP", ()),
    ("CREATE INDEX IF NOT EXISTS idx_llm_provider_overrides_enabled ON llm_provider_overrides(is_enabled)", ()),
]


_CREATE_USAGE_TABLES = [
    # usage_log + usage_daily
    (
        """
        CREATE TABLE IF NOT EXISTS usage_log (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            key_id INTEGER REFERENCES api_keys(id) ON DELETE SET NULL,
            endpoint TEXT,
            status INTEGER,
            latency_ms INTEGER,
            bytes BIGINT,
            bytes_in BIGINT,
            meta JSONB,
            request_id TEXT
        )
        """,
        (),
    ),
    ("ALTER TABLE usage_log ADD COLUMN IF NOT EXISTS request_id TEXT", ()),
    ("ALTER TABLE usage_log ADD COLUMN IF NOT EXISTS bytes_in BIGINT", ()),
    ("CREATE INDEX IF NOT EXISTS idx_usage_log_ts ON usage_log(ts)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_usage_log_user ON usage_log(user_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_usage_log_status ON usage_log(status)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_usage_log_endpoint ON usage_log(endpoint)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_usage_log_request_id ON usage_log(request_id)", ()),
    (
        """
        CREATE TABLE IF NOT EXISTS usage_daily (
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            day DATE NOT NULL,
            requests INTEGER DEFAULT 0,
            errors INTEGER DEFAULT 0,
            bytes_total BIGINT DEFAULT 0,
            bytes_in_total BIGINT DEFAULT 0,
            latency_avg_ms DOUBLE PRECISION,
            PRIMARY KEY (user_id, day)
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_usage_daily_day_user ON usage_daily(day, user_id)", ()),
    ("ALTER TABLE usage_daily ADD COLUMN IF NOT EXISTS bytes_in_total BIGINT", ()),
    # llm_usage_log + llm_usage_daily
    (
        """
        CREATE TABLE IF NOT EXISTS llm_usage_log (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            key_id INTEGER REFERENCES api_keys(id) ON DELETE SET NULL,
            endpoint TEXT,
            operation TEXT,
            provider TEXT,
            model TEXT,
            status INTEGER,
            latency_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            prompt_cost_usd DOUBLE PRECISION,
            completion_cost_usd DOUBLE PRECISION,
            total_cost_usd DOUBLE PRECISION,
            currency TEXT DEFAULT 'USD',
            estimated BOOLEAN DEFAULT FALSE,
            request_id TEXT,
            remote_ip TEXT,
            user_agent TEXT,
            token_name TEXT,
            conversation_id TEXT,
            cached_input_tokens INTEGER,
            cache_write_input_tokens INTEGER,
            cache_read_input_tokens INTEGER,
            billable_input_tokens INTEGER,
            reasoning_tokens INTEGER,
            choice_count INTEGER,
            estimate_source TEXT,
            prompt_fingerprint TEXT,
            prompt_fingerprint_version TEXT,
            world_book_fingerprint TEXT,
            raw_usage_metadata_json TEXT
        )
        """,
        (),
    ),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS remote_ip TEXT", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS user_agent TEXT", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS token_name TEXT", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS conversation_id TEXT", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS cached_input_tokens INTEGER", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS cache_write_input_tokens INTEGER", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS cache_read_input_tokens INTEGER", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS billable_input_tokens INTEGER", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS reasoning_tokens INTEGER", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS choice_count INTEGER", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS estimate_source TEXT", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS prompt_fingerprint TEXT", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS prompt_fingerprint_version TEXT", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS world_book_fingerprint TEXT", ()),
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS raw_usage_metadata_json TEXT", ()),
    ("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_ts ON llm_usage_log(ts)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_user ON llm_usage_log(user_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_provider_model ON llm_usage_log(provider, model)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_op_ts ON llm_usage_log(operation, ts)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_key_ts ON llm_usage_log(key_id, ts)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_operation ON llm_usage_log(operation)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_remote_ip_ts ON llm_usage_log(remote_ip, ts)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_token_name_ts ON llm_usage_log(token_name, ts)", ()),
    (
        """
        CREATE TABLE IF NOT EXISTS llm_usage_daily (
            day DATE NOT NULL,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            operation TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            requests INTEGER DEFAULT 0,
            errors INTEGER DEFAULT 0,
            input_tokens BIGINT DEFAULT 0,
            output_tokens BIGINT DEFAULT 0,
            total_tokens BIGINT DEFAULT 0,
            total_cost_usd DOUBLE PRECISION DEFAULT 0.0,
            latency_avg_ms DOUBLE PRECISION,
            PRIMARY KEY (day, user_id, operation, provider, model)
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_daily_day_user_op_prov_model ON llm_usage_daily(day, user_id, operation, provider, model)",
        (),
    ),
    # org/team role permissions (scoped RBAC)
    (
        """
        CREATE TABLE IF NOT EXISTS org_role_permissions (
            org_role TEXT NOT NULL,
            permission_id INTEGER NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
            PRIMARY KEY (org_role, permission_id),
            CHECK (org_role IN ('owner', 'admin', 'lead', 'member'))
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_org_role_permissions_permission_id ON org_role_permissions(permission_id)", ()),
    (
        """
        CREATE TABLE IF NOT EXISTS team_role_permissions (
            team_role TEXT NOT NULL,
            permission_id INTEGER NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
            PRIMARY KEY (team_role, permission_id),
            CHECK (team_role IN ('owner', 'admin', 'lead', 'member'))
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_team_role_permissions_permission_id ON team_role_permissions(permission_id)", ()),
    (
        """
        INSERT INTO org_role_permissions (org_role, permission_id)
        SELECT 'owner', rp.permission_id
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        WHERE r.name = 'admin'
        ON CONFLICT (org_role, permission_id) DO NOTHING
        """,
        (),
    ),
    (
        """
        INSERT INTO org_role_permissions (org_role, permission_id)
        SELECT 'admin', rp.permission_id
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        WHERE r.name = 'admin'
        ON CONFLICT (org_role, permission_id) DO NOTHING
        """,
        (),
    ),
    (
        """
        INSERT INTO org_role_permissions (org_role, permission_id)
        SELECT 'lead', rp.permission_id
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        WHERE r.name = 'reviewer'
        ON CONFLICT (org_role, permission_id) DO NOTHING
        """,
        (),
    ),
    (
        """
        INSERT INTO org_role_permissions (org_role, permission_id)
        SELECT 'member', rp.permission_id
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        WHERE r.name = 'user'
        ON CONFLICT (org_role, permission_id) DO NOTHING
        """,
        (),
    ),
    (
        """
        INSERT INTO team_role_permissions (team_role, permission_id)
        SELECT 'owner', rp.permission_id
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        WHERE r.name = 'admin'
        ON CONFLICT (team_role, permission_id) DO NOTHING
        """,
        (),
    ),
    (
        """
        INSERT INTO team_role_permissions (team_role, permission_id)
        SELECT 'admin', rp.permission_id
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        WHERE r.name = 'admin'
        ON CONFLICT (team_role, permission_id) DO NOTHING
        """,
        (),
    ),
    (
        """
        INSERT INTO team_role_permissions (team_role, permission_id)
        SELECT 'lead', rp.permission_id
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        WHERE r.name = 'reviewer'
        ON CONFLICT (team_role, permission_id) DO NOTHING
        """,
        (),
    ),
    (
        """
        INSERT INTO team_role_permissions (team_role, permission_id)
        SELECT 'member', rp.permission_id
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        WHERE r.name = 'user'
        ON CONFLICT (team_role, permission_id) DO NOTHING
        """,
        (),
    ),
]


_CREATE_VK_COUNTERS = [
    (
        """
        CREATE TABLE IF NOT EXISTS vk_jwt_counters (
            jti TEXT NOT NULL,
            counter_type TEXT NOT NULL,
            count BIGINT DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (jti, counter_type)
        )
        """,
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS vk_api_key_counters (
            api_key_id INTEGER NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
            counter_type TEXT NOT NULL,
            count BIGINT DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (api_key_id, counter_type)
        )
        """,
        (),
    ),
]


_CREATE_GENERATED_FILES_TABLES = [
    (
        """
        CREATE TABLE IF NOT EXISTS generated_files (
            id SERIAL PRIMARY KEY,
            uuid TEXT UNIQUE NOT NULL,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            org_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL,
            team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL,
            filename TEXT NOT NULL,
            original_filename TEXT,
            storage_path TEXT NOT NULL,
            mime_type TEXT,
            file_size_bytes BIGINT NOT NULL DEFAULT 0,
            checksum TEXT,
            file_category TEXT NOT NULL,
            source_feature TEXT NOT NULL,
            source_ref TEXT,
            folder_tag TEXT,
            tags JSONB,
            is_transient BOOLEAN DEFAULT FALSE,
            expires_at TIMESTAMP,
            retention_policy TEXT DEFAULT 'user_default',
            is_deleted BOOLEAN DEFAULT FALSE,
            deleted_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accessed_at TIMESTAMP
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_generated_files_user_id ON generated_files(user_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_generated_files_org_id ON generated_files(org_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_generated_files_team_id ON generated_files(team_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_generated_files_uuid ON generated_files(uuid)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_generated_files_category ON generated_files(file_category)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_generated_files_source_feature ON generated_files(source_feature)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_generated_files_folder_tag ON generated_files(folder_tag)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_generated_files_is_deleted ON generated_files(is_deleted)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_generated_files_expires_at ON generated_files(expires_at)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_generated_files_created_at ON generated_files(created_at)", ()),
    (
        "CREATE INDEX IF NOT EXISTS idx_generated_files_user_category "
        "ON generated_files(user_id, file_category, is_deleted)",
        (),
    ),
]

_CREATE_ORG_STT_SETTINGS = [
    (
        """
        CREATE TABLE IF NOT EXISTS org_stt_settings (
            org_id INTEGER PRIMARY KEY REFERENCES organizations(id) ON DELETE CASCADE,
            delete_audio_after_success BOOLEAN NOT NULL DEFAULT TRUE,
            audio_retention_hours DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            redact_pii BOOLEAN NOT NULL DEFAULT FALSE,
            allow_unredacted_partials BOOLEAN NOT NULL DEFAULT FALSE,
            redact_categories_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            updated_by INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_org_stt_settings_updated_by ON org_stt_settings(updated_by)", ()),
]

_CREATE_DATA_SUBJECT_REQUESTS_TABLES = [
    (
        """
        CREATE TABLE IF NOT EXISTS data_subject_requests (
            id SERIAL PRIMARY KEY,
            client_request_id TEXT NOT NULL UNIQUE,
            requester_identifier TEXT NOT NULL,
            resolved_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
            request_type TEXT NOT NULL,
            status TEXT NOT NULL,
            selected_categories JSONB NOT NULL DEFAULT '[]'::jsonb,
            preview_summary JSONB NOT NULL DEFAULT '[]'::jsonb,
            coverage_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            requested_by_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
            requested_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            notes TEXT NULL
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_data_subject_requests_requester "
        "ON data_subject_requests(requester_identifier)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_data_subject_requests_resolved_user "
        "ON data_subject_requests(resolved_user_id)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_data_subject_requests_type "
        "ON data_subject_requests(request_type)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_data_subject_requests_status "
        "ON data_subject_requests(status)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_data_subject_requests_requested_at "
        "ON data_subject_requests(requested_at)",
        (),
    ),
]


_CREATE_ADMIN_MONITORING_TABLES = [
    (
        """
        CREATE TABLE IF NOT EXISTS admin_alert_rules (
            id SERIAL PRIMARY KEY,
            metric TEXT NOT NULL,
            operator TEXT NOT NULL,
            threshold DOUBLE PRECISION NOT NULL,
            duration_minutes INTEGER NOT NULL,
            severity TEXT NOT NULL,
            enabled BOOLEAN DEFAULT TRUE,
            created_by_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
            updated_by_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_admin_alert_rules_created_at "
        "ON admin_alert_rules(created_at)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_admin_alert_rules_metric_enabled "
        "ON admin_alert_rules(metric, enabled)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS admin_alert_state (
            alert_identity TEXT PRIMARY KEY,
            assigned_to_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
            snoozed_until TIMESTAMPTZ NULL,
            escalated_severity TEXT NULL,
            acknowledged_at TIMESTAMPTZ NULL,
            dismissed_at TIMESTAMPTZ NULL,
            updated_by_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_admin_alert_state_updated_at "
        "ON admin_alert_state(updated_at)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_admin_alert_state_assigned_user "
        "ON admin_alert_state(assigned_to_user_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS admin_alert_events (
            id SERIAL PRIMARY KEY,
            alert_identity TEXT NOT NULL,
            action TEXT NOT NULL,
            actor_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
            details_json TEXT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_admin_alert_events_identity_created "
        "ON admin_alert_events(alert_identity, created_at DESC)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_admin_alert_events_created_at "
        "ON admin_alert_events(created_at DESC)",
        (),
    ),
]

_CREATE_BACKUP_SCHEDULES_TABLES = [
    (
        """
        CREATE TABLE IF NOT EXISTS backup_schedules (
            id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            target_user_id INTEGER NULL,
            target_scope_key TEXT NOT NULL,
            frequency TEXT NOT NULL,
            time_of_day TEXT NOT NULL,
            timezone TEXT NOT NULL,
            anchor_day_of_week INTEGER NULL,
            anchor_day_of_month INTEGER NULL,
            retention_count INTEGER NOT NULL,
            is_paused BOOLEAN NOT NULL DEFAULT FALSE,
            created_by_user_id INTEGER NULL,
            updated_by_user_id INTEGER NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            next_run_at TIMESTAMPTZ NULL,
            last_run_at TIMESTAMPTZ NULL,
            last_status TEXT NULL,
            last_job_id TEXT NULL,
            last_error TEXT NULL,
            deleted_at TIMESTAMPTZ NULL
        )
        """,
        (),
    ),
    (
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_backup_schedules_target_scope_active
        ON backup_schedules(target_scope_key)
        WHERE deleted_at IS NULL
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_backup_schedules_next_run_at ON backup_schedules(next_run_at)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_backup_schedules_deleted_at ON backup_schedules(deleted_at)", ()),
    (
        """
        CREATE TABLE IF NOT EXISTS backup_schedule_runs (
            id TEXT PRIMARY KEY,
            schedule_id TEXT NOT NULL REFERENCES backup_schedules(id) ON DELETE CASCADE,
            scheduled_for TIMESTAMPTZ NOT NULL,
            run_slot_key TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            job_id TEXT NULL,
            error TEXT NULL,
            enqueued_at TIMESTAMPTZ NULL,
            started_at TIMESTAMPTZ NULL,
            completed_at TIMESTAMPTZ NULL
        )
        """,
        (),
    ),
    ("CREATE INDEX IF NOT EXISTS idx_backup_schedule_runs_schedule_id ON backup_schedule_runs(schedule_id)", ()),
    ("CREATE INDEX IF NOT EXISTS idx_backup_schedule_runs_scheduled_for ON backup_schedule_runs(scheduled_for)", ()),
]

_CREATE_MAINTENANCE_ROTATION_RUNS_TABLES = [
    (
        """
        CREATE TABLE IF NOT EXISTS maintenance_rotation_runs (
            id TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            domain TEXT NULL,
            queue TEXT NULL,
            job_type TEXT NULL,
            fields_json TEXT NOT NULL,
            "limit" INTEGER NULL,
            affected_count INTEGER NULL,
            requested_by_user_id INTEGER NULL,
            requested_by_label TEXT NULL,
            confirmation_recorded BOOLEAN NOT NULL DEFAULT FALSE,
            job_id TEXT NULL,
            scope_summary TEXT NOT NULL,
            key_source TEXT NOT NULL,
            error_message TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            started_at TIMESTAMPTZ NULL,
            completed_at TIMESTAMPTZ NULL
        )
        """,
        (),
    ),
    (
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_maintenance_rotation_runs_active_execute
        ON maintenance_rotation_runs(mode)
        WHERE mode = 'execute' AND status IN ('queued', 'running')
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_maintenance_rotation_runs_created_at "
        "ON maintenance_rotation_runs(created_at)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_maintenance_rotation_runs_status "
        "ON maintenance_rotation_runs(status)",
        (),
    ),
]

_CREATE_BYOK_VALIDATION_RUNS_TABLES = [
    (
        """
        CREATE TABLE IF NOT EXISTS byok_validation_runs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            org_id INTEGER NULL,
            provider TEXT NULL,
            keys_checked INTEGER NULL,
            valid_count INTEGER NULL,
            invalid_count INTEGER NULL,
            error_count INTEGER NULL,
            requested_by_user_id INTEGER NULL,
            requested_by_label TEXT NULL,
            job_id TEXT NULL,
            scope_summary TEXT NOT NULL,
            error_message TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            started_at TIMESTAMPTZ NULL,
            completed_at TIMESTAMPTZ NULL
        )
        """,
        (),
    ),
    (
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_byok_validation_runs_active
        ON byok_validation_runs((1))
        WHERE status IN ('queued', 'running')
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_byok_validation_runs_created_at "
        "ON byok_validation_runs(created_at)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_byok_validation_runs_status "
        "ON byok_validation_runs(status)",
        (),
    ),
]


async def ensure_tool_catalogs_tables_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure tool catalogs tables exist on PostgreSQL backends.

    Returns True if ensured (or not needed), False if skipped due to non-PG backend.
    """
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False  # not postgres
        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"PG ensure authnz core tables before tool catalogs failed: {exc}")
        for sql, params in _CREATE_TOOL_CATALOGS:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                # Continue attempting subsequent statements; log and surface at the end
                logger.debug(f"PG ensure tool catalogs DDL failed: {exc}")
        logger.info("Ensured PostgreSQL tool catalogs tables (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL tool catalogs tables: {exc}")
        return False


async def ensure_backup_schedules_tables_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure backup schedule tables exist on PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False
        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"PG ensure authnz core tables before backup schedules failed: {exc}")
        for sql, params in _CREATE_BACKUP_SCHEDULES_TABLES:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure backup schedules DDL failed: {exc}")
        logger.info("Ensured PostgreSQL backup schedule tables (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL backup schedule tables: {exc}")
        return False


async def ensure_maintenance_rotation_runs_table_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure maintenance rotation run tables exist on PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False
        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                f"PG ensure authnz core tables before maintenance rotation runs failed: {exc}"
            )
        for sql, params in _CREATE_MAINTENANCE_ROTATION_RUNS_TABLES:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure maintenance rotation run DDL failed: {exc}")
        logger.info("Ensured PostgreSQL maintenance rotation runs table (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL maintenance rotation runs table: {exc}")
        return False


async def ensure_byok_validation_runs_table_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure BYOK validation run tables exist on PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False
        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                f"PG ensure authnz core tables before BYOK validation runs failed: {exc}"
            )
        for sql, params in _CREATE_BYOK_VALIDATION_RUNS_TABLES:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure BYOK validation run DDL failed: {exc}")
        logger.info("Ensured PostgreSQL BYOK validation runs table (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL BYOK validation runs table: {exc}")
        return False


async def ensure_mcp_hub_tables_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure MCP Hub management tables exist on PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False  # not postgres
        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"PG ensure authnz core tables before mcp hub tables failed: {exc}")
        for sql, params in _CREATE_MCP_HUB_TABLES:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure mcp hub tables DDL failed: {exc}")
        logger.info("Ensured PostgreSQL MCP Hub tables (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL MCP Hub tables: {exc}")
        return False


async def ensure_privilege_snapshots_table_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure privilege_snapshots table exists for PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False
        for sql, params in _CREATE_PRIVILEGE_SNAPSHOTS:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure privilege_snapshots DDL failed: {exc}")
        logger.info("Ensured PostgreSQL privilege_snapshots table (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL privilege_snapshots table: {exc}")
        return False


async def ensure_authnz_core_tables_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure core AuthNZ tables exist for PostgreSQL backends.

    This covers audit_logs, sessions, registration_codes, RBAC tables,
    role/user rate-limits, and the organizations/teams hierarchy. It is
    intended as a bootstrap guardrail for Postgres deployments that have
    not yet run dedicated migrations for these tables.
    """
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False
        for sql, params in _CREATE_AUTHNZ_CORE_TABLES:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure authnz core tables DDL failed: {exc}")
        logger.info(
            "Ensured PostgreSQL AuthNZ core tables "
            "(audit_logs, sessions, registration_codes, RBAC, orgs/teams)"
        )
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL AuthNZ core tables: {exc}")
        return False


def _parse_json_payload_pg(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.debug(f"PG budgets: invalid JSON payload: {exc}")
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _normalize_threshold_list_pg(values: Any) -> list[int] | None:
    if not isinstance(values, list):
        return None
    if not values:
        return None
    cleaned: list[int] = []
    for val in values:
        try:
            num = int(val)
        except (TypeError, ValueError):
            continue
        if num < 1 or num > 100:
            continue
        cleaned.append(num)
    if not cleaned:
        return None
    return sorted(set(cleaned))


def _coerce_alert_thresholds_pg(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return {"global": value}
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        if "global" in value:
            out["global"] = value.get("global")
        if "per_metric" in value:
            out["per_metric"] = value.get("per_metric")
        return out or None
    return None


def _coerce_enforcement_mode_pg(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {"global": value}
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        if "global" in value:
            out["global"] = value.get("global")
        if "per_metric" in value:
            out["per_metric"] = value.get("per_metric")
        return out or None
    return None


def _normalize_alert_thresholds_pg(value: Any) -> dict[str, Any] | None:
    payload = _coerce_alert_thresholds_pg(value)
    if payload is None:
        return None
    out: dict[str, Any] = {}
    global_values = payload.get("global")
    if global_values is not None:
        normalized = _normalize_threshold_list_pg(global_values)
        if normalized:
            out["global"] = normalized
    per_metric = payload.get("per_metric")
    if isinstance(per_metric, dict):
        cleaned: dict[str, Any] = {}
        for key, values in per_metric.items():
            if key not in _BUDGET_FIELD_KEYS:
                continue
            normalized = _normalize_threshold_list_pg(values)
            if normalized:
                cleaned[key] = normalized
        if cleaned:
            out["per_metric"] = cleaned
    return out or None


def _normalize_enforcement_mode_pg(value: Any) -> dict[str, Any] | None:
    payload = _coerce_enforcement_mode_pg(value)
    if payload is None:
        return None
    out: dict[str, Any] = {}
    global_value = payload.get("global")
    if isinstance(global_value, str) and global_value in {"none", "soft", "hard"}:
        out["global"] = global_value
    per_metric = payload.get("per_metric")
    if isinstance(per_metric, dict):
        cleaned: dict[str, Any] = {}
        for key, val in per_metric.items():
            if key not in _BUDGET_FIELD_KEYS:
                continue
            if isinstance(val, str) and val in {"none", "soft", "hard"}:
                cleaned[key] = val
        if cleaned:
            out["per_metric"] = cleaned
    return out or None


def _normalize_budget_payload_pg(raw: Any) -> dict[str, Any]:
    data = _parse_json_payload_pg(raw)
    if not data:
        return {}
    budgets: dict[str, Any] = {}
    if isinstance(data.get("budgets"), dict):
        budgets.update(data.get("budgets") or {})
    for key in _BUDGET_FIELD_KEYS:
        if key in data and key not in budgets:
            budgets[key] = data[key]
    payload: dict[str, Any] = {}
    for key in _BUDGET_FIELD_KEYS:
        if key in budgets:
            payload[key] = budgets[key]
    thresholds = _normalize_alert_thresholds_pg(data.get("alert_thresholds"))
    if thresholds:
        payload["alert_thresholds"] = thresholds
    enforcement = _normalize_enforcement_mode_pg(data.get("enforcement_mode"))
    if enforcement:
        payload["enforcement_mode"] = enforcement
    return payload


async def _normalize_org_budgets_pg(db_pool: DatabasePool) -> None:
    try:
        rows = await db_pool.fetch(
            "SELECT org_id, budgets_json FROM org_budgets WHERE budgets_json IS NOT NULL"
        )
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"PG budgets normalize fetch failed: {exc}")
        return

    for row in rows:
        org_id = row.get("org_id") if isinstance(row, dict) else None
        if org_id is None:
            continue
        raw_payload = row.get("budgets_json")
        normalized = _normalize_budget_payload_pg(raw_payload)
        if not normalized:
            continue
        try:
            current = _parse_json_payload_pg(raw_payload)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
            current = {}
        if json.dumps(current, sort_keys=True) == json.dumps(normalized, sort_keys=True):
            continue
        try:
            await db_pool.execute(
                """
                UPDATE org_budgets
                SET budgets_json = $2::jsonb, updated_at = CURRENT_TIMESTAMP
                WHERE org_id = $1
                """,
                org_id,
                json.dumps(normalized),
            )
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"PG budgets normalize update failed for org_id={org_id}: {exc}")


async def ensure_billing_tables_pg(
    pool: DatabasePool | None = None,
    *,
    run_backfill: bool = True,
) -> bool:
    """Ensure only OSS budget compatibility tables exist for PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False
        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"ensure_billing_tables_pg: core table ensure skipped/failed: {exc}")

        for sql, params in _CREATE_BILLING_TABLES:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure billing DDL failed: {exc}")

        if run_backfill:
            try:
                await _normalize_org_budgets_pg(db_pool)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG budgets normalize skipped/failed: {exc}")

        logger.info(
            "Ensured PostgreSQL OSS budget compatibility tables "
            "(org_budgets only; retired billing tables are left untouched if present)"
        )
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL OSS budget compatibility tables: {exc}")
        return False


async def ensure_api_keys_tables_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure api_keys + api_key_audit_log tables exist for PostgreSQL backends.

    Returns True when ensured (or already present), False when skipped due to a non-PG backend.
    """
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False

        # api_keys includes optional org/team-scoped columns with FK references.
        # Ensure the core AuthNZ tables (including organizations/teams) exist first
        # so additive `ALTER TABLE ... REFERENCES organizations/teams` statements
        # succeed in minimal test schemas.
        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"ensure_api_keys_tables_pg: core table ensure skipped/failed: {exc}")

        errors: list[Exception] = []
        for sql, params in _CREATE_API_KEYS_TABLES:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                errors.append(exc)
                logger.debug(f"PG ensure api keys DDL failed: {exc}")

        try:
            api_keys_ok = await db_pool.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'api_keys'
                )
                """
            )
            audit_ok = await db_pool.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'api_key_audit_log'
                )
                """
            )
            if not api_keys_ok or not audit_ok:
                logger.warning(
                    "PostgreSQL api_keys schema ensure did not complete "
                    f"(api_keys={bool(api_keys_ok)}, api_key_audit_log={bool(audit_ok)})"
                )
                return False
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            # Treat verification failure as non-fatal; callers can detect missing tables later.
            logger.debug(f"PG ensure api_keys table verification failed: {exc}")

        if errors:
            logger.info(
                "Ensured PostgreSQL api_keys tables (idempotent) with "
                f"{len(errors)} non-fatal DDL errors"
            )
        else:
            logger.info("Ensured PostgreSQL api_keys tables (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL api_keys tables: {exc}")
        return False


async def ensure_user_provider_secrets_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure user_provider_secrets table exists for PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False

        # Ensure users table exists before adding the FK-backed BYOK table.
        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"ensure_user_provider_secrets_pg: core table ensure skipped/failed: {exc}")

        for sql, params in _CREATE_USER_PROVIDER_SECRETS:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure user_provider_secrets DDL failed: {exc}")

        logger.info("Ensured PostgreSQL user_provider_secrets table (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL user_provider_secrets table: {exc}")
        return False


async def ensure_org_provider_secrets_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure org_provider_secrets table exists for PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False

        # Ensure core tables exist first (org/team scaffolding)
        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"ensure_org_provider_secrets_pg: core table ensure skipped/failed: {exc}")

        for sql, params in _CREATE_ORG_PROVIDER_SECRETS:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure org_provider_secrets DDL failed: {exc}")

        logger.info("Ensured PostgreSQL org_provider_secrets table (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL org_provider_secrets table: {exc}")
        return False


def _looks_like_postgres(obj: Any) -> bool:
    """Detect Postgres-compatible DB handles including raw asyncpg connections."""
    if isinstance(obj, DatabasePool):
        return True
    if getattr(obj, "pool", None) is not None:
        return True
    module = getattr(type(obj), "__module__", "")
    return isinstance(module, str) and module.startswith("asyncpg")


async def ensure_org_stt_settings_pg(pool: Any | None = None) -> bool:
    """Ensure org_stt_settings table exists for PostgreSQL backends."""
    try:
        db_target = pool or await get_db_pool()
        if not callable(getattr(db_target, "execute", None)):
            return False

        if _looks_like_postgres(db_target):
            try:
                await ensure_authnz_core_tables_pg(db_target)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"ensure_org_stt_settings_pg: core table ensure skipped/failed: {exc}")

        for sql, params in _CREATE_ORG_STT_SETTINGS:
            try:
                await db_target.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure org_stt_settings DDL failed: {exc}")
                return False

        logger.info("Ensured PostgreSQL org_stt_settings table (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL org_stt_settings table: {exc}")
        return False


async def ensure_identity_federation_tables_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure identity federation tables exist for PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False

        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"ensure_identity_federation_tables_pg: core table ensure skipped/failed: {exc}")

        for ddl_group in (
            _CREATE_IDENTITY_PROVIDERS,
            _CREATE_FEDERATED_IDENTITIES,
            _CREATE_FEDERATED_MANAGED_GRANTS,
        ):
            for sql, params in ddl_group:
                try:
                    await db_pool.execute(sql, *params)
                except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(f"PG ensure identity federation DDL failed: {exc}")

        logger.info("Ensured PostgreSQL identity federation tables (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL identity federation tables: {exc}")
        return False


async def ensure_secret_backend_tables_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure secret backend registry/reference tables exist for PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False

        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"ensure_secret_backend_tables_pg: core table ensure skipped/failed: {exc}")

        for ddl_group in (
            _CREATE_SECRET_BACKENDS,
            _CREATE_MANAGED_SECRET_REFS,
        ):
            for sql, params in ddl_group:
                try:
                    await db_pool.execute(sql, *params)
                except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(f"PG ensure secret backend DDL failed: {exc}")

        logger.info("Ensured PostgreSQL secret backend tables (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL secret backend tables: {exc}")
        return False


async def ensure_byok_oauth_state_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure byok_oauth_state table exists for PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False

        # Ensure users table exists before FK-backed BYOK OAuth state table.
        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"ensure_byok_oauth_state_pg: core table ensure skipped/failed: {exc}")

        for sql, params in _CREATE_BYOK_OAUTH_STATE:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure byok_oauth_state DDL failed: {exc}")

        logger.info("Ensured PostgreSQL byok_oauth_state table (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL byok_oauth_state table: {exc}")
        return False


async def ensure_llm_provider_overrides_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure llm_provider_overrides table exists for PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False

        for sql, params in _CREATE_LLM_PROVIDER_OVERRIDES:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure llm_provider_overrides DDL failed: {exc}")

        logger.info("Ensured PostgreSQL llm_provider_overrides table (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL llm_provider_overrides table: {exc}")
        return False

async def ensure_usage_tables_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure usage and LLM usage tables exist for PostgreSQL backends.

    Mirrors the SQLite migrations (usage_log/usage_daily, llm_usage_log/llm_usage_daily)
    but uses Postgres types and indexes. Intended as bootstrap guardrails when running
    against Postgres without having executed dedicated migrations.
    """
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False
        for sql, params in _CREATE_USAGE_TABLES:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure usage tables DDL failed: {exc}")
        logger.info("Ensured PostgreSQL usage tables (usage_log, usage_daily, llm_usage_log, llm_usage_daily)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL usage tables: {exc}")
        return False


async def ensure_generated_files_table_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure generated_files table exists for PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False
        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"PG ensure authnz core tables before generated_files failed: {exc}")
        for sql, params in _CREATE_GENERATED_FILES_TABLES:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure generated_files DDL failed: {exc}")
        logger.info("Ensured PostgreSQL generated_files table (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL generated_files table: {exc}")
        return False

async def ensure_data_subject_requests_table_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure data_subject_requests table exists for PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False
        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"PG ensure authnz core tables before data_subject_requests failed: {exc}")
        for sql, params in _CREATE_DATA_SUBJECT_REQUESTS_TABLES:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure data_subject_requests DDL failed: {exc}")
        logger.info("Ensured PostgreSQL data_subject_requests table (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL data_subject_requests table: {exc}")
        return False


async def ensure_admin_monitoring_tables_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure admin monitoring control-plane tables exist for PostgreSQL backends."""
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False
        try:
            await ensure_authnz_core_tables_pg(db_pool)
        except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"PG ensure authnz core tables before admin monitoring failed: {exc}")
        for sql, params in _CREATE_ADMIN_MONITORING_TABLES:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure admin monitoring DDL failed: {exc}")
        logger.info("Ensured PostgreSQL admin monitoring tables (idempotent)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL admin monitoring tables: {exc}")
        return False


async def ensure_virtual_key_counters_pg(pool: DatabasePool | None = None) -> bool:
    """Ensure virtual-key counter tables exist for PostgreSQL backends.

    These tables back cross-instance quota counters for JWTs and API keys.
    Runtime guardrails also defensively create them, but this helper centralizes
    the Postgres DDL for bootstrap flows.
    """
    try:
        db_pool = pool or await get_db_pool()
        if getattr(db_pool, "pool", None) is None:
            return False
        for sql, params in _CREATE_VK_COUNTERS:
            try:
                await db_pool.execute(sql, *params)
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"PG ensure virtual-key counters DDL failed: {exc}")
        logger.info("Ensured PostgreSQL virtual-key counters tables (vk_jwt_counters, vk_api_key_counters)")
        return True
    except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to ensure PostgreSQL virtual-key counters tables: {exc}")
        return False

# Admin operations store path

Task: TASK-13260.278.18.83.45; S04 Sources validation.

The feature-flag/incident/webhook store currently always lives beside installed code under `Databases/system_ops.json`. Multiple private profiles sharing that package would share this server-owned state. Sources needs an entitlement for one ordinary user without changing another profile.

Add one optional, trusted operator environment setting, `SYSTEM_OPS_STORE_PATH`. A nonblank value selects that JSON file; absent or blank retains the current `get_database_dir()/system_ops.json` default. Read it at the same module initialization boundary as the current store path. Operators running separate profiles should use distinct absolute paths. Do not copy or migrate a legacy store automatically, change authorization, alter record schemas, or change existing lock/atomic-write/read/error behavior.

Reuse the existing path selection and store mechanisms. Alternatives considered: changing the global utility would affect unrelated database consumers; deriving a new location from the per-user root would silently move existing Admin state; a new store/backend would duplicate the working JSON transaction layer.

Tests first: real child imports must select the configured file before performing any write. Distinct profiles with the same user id must read only their own feature flag; the first profile must remain unchanged after the second. Unit checks cover configured, absent and blank paths. Run affected Admin and Sources entitlement checks, scoped lint/security, independent review and normal hooks. Refresh only the inactive backend package before live validation; no frontend build is needed.

ADR assessment: no new ADR required. This is an optional location for the existing store, preserving the default, authority, schema and persistence protocol. ADR020 governs existing content path isolation; this change does not extend its per-user database model to Admin state.

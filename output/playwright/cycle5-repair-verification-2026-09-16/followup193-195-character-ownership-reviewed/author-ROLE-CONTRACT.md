# PostgreSQL runtime-role qualification for UAT193/194

## Established facts

The native role receipt `character-owner-role-read.json` reports both `rolsuper=true` and `rolbypassrls=true` for the same validated content database fingerprint as the character-owner receipt. The initial24 tests use the official cluster-admin fixture role. Those results establish the bypass-role behavior only.

`backends/pg_rls_policies.py:797` enables and forces RLS on `character_cards` and installs `chars_tenant_isolation`, comparing `client_id` with `app.current_user_id`. No policy modification is proposed. The new required-PG control grants a disposable NOLOGIN, NOSUPERUSER, NOBYPASSRLS role SELECT/UPDATE on character_cards, sets the second owner's scope on the same transaction, verifies the role flags, then observes zero foreign rows and zero rows affected by an attempted owner-changing UPDATE. Original ownership is unchanged after the transaction. **This is a real restricted-role SQL-policy control, not a full native application acceptance run.**

## Documented and implemented runtime contract

- `Docs/AuthNZ/AUTHNZ_DATABASE_CONFIG.md`, Content Database Modes: shared PostgreSQL content mode is documented, with configured credentials, migrations and required RLS policy validation. It does not explicitly declare superuser/BYPASSRLS connections supported or prohibited.
- `Docs/User_Guides/Server/Multi-User_Postgres_Setup.md` creates a normal `tldw_user`; its opening scope explicitly concerns AuthNZ, with per-user SQLite content unless content mode is enabled. Its superuser instruction is for provisioning, not a recommendation to run content requests as a superuser.
- `services/startup_content_backend_validation.py` → `DB_Manager.validate_postgres_content_backend` → `media_db/runtime/factory.validate_postgres_content_backend` checks backend/schema/policy readiness. No `rolsuper` or `rolbypassrls` check was found in application Python sources. Passing startup does not establish that RLS is enforced for the configured role.
- `PostgreSQLBackend._apply_scope_settings` sets user/org/team/admin GUCs. Role switching is disabled by default and requires both an explicit environment opt-in and `ScopeContext.session_role`, with optional whitelist. Ordinary `auth_deps._activate_scope_context` sets user/org/team/admin values but supplies no session_role. Setting ROLE_SWITCH/WHITELIST alone is therefore not a demonstrated ordinary-request role change.
- The nearby product precedent `Docs/ADR/035-canonical-folder-link-suppression-preserves-source-provenance.md`, 2026-08-09 amendment, explicitly requires owner predicates for Notes projection operations even under RLS because privileged service roles can bypass it. This is evidence of a defense-in-depth convention for that domain, not an automatic mandate to rewrite every Character operation.

## Recommended decision boundary

Preserve and qualify194. Do not claim ordinary-role application leakage, and do not erase the native bypass-role exposure or fixture PUT transfer. Decide whether supported content runtimes must reject/use a non-bypass role, or whether Character operations must also defend under accepted privileged service roles. The code currently accepts the connection and does not validate those privileges; the discovered docs do not conclusively settle that policy.

A configuration-only remedy needs proof of a supported normal startup and authenticated requests under the intended non-bypass runtime role. Merely changing the current profile's role flags or enabling ROLE_SWITCH is insufficient evidence: the application uses the content connection for migrations/initialization as well as requests, and the ordinary auth path does not select session_role. No live role/configuration change was performed by this agent.

UAT193's global `UNIQUE(name)` remains independent of bypass: database unique constraints apply across owners even when RLS hides the conflicting row. Under enforced RLS a second user's default can fail to insert/find instead of reusing the visible foreign default. The proposed owner/name migration is necessary in either role arrangement; factory ownership remains unchanged.

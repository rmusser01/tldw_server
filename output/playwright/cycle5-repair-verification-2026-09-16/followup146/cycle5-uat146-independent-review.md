# UAT146 independent review

## Verdict

No actionable finding in the frozen two-file correction. Approved for integration. The parent-owned repeat of normal API startup remains the native acceptance step.

## Scope and source review

Baseline HEAD c27de06742; frozen at 2026-09-16T16:23:31.384Z. Both code/test hashes match /private/tmp/cycle5-uat146-code-freeze.json; independent audit: /private/tmp/cycle5-uat146-independent-hashes.json.

The production change in `tldw_Server_API/app/core/DB_Management/media_db/runtime/factory.py:156` replaces the four obsolete media policy names with media_visibility_access, retains all four sync policies, and adds owned_clone_pending_keyword_access. This exactly matches the six policies in the full canonical `media_db/schema/features/postgres_rls.py` helper. That helper deliberately drops media_scope_* and enables/forces row-level security for media, sync_log and operationownedclonekeywords. Both fresh and existing schema initialization invoke it through ensure_postgres_post_core_structures.

The original failure is a genuine startup-validator/schema mismatch, supported by /private/tmp/cycle5-native-pg-r2-single-backend.redacted.log. The actual startup delegate reaches DB_Manager and the same runtime factory exercised here. No schema, policy expression, role/tenant predicate, backend-selection rule, or RLS setting is changed. Missing and unreadable policy results still raise RuntimeError and flow through the existing validator cleanup.

## Independent checks

Command (host access to the existing official disposable fixture):

`node /private/tmp/cycle5-postgres-fixture-run.mjs tldw_Server_API/tests/DB_Management/test_media_postgres_runtime_validation.py tldw_Server_API/tests/DB_Management/test_media_db_runtime_factory.py tldw_Server_API/tests/Services/test_startup_content_backend_validation.py -q -rs --show-capture=no`

**30 passed across 3 files, zero failed, zero skipped**, exit 0, 4.40s. Redacted log: /private/tmp/cycle5-postgres-146-independent.redacted.log. The runner activates the project venv and requires PostgreSQL; no Docker autostart, held database, or native profile was used.

The 13 new controls include:

- A separate normal Python process with curated config/cwd and no inherited pytest/test-mode flags invokes the actual startup service. It asserts both test-mode predicates are false, verifies the canonical policy names and absence of obsolete media names, and verifies ENABLE plus FORCE RLS flags for all three tables.
- Six real policy removals after actual schema bootstrap, each requiring the selected policy's exact failure. Placing the fault after bootstrap prevents automatic schema repair from hiding the negative control.
- Six unreadable-policy controls at the actual catalog query boundary using BackendDatabaseError. Each fails closed for the selected current policy. These are controlled transport faults, not native database-permission-denial experiments.

Existing runtime tests retain SQLite/default backend behavior, required-PG backend selection, schema/probe errors and cleanup. The three-file run was bounded intentionally; the author's broader 56-case/5-file result was inspected separately at /private/tmp/cycle5-postgres-146-green.redacted.log, not counted as an independent rerun. Valid pre-fix RED is retained at /private/tmp/cycle5-postgres-146-red-confirmed.redacted.log (13 failed); the obsolete media policy caused both normal startup and targeted negative assertions to fail before correction.

## Static evidence and limits

Inspected author baseline/final artifacts: Bandit zero findings/errors in both /private/tmp/cycle5-uat146-bandit-{baseline,final}.json; Ruff retains exactly I001 at line 3 and TRY203 at line 98, zero added, in /private/tmp/cycle5-uat146-ruff-{initial,final}.json. New test has no Ruff finding in that scoped output. Independent scoped git diff --check is clean. Static tools were inspected, not rerun.

This review certifies the bounded policy-name correction and its fail-closed boundary, not all tenant query semantics or an end-to-end native application launch. No repository/task changes, process/profile changes, inference, browser activity or commits were performed.

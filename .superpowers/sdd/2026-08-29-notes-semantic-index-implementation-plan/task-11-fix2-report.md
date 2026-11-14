# TASK-13134 Task 11 Second Review Fix Report

## Scope

Addressed every finding in `task-11-fix-review.md` without a schema migration, an eighth Notes semantic route, a root Jobs API, arbitrary provider/model/endpoint controls, or Task 12 visuals/evidence/conversion. All Python verification used Python 3.11.13.

## Finding Resolutions

1. **Renewal admission gap and binding health**
   - RED: renewal followed by unavailable Jobs, quota exhaustion, or a writer conflict left the updated configuration bound to the old generation and could project Ready; fresh-key recovery and exact pending-receipt replay were not covered.
   - GREEN: status now validates the complete active-generation binding (existence, active state, configuration revision, compatibility hash, model revision, and resolved dimensions). A committed renewal without an admitted rebuild reports `needs_attention/rebuild_required`, suppresses stale counts and semantic edges, and accepts a fresh current-revision rebuild. Active rebuilds retain Preparing/Updating progress, and exact receipt replay does not renew consent twice.

2. **Vector backend drift and cleanup safety**
   - RED: renewal could replace the persisted backend before cleanup, allowing maintenance to confirm absence in the wrong store.
   - GREEN: the sanitized backend key is part of the validated capability snapshot and pending capability identity. Enabled backend changes fail before receipt/configuration mutation with `notes_semantic_backend_change_requires_delete`; the public typed capability exposes `renewal_requires_delete`. Delete continues through the old persisted backend, physical absence is confirmed there, and a disabled configuration can bind the new backend only after work, generations, and obsolete-vector cleanup are empty. Both ChromaDB-to-pgvector directions are covered. Same-backend provider, model, and endpoint-origin renewal remains allowed.

3. **Probe-eligible unknown dimensions**
   - RED: unknown dimensions made capability unavailable and disabled consent, so the worker probe could never run.
   - GREEN: capability policy distinguishes `dimension_probe_required` from known unsupported dimensions. Otherwise-usable unknown dimensions remain available with null dimensions/compatibility; initial and renewed consent atomically persist pending identity and admit build/rebuild. The fixed non-user worker probe resolves dimensions by CAS before any Note content read. Unsupported pgvector probe results fail before Note transfer. Unavailable capabilities may remain unresolved without incorrectly advertising probe eligibility.

4. **No-generation recovery matrix**
   - RED: recovered initial-build failures and post-renewal admission gaps exposed Delete as the only action.
   - GREEN: an enabled configuration with no usable generation, no active run, and a usable capability offers Rebuild plus Delete across preparing/building and needs-attention recovery states. An unavailable capability remains Delete-only.

5. **Exact consent identity**
   - RED: capability disclosure omitted endpoint origin and accepted blank identity labels; pending dimensions also failed to bind vector storage identity.
   - GREEN: strict backend/client contracts require bounded nonblank provider, model, storage, and sanitized `scheme://host[:port]` endpoint identity plus the exact outbound category set. The inspector displays endpoint, provider/model, and execution/storage boundaries before consent. Secret credentials, paths, queries, and fragments are removed/rejected; custom origins, IPv6 origins, explicit default ports, and blank-label fail-closed behavior are covered.

## RED/GREEN Evidence

- Backend public capability RED: unavailable unresolved dimensions and IPv6 origin canonicalization each failed before their narrow schema/sanitizer fixes; both pass after implementation.
- Backend pending-consent RED: ChromaDB and pgvector with unresolved dimensions produced the same disclosure/capability identity; the revisions now differ.
- Frontend client RED: unavailable unresolved dimensions and explicit default-port origins were rejected; both pass after the strict validator correction.
- The original second-review RED matrix covered renewal admission failures, backend transitions, unknown-dimension consent, no-generation recovery, endpoint disclosure, and blank labels; all corresponding focused and adjacent suites are GREEN.

## Verification

- Backend focused and adjacent semantic suite: 313 passed, 2 warnings, Python 3.11.13.
- Frontend semantic client/hook/inspector suite: 76 passed.
- Adjacent Notes graph frontend suite: 126 passed.
- TypeScript: `bun run compile` passed.
- Prettier: canonical `apps/extension/.prettierrc.cjs` check passed.
- Locales: sync output retained only intended English changes; duplicate-key, coverage, and English-fallback contract checks passed.
- Ruff: all changed backend source and tests passed.
- Bandit: zero findings in changed production backend scope; report at `/tmp/bandit_TASK-13134_task11_fix2.json`.
- `git diff --check`: passed.

## Commit

`fix: complete Notes semantic consent recovery (TASK-13134)` (this commit)

## Concerns

- No live PostgreSQL/pgvector service test was run for this follow-up. The conservative no-migration cleanup gate and portable CAS SQL are exercised through SQLite store/integration tests; both logical backend directions and physical old-store absence are covered with backend fakes.
- `bun run check:i18n:glossary` is not runnable because the repository references a missing `apps/extension/scripts/i18n-glossary.json`. Duplicate-key and coverage checks pass; unrelated glossary infrastructure was not changed.

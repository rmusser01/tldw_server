# B03 dictionary response timestamp normalization — 2026-10-08

Associated task:TASK-13260.278.18.83.45 (root-owned; new live observation recorded before edits). Reviewed earlier units remain frozen. Parent owns task updates, API/frontend replacement and live validation.

## Attributed source cause

New dictionary success followed by Updated `6 hours ago` is supported by the current path: dictionary SQLite columns default to UTC-naive `CURRENT_TIMESTAMP`; create/list/get/edit construct `ChatDictionaryResponse` directly from those values; its datetime fields serialize without an offset. Shared dictionary `formatRelativeTimestamp` uses `new Date(input)`, so the LA browser interprets timezone-less ISO text as local time (7 hours ahead during October DST), then takes `Math.abs` and floors hours. Seven hours minus request latency therefore displays6 hours ago. Tooltip/sorter read the same ambiguous timestamp. This is not evidence of an operator clock defect.

Dictionary entry/stat paths already use shared `parse_iso_utc` through endpoint coercion. Normalize only `ChatDictionaryResponse.created_at/updated_at` after standard Pydantic datetime validation using that existing helper. Its derived with-entries/list responses inherit the correction; storage/auth/rules and every unrelated timestamp stay unchanged. No frontend production change is needed.

ADR check:ADR required:no. Searched `Docs/ADR/README.md`; this restores the established aware-UTC response convention with existing helper, without a new API/security/storage/date policy. ADR-020 still governs per-user storage and ADR-006 scoped static scans.

## Stage 1: Demonstrate the serializer/consumer cause
**Goal**: Reproduce with current dictionary response and real endpoint/service DB paths; show LA consumer effect deterministically.
**Success Criteria**: Required dictionary timestamps fail aware-UTC serialization on unchanged source; missing/invalid timestamp rejection and explicit-offset controls remain.
**Tests**: Naive SQLite/ISO/datetime, Zulu/offset/epoch representations; real dictionary create/list/detail/update using official ephemeral DB abstraction; read-only existing frontend helper proof with fixed UTC instant/process-local TZ.
**Status**: Complete

## Stage 2: Normalize only dictionary response timestamps
**Goal**: Reuse canonical `parse_iso_utc` in the response schema after datetime validation.
**Success Criteria**: Serializer emits unambiguous UTC strings and preserves timestamp instants/validation; all four dictionary consumers receive the same correction.
**Tests**: Fresh focused regression, affected frontend utility checks in installed UI/Web environments, supported Python3.12 normal plugins/300case600outer caps; max3 bounded attempts per issue.
**Status**: Complete

## Stage 3: Qualify and freeze
**Goal**: Compile/Ruff and baseline-current static Bandit, connected-source review and root handoff.
**Success Criteria**: No new findings; exact scope/hash/results and production freeze for parent runtime validation.
**Tests**: Touched Python only; existing primary Python3.11/Bandit verified AST-only scanner, whitespace/source review. No unchanged unrelated suites.
**Status**: In Progress

## Boundaries

No live DB/auth/browser/runtime/build/index/parent task/other-unit edits, environment installs/model/network/proof/logs/Git publication. Tests use existing official DB abstraction in tmp_path; no operator SQL or custom DB provisioning. Process-local test TZ is not a user/browser/environment-file change. Keep all unrelated datetime behavior unchanged.

## Attributed proof and causal red

- Existing actual dictionary frontend helper, installed Node26, fixed UTC now15:20:03Z and process-local America/Los_Angeles: timezone-less15:20:00 parses22:20:00Z, yields `6 hours ago`; same timestamp withZ parses15:20:00Z, yields `just now`. No live clock/profile assumption or source change involved. Node reported only inherited module-type warning; no package/config change needed.
- Supported Python3.12.13/pytest9.1.1, normal plugins300case/600outer:9 failed,7 passed,31 warnings,no skips/no fixture errors,13.96s pytest/18.395s outer,natural1. Real create/list/detail/update naive serialization failures match source cause; wrong/missing timestamps still reject and Zulu/epoch controls pass. Offset representations remain the same instant but need canonicalUTC output.
- Baseline schema Ruff0; existing verified primary Python3.11.13/Bandit1.9.3 AST-only scan0findings/0errors. New test Ruff0. Source ruling/red were sent to root before production edit.

## Final scoped qualification and freeze

- Fresh16 regressions passed,no skips/no fixture errors,31 warnings,7.82s pytest/11.484s outer,natural0. Six directly affected existing dictionary metadata/entry/locking controls passed,50 unrelated cases excluded,34 warnings,8.11s pytest/11.709s outer,natural0. Supported Python3.12/configured plugins/caps unchanged.
- Actual corrected production schema JSON passed in memory to the actual existing dictionary frontend helper under LA: serialized `2026-10-08T15:20:00Z`, parsed sameUTC instant, relative `just now`; natural0. No proof file created.
- Existing dictionary list utility suite passed12 cases each through installed UI and WebUI configurations under LA; outer1.011s/0.956s,natural0. Frontend production is unchanged, so no unrelated TypeScript/build work needed.
- Compile2 passed; Ruff baseline/current0 findings; verified existing primary Bandit AST-only baseline/current0 findings/0scan errors. Whitespace checks clean.
- Production schema is frozen: existing helper import +6-line after-validator limited to dictionary response created_at/updated_at. Timestamp instants, standard Pydantic validation, all storage/rules/ownership/security and unrelated timestamps remain unchanged.
- Connected-source self-review found no substantiated Critical/Important/Minor finding. Stage3 remains In Progress solely for root independent review/actual B03 reload/live validation/final task evidence. Source tests and deterministic timezone proof do not establish the loaded live API/UI or other auth/DB profiles. This backend correction requires API replacement; no frontend build is required by this unit.
- No parent task, earlier units, runtime/browser/liveDB/auth/build/index/install/network/proof/log/Git publication changes.

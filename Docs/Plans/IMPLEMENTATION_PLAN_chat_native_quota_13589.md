# Retained native chat quota authority — TASK-13589

Continue the approved H2 design §8 and Task 2.2. Public native-fork capability remains disabled. Native retained bytes use the existing AuthNZ allowance, not a second allowance.

ADR assessment: ADR-049/050 govern native ownership/lifecycle, ADR-064 governs per-user limits and existing shared pools, ADR-059 governs task editing. Proposed ADR-070 records the durable legacy/native split and same-key authority; existing accepted rationale is unchanged.

## Stage 1: Scope and authority contract
**Goal**: Freeze every accounting reader, bootstrap path and cross-store boundary before changes.
**Success Criteria**: Identity freezes verified users.uuid and exact global/workspace scope from the authenticated native owner; applicable groups are checked against active membership. One immutable candidate reservation records user, owner namespace, scope, byte count and active/released state. Exact released keys never allocate again. Existing user/org/team lock order is reused, including stable organization/team identities before optional pool rows. No missing ledger or failed read becomes zero usage.
**Tests**: Real AuthNZ runtime SQLite plus the official PostgreSQL fixture, fresh/upgrade/repeated schema setup, same-key replay/conflict and membership refusals.
**Status**: Complete

## Stage 2: Atomic quota admission
**Goal**: Reserve retained bytes once under existing quota-owner transactions.
**Success Criteria**: Read current uncached allowances and legacy counters; combine fresh native sums before insert. Keep counting when enforcement is off, preserve null/unlimited and existing shared pools. No native call to update_usage; cancellation rolls back. Serialized native callers cannot overspend the same remaining allowance.
**Tests**: Real concurrent reservations; changed allowance after cache prime; zero/off/null allowances; wrong scope, shared-pool races, lost response replay and released-key refusal.
**Status**: Complete

## Stage 3: Existing readers and recalculation
**Goal**: All existing quota/admission views report L + fresh N exactly once.
**Success Criteria**: User calculation/quota/cache/status/breakdown/list/generated usage and team/org repository status use the split. Filesystem recalculation excludes managed chat_assets and persists/caches only L. Public serializers expose native and total usage without mislabeling generated-file categories; fallback readers do not claim L is complete after a ledger failure. Normal API artifacts stay current if schemas change. Ordinary generated-file registration, deletion, restoration and shared counter reporting reuse the existing transaction-bound service view so a required native read cannot leave a partial accounting tail; physical voice removal follows commit. Single/bulk/restore lock all selected quota identities in global user/org/team order, reread only the originally authorized IDs after waits and account the exact SQL transition IDs.
**Tests**: Cache-prime → reserve → publish → recalculate → release sequence; user/team/org totals and counters; current profile/admin/storage views; existing non-native registration regressions.
**Status**: In Progress

## Stage 4: Cross-store reserve and cleanup recovery
**Goal**: Durable ChaCha intent precedes AuthNZ reservation; byte admission follows confirmed same-key authority.
**Success Criteria**: AuthNZ calls occur outside every chat transaction; durable reserve/resolve/release facades refuse caller-bound uncommitted AuthNZ views. Original candidate/generation/owner is rechecked after awaits. Lost reservation response is resolved under the original key. Adopted/soft-deleted assets remain charged. Guarded writer drain and verified removal precede durable release intent and idempotent quota release; unknown authority keeps the intent unresolved conservatively.
**Tests**: Real authority with native I/O crash windows, stale workers, missing/unknown outcome, reserve before acknowledgement and unlink before release acknowledgement; existing FD and reclamation controls remain intact.
**Status**: Complete

## Stage 5: Review and qualification
**Goal**: Finish this accounting unit with bounded evidence.
**Success Criteria**: Two independent source reviews clear; supported-runtime affected tests, normal lint, Bandit, API artifact checks where applicable and exact-owned hooks pass before commit. Keep H2 projection/read/send/persona/surface/Sync/physical-AT/study gates open. Record unavailable official PostgreSQL instead of claiming qualification.
**Tests**: Connected storage/native/user profile/AuthNZ regressions; baseline deltas for inherited failures; no test suppression or runtime shim.
**Status**: Complete

Scope map: StorageQuotaService; AuthNZ storage_quotas_repo and quota_resolver; DB_Management reservation queries; SQLite migrations and packaged AuthNZ schema; PostgreSQL additive bootstrap and packaged schema; UserProfiles live/fallback quota reader; existing admin/auth/storage serializers and idempotent user-row overlays; Billing active-member summaries with legacy-only cache and fresh native reads; native asset store and NativeChatAssetIO orchestration; real-runtime integration and native I/O tests; generated API contracts if public schema changes. Reuse DatabasePool.transaction, existing transaction-bound pool and stable VN quota locks. No new dependencies, public routes, options, native allowance or global legacy writer lock.

Known inherited limit: a previously admitted non-native upload can race this admission under its existing check-then-write semantics. Logical retained usage is not temporary peak disk space. Missing reservation during cleanup is unknown, not authorization to free a charge; retry reconciliation may retain a conservative unresolved intent.

Earlier qualification: actual SQLite identity/replay/races/cache-recalculation/lost-ack/lease/receipt/membership/policy/cancellation controls pass in the focused 88-pass, 13-official-PostgreSQL-skip run. Independent review then found seven ordinary lifecycle/shared-counter partial commits; real RED reproduced all seven, and their transaction-bound correction is under qualification. No broader readiness claim.

Final accounting foundation qualification: full26-file supported Python3.12 connected run444 passed/258 official/backend skips; final bound-SQL/admin/bridge controls22 passed/3 official PostgreSQL skips. Ruff baseline21/current21 and Bandit baseline11/current11 introduce zero findings; inherited Bandit token-type labels are unchanged. Both normal client configurations exit0. Two final independent source reviews are bounded CLEAR. Canonical generation loaded this checkout but retains2114 paths/3276 schemas and the prior fingerprint because the existing storage router is omitted from full production registration. TASK-13590 tracks that separate discovered root; Stage3/public storage reach stays open until its registration/contract qualification. No current public H2, physical PostgreSQL/AT, installed-extension or study claim.

Normal exact-owned41-file hooks and supported3.12 compileall exit0 without bypass. Stage3 remains the single open bounded gate for TASK-13590.

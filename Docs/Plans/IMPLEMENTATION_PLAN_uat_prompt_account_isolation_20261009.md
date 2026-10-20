# UAT prompt account isolation — 2026-10-09

Associated task: TASK-13260.278.18.83.45 (root recorded this distinct unit before edits).

## Scope and design
Canonical Prompt Studio rejects foreign account access. Browser-wide local records and query keys retain private prompt data after account changes. Reuse the existing opaque persistence owner identity for local prompt ownership. Authenticated new saves and canonical server copies are private to that owner; unowned/unlinked legacy and offline records remain local. Ambiguous unowned server-linked records remain stored and hidden, never automatically claimed or deleted. Guard local reads/mutations, link/sync settlement, project default IDs, query caches and editor lifetimes. Preserve recipe uncertainty/receipt behavior and existing backend authorization.

ADR assessment: no new ADR required. ADR-055 (Docs/ADR/055-prompt-studio-db-single-implementation.md) governs backend ownership; this fixes browser copies to honor that boundary without changing the backend or introducing new durable authority. Existing Notes authority-state/storage and recipe ownership/request snapshots provide the client patterns.

## Stage 1: Demonstrate and specify the leak
**Goal**: Trace local and transport consumers and add causal account-switch/storage regressions.
**Success Criteria**: Foreign owned and ambiguous linked rows are distinguished from shared local legacy data; current source reproduces storage and mounted cache leakage.
**Tests**: Real PageAssistDatabase authorization with the existing isolated storage adapter; mounted cached-inventory account switch; real request transport rejection before foreign dispatch.
**Status**: Complete

## Stage 2: Restore the complete ownership boundary
**Goal**: Minimal owner-aware local storage, sync/link/project settings, and inventory/editor authority lifetimes.
**Success Criteria**: Foreign reads/writes/exports/trash/linking are denied; new authenticated saves are owned; captured A content cannot dispatch under B; canonical pulls partition duplicate numeric server IDs; offline legacy and recipe durability remain intact.
**Tests**: Added causal regressions plus affected existing helper/sync/recipe/editor and inventory suites in normal UI and Web configurations; touched lint and configured types. TypeScript-only paths are Bandit N/A; backend expected-user guards receive baseline/current scoped Bandit.
**Status**: In Progress

## Stage 3: Independent review and actual browser qualification
**Goal**: Freeze and independently review changed source, then root validates Bob/Alice with the actual browser and canonical Prompt Studio.
**Success Criteria**: Independent review clear; root verifies foreign content disappears, Bob content survives, and valid owner operations/reloads work. Broader 43 UAT and extension/Postgres acceptance remain separate.
**Tests**: Root-owned live qualification; no author runtime/browser action.
**Status**: Not Started

## Limits
Installed Dexie is available; fake-indexeddb is absent. No dependency installation or fabricated IndexedDB implementation. Adapter tests retain real application authorization; actual IndexedDB is a root browser gate. No proof/baseline/log files or old controls replayed. All unrelated WIP is preserved.

## Current verification and reassessment

The scope includes an optional expected-user assertion on Prompt Library and Prompt Studio project/prompt routers, reusing the existing dependency and preserving absent-header compatibility. Backend scope was recorded by root before edits. Existing require-mode recipe guards and the separate createChat/C08 changes are preserved.

New storage, sync, cookie-request, migration, deferred export/import, configuration-read failure and readonly legacy controls have causal red/green results. The readonly slice never probes ownership and contains only unowned records without any server/library/Studio/parent link. Authenticated inventory remains unavailable until ownership resolves. A distinct settled, unconfigured localOnlyReady state allows ordinary Settings preferences; it does not authorize private inventory or unknown configured mutations.

The offline mounted Settings control reached three failed checks: the first demonstrated the permanent source Skeleton; after fixing that source, an incomplete authored defaults fixture failed real Form validation, then a selector matched both legitimate Execute/Eval model fields. Those two are fixture/operator failures, not demonstrated application Save failures. Reassessment found the existing real-Form account-switch test and house all-element selectors: keep complete defaults and select both model fields unambiguously before pressing Save. Root authorized one final fixture-only check after this reassessment, retaining the original owner-null Save assertion and no-owner-probe assertion. No source attempt cap is reset.

An earlier broad 600-second test run was unqualified. Subsequent affected groups completed naturally but exposed fixture ownership, resolver-sequence and query-key mismatches; only failed groups are corrected, with original recipe/receipt/unknown/error controls retained. Large combined failure displays were truncated and are not claimed as complete failure-body review. Current combined types, independent final review and actual IndexedDB/browser qualification remain root-owned and open.

The recipe compatibility fixture reassessment identified two distinct authored setup mistakes after the ownership correction: the initiating manual-credential row was stamped with a different baseline owner, and the final A view was rendered while canonical configuration remained B. The correction captures the actual initial owner/configuration, arranges a separate B-owned update row with a distinct local ID, and restores A's actual configuration before inspecting A. Stale-A/no-mutation, exact retained A data, B dispatch/receipt/uncertainty and disabled retry assertions remain. Only the six remaining update failures are selected for the final fixture check; the six create passes are excluded.

Independent review found a distinct-store same-ID Firefox mirror gap. Causal controls show foreign/ambiguous/shared rows overwritten or deleted by a legitimate Dexie-owned mutation. Mirroring now requires the captured-owner predicate and matching stored owner identity/class; current owner must still match before Dexie mutations. The old rollback mocks are extended with actual row reads/ownership and isolated unknown-marker cleanup, preserving every original assertion. No browser/IndexedDB acceptance is implied by adapter tests.

## Final source freeze and bounded qualification

The local unit is frozen across 35 production paths. The final contract additions include the raw Firefox Prompt owner metadata type and captured-owner native rereads/bulk trash continuations. Five new isolated captured-helper cases reproduced three causal failures and two matching/shared positives; the corrected helper and affected mirror module passed 30 cases in both normal UI and Web configurations. The original 25 mirror bodies were preserved. An automatic review initially rejected an append to that existing mirror file, confusing it with the previously rejected recipe fixture rewrite. Root reassessed the distinct source/control and recorded a safer NEW isolated regression module before its creation; neither original rejected path was bypassed.

The six remaining recipe compatibility failures were resolved by separating storage privacy from transport recovery. A's existing owned row remains byte-equal and B's local push is denied. Real createPrompt receives an independently authored literal public v2 body with the same receipt key under B; unknown-owner quarantine still blocks the actual request, while scoped A uncertainty and delivered acknowledgements permit legitimate B dispatch. The stale project-less checks retain zero non-auth requests and exact original synced-row immutability. There is no same-ID ownership transfer, deletion or private-body fallback.

The normal Web dispatch file first passed 63 cases and exposed two unchanged transient-storage ambiguity guard failures. A typed PromptOwnershipError now identifies only real ownership refusals; genuine local DB I/O retains the existing transient fallback and durable lock reporting. Only the two failures were rerun in Web and passed. UI's final eight affected cases passed with 57 excluded. All 65 distinct dispatch cases are qualified through split runs; no single complete 65-case green run is claimed. Earlier 600-second broad-run and truncated display limitations remain.

Final new Web controls qualify 93 distinct cases across 17 modules by split checks, including the five captured-helper additions; no single 93-case run is claimed. The final unreadable-local-storage control passed under the actual Web adapter; the explicit local-area positive and prior passes were excluded from unchanged replay. The offline Settings final fixture-only check passed with the exact owner-null Save/no-owner-probe assertions retained.

The backend expected-user routes and their actual guard/store regressions plus affected structured-route and principal suites passed 65 cases (625 warnings; 10.75 pytest/13.224 outer), within the declared 600/300 limits. Python 3.12 final five-file compile passed. Scoped baseline/current Bandit found zero findings/errors in four production paths and the new test. Six narrow pytest assertion comments use an explanatory delimiter accepted by Bandit; final formatting/comment edits preserve the executed test AST. Ruff retains three exact inherited diagnostics and introduces none; the new test format check passes without cache. Production ESLint reports zero errors/new warnings; authored fixture tests retain explicit-any warnings, so warning-free lint is not claimed. One fixture unused-expression error was corrected with equivalent if/else, preserving all assertions.

Root's first normal configured UI and Web semantic checks exposed six real type-contract errors, now corrected: the raw fallback Prompt type carries optional ownerId; only the Prompt restore receives captured ownerScope, while shared chat/nickname/model import calls retain their original empty options. These changes preserve transaction/auth boundaries. Root owns the configured type rerun, combined builds and actual IndexedDB/browser qualification. The independent review remains a root acceptance gate. The earlier adjacent three history failures are unqualified and are not attributed to this Prompt unit. Broader Studio/template, extension, PostgreSQL and all-43 UAT gates remain separate and open.

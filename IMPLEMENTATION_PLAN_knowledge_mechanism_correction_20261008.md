# Knowledge mechanism correction plan

Spec: Docs/Design/2026-10-08-knowledge-mechanism-correction.md
Task: TASK-13534. Base:97ea9cd5fa7e3a61ee4e7c56f3d9643b7311d511.

## Global constraints

Use existing mechanisms and minimum supported corrections. No assumption-based deletion, new acquisition subsystem, dependency declaration, gate modification or assertion weakening. Preserve owner/data/security/version/receipt contracts. Sequential implementers; independent task reviews and whole-branch review. Failed receipts remain historical; new successes never relabel them. ADR assessment: no new rule; existing026/031/034/042/059/065/066 apply, historical065 deviation recorded.

## Stage 1: Owning test and quality contracts
**Goal**: Resolve the five storage cases, deterministic lazy export readiness and CSV download coverage, scoped lint errors/introduced findings and correction-file formatting.
**Success Criteria**: Owning storage tests pass default and native configs with all assertions retained; lazy readiness survives controlled delayed real import without timeout changes; both export formats validated. All189 workstream client files have zero scoped errors, introduced warnings corrected; remaining broader warnings have exact historical comparison and owning records, not dismissal.
**Tests**: Storage/quota/split/saved-view neighbors; literature export/affected Research suite with existing house readiness; canonical ESLint/Prettier/Ruff/Black.
**Status**: Complete (scoped corrections verified; historical CSV cause, whole-file formatting and wider warning debt remain explicitly qualified in Task1 report)

## Stage 2: Notes content and durable retry correctness
**Goal**: Fix conflict-base advancement and uncertain create retry loss in Quick Notes and Knowledge Export using existing Notes draft/pending operation persistence.
**Success Criteria**: Second save after409 cannot overwrite remote content without explicit reload/merge; committed/lost-response collapse orClose/remount reuses exact key/body and canonical identity, including owner and newer-edit boundaries. No second draft store.
**Tests**: QuickNotes save-ownership plus ExportDialog a11y/retry and existing Notes offline draft/lifecycle suites; red reproductions for conflicts and unmount.
**Status**: Complete (conflict base and owner-scoped durable pending operations verified default/native; controlled cross-context/extension-adapter evidence and residual live-runtime limits in Task2 report)

Task2 review fix round1: all three Important findings corrected through captured house service-authority identity, actual persisted Workspace pointer readback, narrow canonical update/explicit migrated-create recovery, and malformed ordinary-map recovery. Tombstones remain authoritative; no discarded Workspace snapshots recreated. Normal migrated pointerless Save refuses unresolved operations; explicit previous-save retry uses a fresh request scope and leaves unrelated live drafts unchanged. Focused RED→GREEN, affected/default/native/extension and quality evidence: `.superpowers/sdd/IMPLEMENTATION_PLAN_knowledge_mechanism_correction_20261008/task-2-fix1-report.md`. Scoped re-review pending; existing ADR031/034/065 govern.

## Stage 3: Restore capture evidence from canonical sources
**Goal**: Repair migrated source restore when local capture checkpoints disappear using existing owned clip/version readbacks.
**Success Criteria**: Accepted source retains exact validated pin; ambiguous/missing known capture evidence surfaces unavailable instead of current-source downgrade. Ordinary source handling remains compatible; latest active alone cannot identify accepted evidence.
**Tests**: Existing research-web-capture restore/readback/current-material and ChatPane captured-source guards; checkpoint deletion and version metadata mutation cases.
**Status**: Not Started

## Stage 4: Reuse shared backend mechanisms
**Goal**: Restore governed canonical scraper backend selection, bound preflight, consolidate HTTPX stream overlap, remove unused provenance writer and complete strict capabilities discovery.
**Success Criteria**: Shared curl/HTTPX meet public profile before backend overrides/refusals are removed. Real optional curl validation covers env isolation, DNS pinning, identity/compression/size bounds, cookies, redirect and cleanup. Lifecycle tests use production compound plan; discovery exposes existing strict contract.
**Tests**: Public capture/preflight/central HTTP/egress tests, actual curl controlled transport, Notes/Sync lifecycle/capabilities and relevant owner/version API tests. Bandit on touched Python.
**Status**: Not Started

## Stage 5: Integrated verification and accurate closeout
**Goal**: Verify the integrated correction, publish reuse evidence and update all associated Backlog tasks including13514.
**Success Criteria**: Affected backend/client tests, security, types, builds and relevant CDP checks pass; independent broad review clean. Exact commands/head/hash evidence and unresolved hardware/live-service limits recorded. Concrete corrective PR against dev follows repository gates.
**Tests**: Existing affected suite scopes and canonical required checks, no custom dependency aliases; actual merge procedure evidence if authorized landing proceeds.
**Status**: Not Started

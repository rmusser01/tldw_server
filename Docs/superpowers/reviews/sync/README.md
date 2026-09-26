# Review ledger — `tldw_Server_API/app/core/Sync/` (slug: `sync`)

Audit date: 2026-09-21. Module size: 43,905 LOC across 65 `.py` files (see
`2026-09-21-stage1-source-inventory.txt`). Read-only maintainability + correctness audit;
no source file under `tldw_Server_API/` was modified.

## Four rules for this ledger

- Write findings before suggested actions in every stage file.
- Label uncertain items as probable risks or assumptions instead of confirmed defects.
- Keep later-stage summaries pointed back to the stage files; do not replace the per-stage record with
  a rolling summary.
- Keep the stage files as the durable review ledger for this audit.

## Stage order

| Stage | File | Covers |
| --- | --- | --- |
| 1 | `2026-09-21-stage1-architecture-and-inventory.md` | Module shape, churn, layering, the `SyncV2Service`/`SyncV2Store` pair, test reachability |
| 2 | `2026-09-21-stage2-push-pull-conflict-core.md` | `push` / `pull` / conflict resolution — cursor and watermark correctness |
| 3 | `2026-09-21-stage3-domain-family-boundaries.md` | Domain adapters, materializers, bootstrappers, contracts — the sibling-family duplication |
| 4 | `2026-09-21-stage4-blob-and-retention-boundaries.md` | Blob upload/download lifecycle and retention/GC — the data-source boundary |

Stages 5 and 6 of the canonical RAG arc are collapsed. Sync has no separate "composition" layer
worth its own stage — composition lives entirely in `v2/factory.py` (570 LOC) and is covered in
stage 1. Test-gap analysis is recorded in each stage's `## Tests Reviewed` section and consolidated
in stage 1's finding `sync-11`, rather than padded into a sixth file.

## Canonical links

Binding ADRs checked before asserting anything in this ledger:

- `Docs/ADR/031-notes-capability-sync-domains.md` — notes capability sync domains
- `Docs/ADR/034-durable-server-origin-sync-mutation-batches.md` — durable server-origin batches,
  the append-authority/projection-fence rules, and the conflict-as-ordering-blocker amendment
- `Docs/ADR/035-canonical-folder-link-suppression-preserves-source-provenance.md`
- `Docs/ADR/037-canonical-notes-link-sync-and-derived-graph-projections.md`
- `Docs/ADR/038-canonical-notes-attachment-registry-and-blob-lifecycle.md`
- `Docs/ADR/039-canonical-notes-task-sync-and-derived-checklist-projections.md`
- `Docs/ADR/040-synchronized-moodboards-and-studio-authority.md`
- `Docs/Architecture.md` — operative layering document

No finding in this ledger contradicts a binding ADR. Finding `sync-1` is a defect *against ADR-034's
own conflict-as-ordering-blocker rule*, not against it: ADR-034 requires an unresolved projection
conflict to block later history, and the versioned pull path implements the read-side of that rule
while the legacy pull path does not.

## Machine-generated sidecars

- `2026-09-21-stage1-source-inventory.txt` — every `.py` file with LOC
- `2026-09-21-stage1-churn-baseline.txt` — 12-month commit counts per file
- `2026-09-21-stage1-test-inventory.txt` — the 100 test files reaching `core.Sync` by import-grep

## Finding index

| ID | Axis | Severity | Stage |
| --- | --- | --- | --- |
| `sync-1` | correctness | High | 2 |
| `sync-2` | correctness | High | 4 |
| `sync-3` | duplication | Medium | 3 |
| `sync-4` | encapsulation | Medium | 1 |
| `sync-5` | duplication | Medium | 1 |
| `sync-6` | duplication | Medium | 3 |
| `sync-7` | duplication | Medium | 3 |
| `sync-8` | efficiency | Medium | 2 |
| `sync-9` | efficiency | Medium | 4 |
| `sync-10` | correctness | Medium | 2 |
| `sync-11` | duplication | Medium | 1 |
| `sync-12` | duplication | Low | 2 |
| `sync-13` | duplication | Low | 2 |
| `sync-14` | duplication | Low | 1 |

# ADR Task Reference Reconciliation - 2026-09-25

**Related task:** TASK-13373
**Baseline:** `origin/dev` at `3f909e133b`
**Scope:** Task identities referenced by the ADR backfill work stream. This is reference metadata, not a change to an accepted architecture decision.

## Verified Renames

Git commit `3a660d4f1da59e3752f9824b6b97eb476660ea67` (`chore: make active backlog task ids unique`, 2026-07-05) records each rename below with 99% similarity. The linked task files exist at the baseline commit and carry the destination ID and matching title.

| Historical identity | Current task record |
| --- | --- |
| TASK-519 | [TASK-12537: Confirm ACP AuthNZ RBAC ADR candidates for backfill](https://github.com/rmusser01/tldw_server/blob/3f909e133b22cd966b8f85a636f7fae2febd8bd1/backlog/tasks/task-12537%20-%20Confirm-ACP-AuthNZ-RBAC-ADR-candidates-for-backfill.md) |
| TASK-520 | [TASK-12541: Backfill ACP persistence and scoped RBAC ADRs](https://github.com/rmusser01/tldw_server/blob/3f909e133b22cd966b8f85a636f7fae2febd8bd1/backlog/tasks/task-12541%20-%20Backfill-ACP-persistence-and-scoped-RBAC-ADRs.md) |
| TASK-2234 | [TASK-12671: Backfill Resource Governance endpoint policy ADR](https://github.com/rmusser01/tldw_server/blob/3f909e133b22cd966b8f85a636f7fae2febd8bd1/backlog/tasks/task-12671%20-%20Backfill-Resource-Governance-endpoint-policy-ADR.md) |
| TASK-2261 | [TASK-12684: Confirm Embeddings ADR candidate for backfill](https://github.com/rmusser01/tldw_server/blob/3f909e133b22cd966b8f85a636f7fae2febd8bd1/backlog/tasks/task-12684%20-%20Confirm-Embeddings-ADR-candidate-for-backfill.md) |
| TASK-2262 | [TASK-12685: Backfill Embeddings API and media pipeline ADR](https://github.com/rmusser01/tldw_server/blob/3f909e133b22cd966b8f85a636f7fae2febd8bd1/backlog/tasks/task-12685%20-%20Backfill-Embeddings-API-and-media-pipeline-ADR.md) |
| TASK-2272 | [TASK-12693: Confirm Data Tables ADR candidate for backfill](https://github.com/rmusser01/tldw_server/blob/3f909e133b22cd966b8f85a636f7fae2febd8bd1/backlog/tasks/task-12693%20-%20Confirm-Data-Tables-ADR-candidate-for-backfill.md) |
| TASK-2275 | [TASK-12694: Confirm DeepSeek OCR ADR candidate for backfill](https://github.com/rmusser01/tldw_server/blob/3f909e133b22cd966b8f85a636f7fae2febd8bd1/backlog/tasks/task-12694%20-%20Confirm-DeepSeek-OCR-ADR-candidate-for-backfill.md) |
| TASK-2276 | [TASK-12696: Backfill DeepSeek OCR backend ADR](https://github.com/rmusser01/tldw_server/blob/3f909e133b22cd966b8f85a636f7fae2febd8bd1/backlog/tasks/task-12696%20-%20Backfill-DeepSeek-OCR-backend-ADR.md) |
| TASK-2312 | [TASK-12705: Audit Security secrets and serialization ADR candidate](https://github.com/rmusser01/tldw_server/blob/3f909e133b22cd966b8f85a636f7fae2febd8bd1/backlog/tasks/task-12705%20-%20Audit-Security-secrets-and-serialization-ADR-candidate.md) |
| TASK-2313 | [TASK-12706: Backfill Security crypto envelope ADR](https://github.com/rmusser01/tldw_server/blob/3f909e133b22cd966b8f85a636f7fae2febd8bd1/backlog/tasks/task-12706%20-%20Backfill-Security-crypto-envelope-ADR.md) |

The active decision inventory uses these current identities. Accepted ADRs preserve their original task references and rationale, with a dated metadata note pointing here. Historical specs, plans, audits, and task notes may still contain the old numbers; use this table rather than opening an unrelated task that now occupies that number.

## Unresolved Historical Identities

No verified current-task mapping was established for the earlier ADR-workflow identities TASK-506 through TASK-512, TASK-514 through TASK-518, or TASK-3701. Most of these numbers now identify unrelated work; TASK-3701 has no task file in the current task directory. Do not infer a replacement from a matching number or similar title. Their ADR references remain historical provenance, not links to current Backlog work.

The earlier Evaluations audit's TASK-517/TASK-518 references are in this category. Its evidence and resulting ADR-012 through ADR-015 remain useful; their task identity is not restored or renumbered by this work.

## Ambiguous Current IDs

The current task directory also has more than one file for each of TASK-12158, TASK-13008, TASK-13192, TASK-13197, and TASK-13208, all of which appear in newer ADRs. Use the full title and path when identifying those records. This audit does not choose a new ID or rename either record; tracker-wide collision repair is separate from the verified historical renames above.

## Verification Boundary

Verify a mapping against both Git rename evidence and the current file's frontmatter/title. Recovered identities need a separate provenance-backed update; do not silently turn an unresolved historical number into a current task claim.

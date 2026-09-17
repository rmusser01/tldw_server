# H1 selected history and fork ownership: verification record

Status: **In progress. H1 is not qualified for release.** This record separates reviewed implementation evidence from the real-browser and owner checks that remain. Counts from different runs overlap and must not be added together.

## Scope and source

- Design: [H1 selected history and fork ownership](../Design/2026-09-16-chatbook-h1-history-selection-design.md).
- Execution plan: [H1 implementation plan](../../IMPLEMENTATION_PLAN_chatbook_h1_history_selection.md).
- Backlog: TASK-13261.1, In Progress.
- Branch: `codex/chatbook-h1-history-selection`; isolated worktree `.worktrees/chatbook-h1-history-design`.
- Implementation base: `f00e12a5aa`, based on server dev `59049e094e0845a4611ea725ae19b7c1754ea709`.
- Most recent source refresh: Chatbook dev `1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6`; server dev unchanged. [Source delta review](../Design/2026-09-17-chatbook-chat-parity-source-refresh.md).
- Unrelated main-checkout UAT changes, processes and data are outside this work.

H1 is one part of the broader parity effort. Atomic native forks/receipts (H2), complete local and temporary rich-state recovery (H3), synchronization enrollment/transfer (H4), independent model connection/cold WebUI installation (F02) and browser execution hosts (D3) are separate work. `tldw-agent` remains an external OS/tool execution agent acting for the server; it is not a Chatbook runtime or chat storage/sync layer.

## Reviewed implementation checkpoints

| Task | Reviewed result | Evidence boundary |
|---|---|---|
| 1.1 — selected-path contract | Commits `28c7d1904e`, `2fed0871c4`, `77dde3d2b4`; independent review and two fix reviews resolved identified P1/P2. | 18 focused Vitest and 17 Python tests; focused types, Ruff, compile and Bandit. Owner-specific fork exclusions remain Task 4.1. |
| 2.1 — native snapshots and legacy CAS | Commits `8a293d1ef3`, `c11a8bd9e9`; native review and fix review clean. | Final affected SQLite/PostgreSQL scope: 32 passed, including 20,001-row review and accepted projection reopen. This is owner evidence, not browser evidence. |
| 2.2 — native admission and settlement | Commits `3f8144d505`, `db1c6feddf`; all six P2 findings addressed by independent fix review. | Final owner run: 67 passed, one intentionally SQLite-only parametrization skipped; endpoint/context: 69 passed; actual saved-character factory compatibility: 1 passed. Existing Ruff/third-party warnings are recorded, not hidden. |
| 2.3 — local owner and browser adapters | Commits `f93b78c5d9`, `0bcdc845ef`; all three P1/P2 findings addressed. | Local owner, bookmark, stable identity and scoped adapter tests. Real IndexedDB commit/abort/reopen remains Task 5. |
| 3.1 — mounted selection/review/restore | Commits `9a2c54ff7f`, `5881b02870`; independent fix review clean. | Mounted view/controller tests. Workspace/Research/Document/Model panel routing is not claimed browser-qualified. |
| 3.2 — ordinary/overlay send | Commits `0e71a732d4`, `16e911c79a`, `9213012401`; all original findings and the additional mounted before-first coverage gap resolved after two fix reviews. Milestone `6e8db34b0f`. | Original run 216 tests / 22 files; fix 1: 87 / 7; fix 2: 40 / 4. These overlap. Focused types pass; two proven baseline expanded prompt-sync diagnostics remain. Provider boundaries are mocked in mounted tests. |

The implementation reports, commands, logs and review findings currently live in the plan-owned SDD workspace. The final record must retain the decisive evidence and controller rulings before any workflow-artifact cleanup.

## Reviewed native-send milestone

Task 3.3 commit `4822de5d716c5dc3e2c92144624670a94f3e4489`, based on `6e8db34b0f25345248a89c7f9552a6e6488634c1`, integrates supported tracked-character sends through native versioned completion in both mounted handlers. The server owns input/result persistence; ordinary/overlay sends keep client-managed stateless inference.

Final backend run `/tmp/h1-task33-backend-final.log`: **113 passed, 1 skipped, 10 warnings, 42.43s**. The skipped `TestCreateStreamingResponseWithTimeout.test_heartbeat_integration` is pre-existing and does not count as required H1 proof. Warnings include Starlette/httpx, Pydantic schema shadow/serialization, pytest config, passlib crypt, legacy HTTP422 naming and the existing character rate limiter. The new checks cover the native endpoint acknowledgement wiring, metadata-disabled saved identity, provider-field spoofing, save failure and cancellation.

The worker reports 222 UI tests across 12 files plus 53 mounted tests after a final local-owner gate adjustment; these are overlapping runs. Focused types/static/security pass; expanded types retain two known prompt-sync diagnostics. Root verified the full report, saved test/static/security summaries and exact commit/package. Independent review found one P2: admitted native recovery could say the response was not saved even though a missing acknowledgement leaves persistence unknown. Fix `5d62386395ad1899b1236df1a48af678f0f4b7f3` makes the presentation native-aware and preserves known admission plus inspectable text. Its focused run passed 64 tests across both mounted handlers and the renderer; focused types, formatting and locale generation passed. Fresh scoped re-review marked the P2 addressed with no new important breakage. Task 3.3 and Stage 3 are complete within their stated scope; real-browser/IndexedDB and full H1 acceptance remain outstanding.

## Remaining qualification

- Task 4.1: exact allowlisted local copies, stable mutation IDs, independent child files/parents and comparison materialization. Resolve the deferred owner-specific fork-exclusion invariant.
- Task 4.2: atomic client dispatch claims, honest unknown/partial outcomes and no automatic retry or owner fallback.
- Task 5.1: actual WebUI, extension full page, compact sidepanel and extension expansion; real IndexedDB copy abort/reopen and concurrent bookmark/recovery updates; required native SQLite/PostgreSQL checks; consumer type checks, API generation and security checks.
- Final whole-branch review and fixes. No skipped required mode counts as parity or successful qualification.

## Acceptance status

| Criterion | Status |
|---|---|
| H1-A — independent views, exact selected path and late completion | Mounted/owner evidence exists; full-shell qualification pending. |
| H1-B — before-first, empty, stale bookmarks and reopen | Unit/mounted evidence exists; browser qualification pending. |
| H1-C — complete legacy review, CAS and independent interpretations | Reviewed native owner and local adapter evidence; real IndexedDB and full-shell review/reopen pending. |
| H1-D — accepted-parent admission and settlement with the existing composer | Ordinary/overlay and supported native tracked-character tasks reviewed; final regression and browser qualification pending. |
| H1-E — independent local child, files and supported context | Task 4 implementation and real IndexedDB evidence pending. |
| H1-F — unknown/partial fork outcome and no automatic fallback/replay | Task 4.2 and reload/browser evidence pending. |
| H1-G — model-qualified comparison child, shared controls and account/workspace scope | Comparison copy and final mounted/browser evidence pending. |
| H1-H — capability rejection, unversioned compatibility and internal-field isolation | Existing API/adapter evidence; final client/native regression qualification pending. |

No passing release verdict, complete parity verdict, push or merge is recorded here.

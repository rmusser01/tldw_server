# H2 native fork design review closure — 2026-09-18

Task: TASK-13261.2. Scope: source audit, [focused design](../Design/2026-09-18-chatbook-h2-native-fork-design.md) and [five-stage implementation plan](../../IMPLEMENTATION_PLAN_chatbook_h2_native_fork.md). H1 PR [#2968](https://github.com/rmusser01/tldw_server/pull/2968) remains separate at `ac76c4bc5b035561bf816009c1326a114e87def9`; this continuation is on `codex/chatbook-h2-native-fork`.

The lifecycle review and independent plan review each identified three P2 findings and no P1. Their follow-up reviews closed all six. Two minor plan corrections were also applied. This is readiness to implement the bounded design, not runtime or full-parity acceptance.

## Review evidence and dispositions

| Review | Finding | Resolution and required implementation proof |
|---|---|---|
| [Lifecycle pass 1](CHATBOOK_H2_NATIVE_FORK_DESIGN_REVIEW_PASS1_2026_09_18.md), R1/P2 | Soft delete released bytes required by trash restore. | Receipt becomes gone, but recoverable claims/charge remain until hard purge. Test cleanup between trash and restore. [Pass 2 closure](CHATBOOK_H2_NATIVE_FORK_DESIGN_REVIEW_PASS2_2026_09_18.md). |
| Same, R2/P2 | Stale filesystem publisher could recreate bytes after GC finalized. | Shared stable native I/O guard for chunk/final writes and reclamation; durable state check; drain/remove final and upload bytes before quota release. Test suspended writers and process death. |
| Same, R3/P2 | ChaCha-only reservation did not share existing quota authority. | Operation-keyed existing-service ledger; explicit legacy L/native N accounting; reserve/release crash reconciliation. Native concurrency is serialized; inherited legacy check-then-write races are disclosed. Cache/recalculation tests prevent double counting. |
| [Plan pass 1](CHATBOOK_H2_NATIVE_FORK_PLAN_REVIEW_PASS1_2026_09_18.md), P2-1 | Detached summary update omitted history fence. | Carry post-input history version and settings version; compare both under child owner lock, discard stale write. Hold real edit/delete/append during composition. [Pass 2 closure](CHATBOOK_H2_NATIVE_FORK_PLAN_REVIEW_PASS2_2026_09_18.md). |
| Same, P2-2 | Browser remove/replace had no canonical server transition. | Scoped GET/PATCH native context contract with exact reference/asset/manifest revision CAS, owned replacement and explicit restore hash; lost response reconciles canonical state. No source-job cancellation. |
| Same, P2-3 | Existing reference-image path only understood live MediaFiles IDs. | Typed native reference through both serializers, FileArtifacts normalize/export and authenticated native resolver into actual provider bytes. Async/re-export remains gated; test after original Media deletion. |
| Same, minor transport path | Extension entrypoint was only a re-export. | Name actual shared background entry, background-proxy and request-core paths; both consumers still require real transport proof. |
| Same, minor result key | Test sample used status instead of H1 state. | Preserve `state`, `child_id`, owner/operation and map field names consistently. |

The final interface refinement distinguishes pre-allocation binding templates from child-bound descriptors and includes reference identity/role/context-enabled state in the fork digest. Every child, including a neutral plain child, has server-owned `native_bundle` metadata so cold reopen does not depend on a local receipt or character binding.

## Earlier audit obligations

All [source audit](CHATBOOK_H2_NATIVE_FORK_SOURCE_AUDIT_2026_09_18.md) observations have an explicit destination; they were not silently dropped when the review changed scope.

| Audit observation | Design/plan disposition |
|---|---|
| A1 cross-store authority | Explicit native retention before accepted capture; user-uploaded bytes become independent native claims. No foreign row/path adoption. Tasks2.1–2.3. |
| A2 incomplete document/generated representation | Exact typed original/text/generated revision inventory; missing/unknown cases fail or receive reviewed degradation. Tasks1.1,2.3,3.3,4.2. |
| A3 cleanup/adoption | Candidate CAS + physical writer guard and independent namespaces. Tasks1.2,2.1,2.2. |
| C1 live-card deletion cascade | Protected snapshot child identity with null card FK; readers/filter/edit routes understand it. Task3.1. |
| C2 live next-send behavior | Frozen supported rich composer admitted before current input, plus next provider payload/cold reopen tests with live readers poisoned. Task3.2. |
| C3 excluded context vs retained policy | Fork-purpose canonical projection, nested summary reset, new digest binding, remapped selected pins. Task1.1; independent future summary fencing in Task3.2. |
| R1 receipt/key lifetime | No cascading receipt FK; permanent accepted-key tombstone, committed-first lookup and current child reauthorization. Tasks1.2,4.1. |
| R2 legacy pending records | Tagged protocol/request union; no migration of unknown legacy operation into atomic retry. Task4.2. |
| R3 stale owner/view | Original scoped transport and leases across capture/dispatch/resolve/load/result writes. Tasks4.1–4.2. |
| R4 PostgreSQL lock inversion | Operation→conversation→new/native-asset order; no conversation→old-message row locks; actual existing writer race tests. Tasks1.2,4.1. |

The [retention review](CHATBOOK_H2_NATIVE_RETENTION_REVIEW_2026_09_18.md)'s three integration traps are incorporated: an existing Media ingest is not native ownership; extracted text is not an original PDF/DOCX; selected revision and native references must survive promotion without executable foreign IDs. The complete independent [asset](CHATBOOK_H2_NATIVE_ASSET_AUDIT_2026_09_18.md) and [character](CHATBOOK_H2_NATIVE_CHARACTER_AUDIT_2026_09_18.md) reports remain preserved.

H1 prior findings remain closed under its own [acceptance record](CHATBOOK_H1_HISTORY_SELECTION_VERIFICATION_2026_09_17.md) and [whole-branch fix review](CHATBOOK_H1_WHOLE_BRANCH_FIX_REVIEW_2026_09_17.md); this documentation continuation changes no H1 runtime code and does not rebrand H1 tests as H2 proof.

## Verification boundary

Both source dev pins were rechecked during final document review and remain server `59049e094e0845a4611ea725ae19b7c1754ea709` and Chatbook `e89f28d751bc8a5b4f4545b8894b87437252c657`. GitHub confirms PR2968 is open/draft against dev, with the unchanged H1 head above.

This task changes documentation and Backlog records only. No application tests, builds, database migrations or browser cases were run for H2; Bandit is not applicable to the touched documentation scope. H2-A1–A8 remain unqualified runtime gates until the implementing tasks run them. Required SQLite/PostgreSQL, both full-page shells, compact extension expansion, byte/GC/quota races and frozen provider payload tests are assigned in the plan. H3/H4/F02, unsupported persona/multi-participant/comparison, separate WorkspaceChatPanel qualification, and asynchronous native-reference export remain explicit broader dependencies; no parity row is marked Equivalent.

Documentation audit: 11 documents, 23 local links, five Not Started stages, 12 implementation tasks and all eight acceptance gates checked; zero missing links, trailing-whitespace or placeholder/obsolete-result-field errors. All seven independent audit/review reports match their original report bytes, including the final interface supplement. The associated Backlog task records this result; `/private/tmp/chatbook-h2-documentation-audit.json` contains the per-file hashes from the audit. Staged `git diff --check` is the final whitespace gate before the documentation commit.

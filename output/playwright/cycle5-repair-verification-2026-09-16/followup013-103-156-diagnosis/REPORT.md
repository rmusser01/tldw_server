# Ordinary source Chat diagnosis — 2026-09-16

Source: `2d5ad06c86`. Read-only production investigation; no browser/runtime/provider call, profile reset, repository edit, commit, or Backlog mutation. Root owns tracking and retention. All probes below use public synthetic facts and replace only external/storage boundaries; they are diagnostics of existing bad behavior, not repair verification.

## Evidence limits

The latest native169-second stream/body/canonical snapshots did not survive temporary storage. The conversation records scoped RAG media2 success with Rowan facts, wrapped final request user, then false source refusal; visible count17→19 on reload, ending assistant then raw user. Retained `followup-native-single` evidence is the earlier10-second retrieval timeout only. Do not relabel it as the later failure or assert exact latest row identities from it.

## Proven ordering defect (does not require missing ACK)

Actual `saveMessage` → `reconcileServerChatMessages` → `reconcileServerChatMirror` → `formatToMessage` reproduces current source question below its answer with both canonical ACKs present.

1. `ragMode.ts` renders retrieval wrapper for the provider while `chatModePipeline.ts` passes raw `message` to success persistence.
2. Backend saves wrapped user before generation; SSE carries its ID. Local success saves raw user after streaming completes (`saveMessage`, createdAt=Date.now()+1).
3. `server-chat-mirror.ts:195–207`: unknown/equal-version differing local content is preserved by spreading the entire local row after canonical fields. This retains the user post-stream createdAt.
4. Identical assistant content does not take that preservation branch, adopting the earlier canonical answer createdAt.
5. `helpers.ts:259` sorts by createdAt. Answer is therefore before user. Synthetic timestamps: canonical user1000/answer2000; local user3001/answer3002. Reload actual output: answer2000 then raw user3001. Same-content ordinary control correctly restores user1000 then answer2000.

The mirror content-preservation rule is valid; conflating content ownership with chronology is the defect. The current in-memory reconciliation first remains user/assistant in canonical input order. Persisted-mirror formatting subsequently reverses it.

## Older local timeout-pair promotion: reproducible +2 visible copies

Actual backend builder fixture semantics: saved Cedar history + previously local-only raw source question and plain retrieval refusal + new wrapped source user. No retry metadata, owned neutral conversation, save_to_db=true. ASC and DESC both persist exactly three messages: old local question, old local refusal, current wrapped question. Only the final user receives client correlation and a returned user ACK. Actual adapter last user still contains all five Rowan facts (XML-like delimiters escaped, content retained).

Feed those canonical promoted rows plus final assistant through actual mirror/formatter with original local pair and current user/assistant ACKs: six initial visible rows become eight. The old local question and refusal have no ACK/correlation/anchored canonical answer proof, so safe recovery preserves them and adds the newly saved pair. Current answer then current raw question are last due to the independent chronology bug. This explains the *shape*17→19 without assuming the latest current-user ACK was missing. Native exact identity confirmation remains unavailable.

This is not proof that the backend stored two canonical copies of the latest user. The reproduced server appended previously unsaved history; visible duplicates result from missing promotion receipts. No text deduplication is warranted. The plain retrieval refusal is not a `__tldw_error__` envelope and should not be silently reclassified as such.

## Missing current ACK control (separate conditional defect)

If ACK is absent, exact client correlation recovery refuses raw local question vs wrapped canonical user because content equality is mandatory. Actual reconciler returns wrapped user/answer/raw local user even with matching client ID. This is separately reproduced but is not required for the proven timestamp defect and is not established as the latest native cause. Existing stream transport and endpoint correctly forward user_message_id in the inspected source.

## Grounding / wrong answer remains unresolved

Private backend probe uses real builder, prompt templating, call-parameter construction and CustomOpenAIAdapter payload builder, with mock storage and no network. Under ASC and DESC, final provider user contains Rowan, Mara Chen,7December2026,east,Friday19:30. Escaping `<doc>` is visible but no evidence establishes it caused refusal. These reduced fixtures are not an exact recovered native transcript. Neither ordering repair nor duplicate handling proves source-answer acceptance. Root must retain a fresh actual provider request/body and final answer on the original source scenario.

## Existing tracking search

Searched tracker and all task13260 descendants for mirror, timestamp, ordering, raw/wrapped and RAG duplicate terms. UAT103/TASK13260.44 already owns duplicate visible saved Chat rows and canonical user acknowledgements. UAT070/TASK13260.15 owns broader mirror content/identity preservation; its later notes explicitly defer duplicate mirror103 to.44. UAT108/.49 owns explicit failed Retry identity and diagnostic-envelope projection; this older plain retrieval refusal is distinct. UAT131/.71 owns first greeting promotion receipts, not later local history promotion. UAT013/.3 already records the latest source refusal and17→19 observation. No prior tracked finding specifically names RAG raw/wrapped timestamp inversion. Recommend root keep source refusal in013; associate duplicate follow-up with103/.44 rather than duplicate issue creation, and decide whether ordering deserves its own finding or explicit mirror-follow-up acceptance case. No ID assigned here.

## Minimal repair proposal and required controls

### Chronology (bounded/proven)

Separate canonical chronology from editable content: for an already matched canonical row, preserve protected content/local draft fields while taking a valid canonical createdAt. Apply consistently to memory and persisted mirror; when remote timestamp absent/invalid, retain local fallback. Do not change the raw user question to the provider wrapper to make equality pass, and do not alter identities or delete rows. Candidate scope `server-chat-mirror.ts` plus its tests; helpers sorting is behaving correctly.

Permanent behavioral regression should use actual saveMessage/mirror/formatToMessage, realistic post-stream timestamps, acknowledged raw/wrapped user and identical assistant. Desired assertion user→assistant (our current diagnostic asserts the known broken order). Controls: ordinary identical content; protected newer/unknown-version local edits; beforeAwait edits; local drafts/repeated equal content remain; missing canonical timestamp fallback; same-ID owner/history rejection; repeat reload idempotence; older local timeout pairs; acknowledged assistant parent links; both immediate memory and persisted formatter output.

### Promotion identity (separate design before editing)

The completion API currently only acknowledges its latest user and new assistant. Previously local-only request-history rows are saved without durable source IDs. A safe repair needs exact source-row receipts or explicit promotion before completion with ownership checks. Reuse existing tracked promotion mechanisms where possible; do not deduplicate text or simply discard local history. Alternatively define which local-only diagnostic rows intentionally stay local and exclude only those through explicit provenance. The plain refusal fixture must not be treated as an error envelope by text. This is broader than a timestamp correction and is not yet a selected implementation.

## Probe results

- `mirror.test.ts`:4 diagnostic assertions PASS with real production functions; one proves wrong ordering, one ordinary control, one conditional missing ACK, one older local pair +2 duplicates. These are *known-bad-behavior assertions*, not passing user acceptance tests.
- `backend_probe.py`: ASC/DESC assertions PASS;3appends with correlation only on latest user; all Rowan facts reach final adapter message. `backend-results.json` contains complete synthetic output.
- No tests of repaired code or Bandit claim (no production changes).
- `manifest.json`: SHA256 for all probe/report outputs and investigated production files at capture. Backend import log intentionally not copied into report; it has no required evidence beyond synthetic JSON output.

# UAT103 independent implementation review

2026-09-16. Read-only source/test review by sidebar155_review. Root independently runs the combined suite/compiler checks. No production/test/task/root-doc edits, runtime/browser/inference, staging, or commit by this reviewer during this pass. UAT158 remains frozen and separate.

## Final bounded verdict

**No remaining concrete defect found in the refreshed reviewed scope.** The system-role deletion finding below is corrected, and the independent private counterexample now passes. Root owns the full suite/compiler/native acceptance; this is not a native UAT closure.

Final manifest SHA-256: `2d972e7ac51ef8b715264e3fdbff150d2f236f29511afaa5d727a067bf647c87`. Independently verified all **8/8 production, 6/6 test, and 31/31 evidence entries** against the refreshed manifest on 2026-09-16 at approximately 13:28 local. Exact source/test hashes are copied to `/private/tmp/uat103-independent-final-hashes.json`.

Key corrected files:

- useChatActions.ts: `47ffe105f418a2a458ebb58bd92ddae89369bc2289feeecfa21bea94232a5113`
- useChatActions.saved-normal.integration.test.tsx: `d1dbba6d620874e139c32d956ec2c0c614cb497890f062087ff03b97df781679`

Selected independent final probe: **1 passed, 91 unrelated cases filtered**, exit 0, 13:28:12 local. Log: `/private/tmp/uat103-review-system-delete-final.log`. The equivalent corrected post-mount baseline probe passed and pre-correction current probe failed, establishing the regression and resolution without altering repository source.

The final narrow role correction is `message.role ?? (message.isBot ? "assistant" : "user")`. The permanent post-mount regression preserves both visible and prompt role=system while deleting the diagnostic. The Continue wrapper now checks its returned result explicitly before returning the skipped fallback; reviewed behavior matches the intended failed/skipped/submitted contract. Author's final combined log records 215/14 passing; root independently owns full verification and compiler baseline comparison.

## Review target and integrity

Initial final handoff: `.tmp/uat103-local-preflight-20260916/owned-manifest.json`, SHA-256 `72bd2c7a3d7d5cf9072dadcc91b31f844d286eb1a864571d2d25accb36b6166a`, HEAD `38fd4948216d36043ad9ea1191cbb8fff614642f`. Independently verified all 8 production paths, 6 test paths, and 24 evidence entries matched that manifest. Author subsequently reopened two narrow corrections: root-found Continue return-type compiler error, and the concrete deletion-role finding below. The final verified release and verdict appear above; the following finding retains the review history and original failure evidence.

## Concrete finding — preserve canonical system roles when deleting a local diagnostic

At `useChatActions.ts` new diagnostic-only deletion branch (~4620), `buildHistoryFromMessages(...)` rebuilds the retained prompt rows. That existing helper mapped every non-bot row to role=user, ignoring an explicit role=system. A canonical system row loaded after component mount retains its visible system role but becomes a user message in the next prompt when the user deletes an excluded diagnostic user/assistant.

**Reproduction:** mount the actual action harness; then supply loader-shaped rows containing an ACKed system row and an exact marked local diagnostic pair, with compact prompt history containing only the system row; delete the local diagnostic user. Before-delete assertion proves role=system. Current implementation returns role=user; baseline useChatActions from 38fd494 preserves role=system.

Private nonmutating Vite injection artifacts:

- `/private/tmp/uat103-review-system-delete-test.txt`
- `/private/tmp/uat103-review-system-delete.vitest.config.mts`
- `/private/tmp/uat103-review-system-delete-current.log`: 1 failed selected probe.
- `/private/tmp/uat103-review-system-delete-baseline.log`: 1 passed selected probe.

The first exploratory fixture mounted with rows already present and both versions failed because a preexisting mount-time rebuild changed the role. That fixture was corrected to inject rows after mounting, as an actual loader does. Only the corrected current-fail/baseline-pass pair establishes this new deletion regression. Author and root were explicitly notified of this distinction.

Recommended bounded correction: preserve explicit existing roles in the history builder (fall back to bot-derived user/assistant only when absent), with the post-mount system-role regression made permanent. Author applied it alongside the compiler correction; independent final verification is recorded above.

## Confirmed design boundaries

- Complete mode=rag/grounded=false/exact reason signature; no prose or missing-ACK-only classification.
- Canonical receipt guards, exact parent identity, and real-answer variant checks preserve acknowledged users, repeated genuine turns, detached drafts, and temporary success without ACK.
- Producer `skipHistoryAppend` and text fallback exception match the existing saveToDb:false contract. Image-event sync and ordinary handled text remain independent.
- All current prompt projections and global question rewrite remove the diagnostic before losing provenance. Context mutation retains the RAG preflight WeakMap identity; actual cache-reuse control is present.
- Exact-parent Retry no longer inherits older image/type, reuses the local user identity, preserves prior eligible history, and avoids backend failed-turn reuse. Character diagnostic Retry bypasses branch creation.
- Edit index correction and Continue result behavior have focused controls. Root independently identified a TypeScript narrowing problem in the Continue wrapper; the explicit narrowing correction is reviewed in the final release.
- New-chat promotion uses the same eligibility view for ordered source/receipt alignment and retains existing ownership guards.
- Local visible/Dexie rows remain; server-chat-mirror is untouched.

## Limits

This is source/test review plus a single selected counterexample/baseline probe, not a duplicate execution of the full 214-case suite. Root owns full scoped tests, TypeScript baseline comparison, and fresh native source failure → later Send → settled reload acceptance. No certification of lost prior native artifacts. UAT013 source-answer correctness, UAT156 chronology, and UAT157 image duplication remain separate. Already canonicalized diagnostics and other pending normal/partial history receipt cases are outside this repair's claimed closure.

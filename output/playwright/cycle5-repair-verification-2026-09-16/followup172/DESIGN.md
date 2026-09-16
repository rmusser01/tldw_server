# UAT172 — recover the generated-card save after a deck-list failure

Task: TASK-13260.109. Scope: GeneratePanel and focused mounted behavior tests. Parent owns native acceptance, shared design/tracker, and commits.

## Diagnosis
The real scoped deck query has failed. The production QueryClient disables refetch on focus/reconnect. Save/Retry only checks isSuccess and throws the readiness error, so neither action can recover after the server becomes healthy. No mutation starts. The live DOM shows Save and Retry enabled and no AntD loading class; the old snapshot loading icon is a harness artifact, not an isSaving defect.

## Approved small change
When resolving a save target, refetch an errored deck query through its existing observer and await its result. Use the successful returned catalogue directly for this save, including an empty list; do not use stale closure data or bridge an older create acknowledgment over a newer catalogue. Preserve the pending-list guard. Keep current-scope assertions before the read, after the read, and before all mutations; unresolved/aborted/replaced scope cannot save. Propagate a failed read through the existing visible save error and retain drafts/source. Ready lists keep the existing request behavior and deck-selection policy. No query-client, service, or account protocol changes.

## Verification stages
1. Permanent mounted GeneratePanel + real scoped useDecksQuery/QueryClient + real mutation hooks, mocking only service boundaries. RED: failed read, failed first Save, healthy retry saves the original edited draft and source attribution under the original account.
2. GREEN: minimal resolver change. Controls: still failing; pending; account abort/change/unresolved; refetched catalogue rejects a deleted selected deck; ready-list path; no duplicate deck/card save.
3. Focused regressions, ESLint baseline comparison, compiler before/after comparison, Bandit with TSX parse limitations, exact patch/snapshot/manifest for independent review. Parent alone clicks preserved native draft after review.

## Stage status at handoff
Stages1–2 complete; stage3 author checks complete (65/4 pass, lint unchanged, compiler90/90, Bandit TSX limitation). Independent review and parent native acceptance remain pending. See IMPLEMENTATION.md for exact failures, commands, and frozen manifest.

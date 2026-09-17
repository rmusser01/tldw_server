# UAT205 — accepted Study Pack job lifetime

TASK13260.143 exists before edits. Parent approved the bounded pending-job guard/status repair; native acceptance and worker availability investigation remain parent-owned.

## Root cause and choice

The drawer saves the POST202 job ID but `canSubmit` checks only the POST mutation and the button spinner checks `jobQuery.isFetching`. The existing query deliberately stops fetching between 1500ms polls, so an accepted queued/running job leaves no status and permits another POST. Keep the existing job ID as the pending lifetime until existing failed/cancelled handling clears it or completed navigation closes the drawer. Add a polite status message for accepted/queued/running and temporary status-fetch errors. Do not change worker/configuration, service calls, query polling, success navigation or terminal retry inputs.

Similar existing patterns inspected: SourceReviewPlanDrawer keeps POST pending/error in its drawer; QuickIngest ProcessingStep renders durable job tracking and a role=status polite banner independently of request fetching; GenerateCharacterPanel marks the complete generation lifetime busy/live. Existing useStudyPackJobQuery already treats completed/failed/cancelled as terminal and owns polling.

Existing ImportExportTab is keyed by the generation/account authority. Preserve that remount boundary. Existing close→reopen/new intent resets the local drawer; durable cross-close/reload job tracking is not introduced without scope approval.

## Three stages

1. **Causal RED:** real Drawer + real TanStack create/poll hooks; fake only remote Study Pack service, notification/navigation seams and feature availability. Prove first-poll delay, queued/running settled gaps and polling errors retain an accessible status and prevent a second POST. Controls cover failed/cancelled retry preserving source/title, success navigation once, rejected POST, and account-key remount/late response isolation.
2. **Minimal GREEN:** drawer pending guard and status copy, authoritative English keys if needed, focused permanent test file. No generic state abstraction.
3. **Verification/handoff:** old drawer + query and affected ImportExport controls, scoped ESLint, current/baseline compiler comparison, Bandit TSX parse limitation, exact frozen source/test manifest for parent independent review. Native queued→terminal acceptance pending.

## Approved independent-review correction

A completed job may have no usable deck (missing/deleted result); the real hook correctly stops polling at completion. The initial pending guard therefore exposed a terminal dead end. Parent approved explicit terminal recovery within TASK13260.143: show a translated result-unavailable error, clear the accepted job ID, preserve inputs and require deliberate retry, with no success navigation. Two actual-hook permanent controls failed before the narrow correction and pass afterward. Five English keys total; no query/worker changes. All three stages are complete for author verification; independent re-review and native acceptance remain pending.

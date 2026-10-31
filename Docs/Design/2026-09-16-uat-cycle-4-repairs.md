# Cycle4 UAT repair design

## Evidence and scope

TASK-13260 owns the continuing fresh single/multi acceptance effort. Frozen product `7c9409fad2` was exercised through the named frontend journeys and shared workflow definitions. Both matrices ended with failures/explicit blocks; no release sign-off exists. Evidence is preserved in `output/playwright/cycle4-full-uat-2026-09-16/`. The independent audit verified203 single captures and62 diagnosis hashes, parsed28 multi JSON files, and found only historical summary labels to correct. Checkpoint `b2595a4edb` preserves the completed run. Merge `f4e9f954d7` includes freshly fetched dev `59049e094e0845a4611ea725ae19b7c1754ea709`.

Repair the sixteen new findings102–117 and confirmed058/067 scope gaps before another full fresh run. Existing dependencies are reused; this cannot certify clean-machine dependency installation. Wikipedia access blocking remains an explicit external limitation. Successful source preservation with a failed analysis is not a successful analysis. A successful HTTP response containing no final answer is not a completed answer.

## Design decisions

### Saved Chat identity and recovery — tasks44/49/51/53/57

Normal Chat must carry canonical user-message acknowledgement through the existing completion metadata, pipeline and local persistence boundary, as it already does for assistant identity. Character Chat already obtains a canonical user ID; propagate it to the same mirror boundary. Preserve local primary keys, server identity, captured owner/generation and user drafts. Do not deduplicate arbitrary equal text: two identical user turns can be intentional, and an unsent draft must survive. Defensible legacy recovery must be anchored to saved-turn provenance; ambiguous rows remain explicit unsaved state. The Note backlink guard stays intact and must accept genuinely acknowledged saved turns.

Retry should project only conversational messages to the model and reuse the failed turn. Recognized assistant error-display envelopes are UI state, not assistant prose. Preserve malformed/user-quoted marker text and meaningful partial answers. Character creation entry is consumed explicitly, then the saved conversation becomes the canonical reload destination; explicit new-character actions must still create new drafts. A reasoning-only completion retains its history/reasoning but presents a recoverable missing-final-answer status. Do not infer a finish reason that was not captured.

### Model selection and configuration — tasks47/48/55

Use the existing consolidated model owner for explicit Media selection and for a one-time successful setup handoff. Publish the verified provider-qualified selection before completion refresh can unmount setup. Preserve an existing deliberate choice, a newer selection during verification, and account/target generation boundaries. Never continuously impose the server default or write storage behind the mounted model owner.

Chat model validation must read the current provider configuration after setup saves. Remove stale parser/cache authority while retaining strict unknown-model rejection, aliases, environment precedence and numbered provider slots. A restart is not the repair.

### Ingest/analysis outcomes — tasks45/46

Quick Ingest Minimize must invoke the modal owner's existing dismiss path while preserving jobs, resumable state and account ownership. Closing the modal does not cancel processing.

At the non-streaming summarization boundary accept only nonblank supported answer text. Remove the arbitrary object-to-string fallback. Empty, malformed, reasoning-only or length-truncated responses produce a safe analysis warning through the existing caller contract; source ingestion remains preserved. Reject even nonempty length-truncated analysis rather than silently declaring it complete. Do not log provider envelopes or mutate previously stored UAT data. Keep this local to summarization; no broad adapter rewrite is justified.

### Async failures and notifications — tasks50/54/56

Own/catch stream-reader cancellation rejections; expected timeout/cancel/provider failures use the existing local error surface without causing a development overlay. Preserve prior analysis and recovery actions. QA empty-output guidance uses the completed request's generation settings, distinguishing disabled generation, requested-but-empty output, evidence insufficiency and transport failure. A recovered request clears stale assertive announcements.

Suspend hidden-tab notification streams and catch up when visible, preserving auth rotation, cursors, unread state and account isolation. Start with visibility-aware transport ownership; avoid introducing a global cross-account stream or custom coordination framework without evidence that visible-window concurrency still requires it. Verify six-tab ordinary requests on the actual browser.

### Presentation gaps — tasks43/52/20/15

Use contextual AntD feedback for admin creation and Prompt synchronization. Give the Flashcard FAB a meaningful stable accessible name and hide completed decorative loading indicators from the accessibility tree without disabling the action. Add actual titles to Prompts/Characters and reject late Chat-title publication after Settings navigation/reconnect. Reproduce the New-character disconnected-form warning through its actual lifecycle before changing that lifecycle. Triage the recorded raw Markdown Review observation against the component's intended presentation; record the conclusion rather than silently losing the observation.

## Constraints and verification

- Preserve authentication, role permissions, account isolation, source classification and external restrictions.
- Preserve unsaved drafts, prior successful source/analysis and canonical identity; no broad text deduplication or store reset.
- No live credentials in logs, tests, commits or retained artifacts. No mocked response counts as real-model acceptance.
- Every repository edit belongs to an existing Backlog task, updated only through official MCP/CLI.
- Use current libraries and actual component/store/service boundaries. Reuse retained failing probes as evidence, then add permanent behavioral regressions before production edits.
- Activate `.venv` before Python/pytest/Bandit. Run Bandit on touched Python and resolve new findings.
- Commit reviewable units, preserve unrelated work, and retain exact command/results. Existing TypeScript/lint baselines must be compared, not called clean.
- Independent review and targeted native verification precede another full fresh single/multi run. External blocked rows remain explicit.

## Alternatives rejected

Text-only Chat deduplication can delete legitimate drafts. Storage-only model writes reproduce115. Raising token/browser limits masks105/114 without fixing outcome or connection ownership. Suppressing all console errors hides real failures. Restarting the API after setup hides107. Reseeding UAT data or replacing the blocked Wikipedia source would invalidate workflow evidence.

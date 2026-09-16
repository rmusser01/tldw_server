# UAT178 / TASK-13260.115 — Cram queue failure is not completion

## Actual cause
ReviewTab collapses missing query data into an empty array. The completion card checks loading but not query success, so a failed selected-deck queue GET is displayed as completed Cram. The auto-end-session effect already requires isSuccess and is not the cause. Native evidence shows a real saved card and repeated500 queue responses; original039/session-start and128/identity-based queue progression are separate.

## Proposed bounded production change
ReviewTab only. Require a successful settled Cram query before showing successful completion/recovery UI. Show an inline failure with Retry calling the existing cramQueueQuery.refetch; reflect its fetching state to prevent repeated clicks. Show initial loading when Cram has no active card. Preserve cached active cards, practiced identities, deck/tag/scheduling selections and the existing auto-end effect. Due-mode behavior and query/service implementations remain unchanged. No useFlashcardQueries edits; another agent owns it.

## Tests before implementation
Extend the existing mounted Cram suite with actual useCramQueueQuery and QueryClient, mocking listFlashcards at its service boundary. RED failure→visible Retry→card, continued failure and busy retry, initial pending never complete, cached-card/progress preservation, true success empty/completed controls. Existing queue identity/rerating tests retain their behavior. Correct mocked successful-query fixtures to expose the actual isSuccess/isError/loading flags where needed; do not weaken production success checks for incomplete mocks.

## Ownership / stages
Task is In Progress. Parent owns browser/runtime, DB177 repair, tracker/task/commit and native failure-to-recovery acceptance. Parent approved this bounded proposal before production edits. Stages: 1 diagnosis/design complete; 2 permanent RED complete; 3 minimal GREEN complete; 4 static verification complete, independent review/native pending.

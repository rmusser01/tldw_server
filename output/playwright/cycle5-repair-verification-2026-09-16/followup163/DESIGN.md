# UAT163 bounded design
Associated task TASK-13260.100; root explicitly authorized the existing repair unit. No shared design/plan/tracker edits, browser/services or commits by author.

## Cause
Real Home rag_media handoff sets chatMode=rag, ragMediaIds, and fileRetrievalEnabled=true. Session snapshot/store partialization preserve the first two only. Actual cold option state defaults the flag false. shouldUseRagForTurn requires the flag plus selected IDs, so a reload restores the same conversation and IDs while routing Send to ordinary completion.

## Change
Persist an explicit boolean with initial false in existing store; select/snapshot/depend on it in usePlaygroundSessionPersistence and restore ===true under the existing source-selection guard. Do not infer consent from rag mode or IDs. Preserve existing owner, restoreRevision, and intentional chat-switch clearing contracts. If a new actual callback regression proves that explicit user disable loses to delayed restore, mark user intent only on the Form Knowledge retrieval toggle callback, following existing Form handoff intent marking. No generic persistence layer or new race abstraction.

## Behavioral checks
Cold real localStorage write/rehydrate, no replayed handoff, actual Send into request serializer and returned source facts. Positive prior retrieval-failure case. Explicit false and legacy missing false. Account/server rejection, selected-chat cancellation, accepted source handoff intent, and same-value enable/disable intent through pending restore. Existing UI/action mocks retain real session and action boundaries; sourceFlow does not mount the real useMessageOption wrapper, so user intent tests target the real Form callback passed to KnowledgeSection, not a raw store mutation. This limit will be recorded in the author report.

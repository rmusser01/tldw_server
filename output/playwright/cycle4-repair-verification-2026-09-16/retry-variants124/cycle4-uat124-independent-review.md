# UAT124 independent review

**Clear within the approved bounded scope. No actionable findings.** Reviewed both frozen code/test paths at 2026-09-16T07:40:23.826Z; **2/2 hashes match** /private/tmp/cycle4-uat124-owned-manifest.json. Audit: /private/tmp/cycle4-uat124-independent-manifest.json.

## Source assessment

The general pipeline catch now forwards resolvedAssistantParentMessageId to saveMessageOnError. This matches its own live assistant builder, successful completion and interrupted-stream saves. The resolver retains an explicit caller parent; otherwise first send uses the resolved local user and Retry uses the previous assistant parent/fallback user. The same captured value flows through scoped and unscoped persistence helpers. No server Retry intent, client correlation, ACK, owner/cancellation, schema or restoration algorithm changes.

The existing formatter groups only eligible assistant records with a nonnull shared parent, omits compare/cluster groups, sorts variants by creation time and activates the latest generated variant. Thus supplying the already resolved parent fixes the native live-versus-reload discrepancy without content-based deduplication or guessed legacy identity.

## Independent verification

- **42 tests /5 suites passed**, /private/tmp/cycle4-uat124-independent-green.log. Suites: new error-variants persistence3; abort-lifecycle12; saveMessageOnError9; saveMessageOnSuccess.scope11; messageHandlers.regenerate7. New permanent test uses actual Retry handler, pipeline, error helper, saveMessage/PageAssistDatabase and restore formatter with only the storage table/model boundary controlled. It tests three failures, restoration between retries, exact user/image, ordered variant IDs/latest active index, and independent legacy unparented rows. Existing tests cover explicit Retry/ACK and cancellation/account-scope persistence controls.
- **2 actual Dexie round-trip probes passed**, /private/tmp/cycle4-uat124-real-dexie-independent-green.log. Unchanged author probe/config /private/tmp/cycle4-uat124-variant-persistence-probe.{test.ts,config.mts}: actual pipeline/save helper/Dexie schema over fake-indexeddb, DB close/open, implicit and explicit parent controls. Persisted three assistant parents all equal user-local; reopened formatter restores one assistant with three variants and the original image.
- Inspected author retained permanent RED2fail/1pass and implementation report; no production edits made by review.

## Interaction and limits

UAT125 separately suppresses misleading accessible completion announcements. Restoring failed variants does not make them successful; the fixes operate on distinct persistence/presentation boundaries. Known legacy unparented rows remain separate, and manually swiping to an older historical variant remains memory-only, as explicitly excluded by the approved design.

No native browser/runtime/inference, repo/task/tracker edits, whole compiler or lint/security run by this reviewer. Actual IndexedDB behavior is represented by the installed in-memory fake-indexeddb shim, not the user's browser. Native repeated Retry/reload verification remains root-owned.

# Retain invalidated native-history leases during automatic restoration

Tracking: TASK-13264.8, release PR3027.

An account configuration event invalidates the mounted native-history lease and hides fork outcomes. Assistant readiness changes then restart Playground initialization; saved-reference restoration and the server loader can each acquire a new lease automatically. The existing held-response epoch fences work, but these new reads erase the owner-change warning. This fixture reproduction does not establish a backend cross-principal leak.

Add a controller predicate that rejects automatic loading when its current native owner has an invalid lease or a request_config_scope_changed error. Check it before automatic Playground initialization, persistence beginLoad, and server hydration. The sidebar writes a transient server selection intent, including same-ID reopening. The loader consumes it once and passes its identity, selected ID and cancellation predicate through loadConversation and the owner lease. Account invalidation clears the intent; it is never persisted. Checks before the replacement account watcher installs and capture/error publication prevent a cancelled intent from acquiring new authority. Explicit loadConversation remains the deliberate reopen path; cold controllers without an invalidated owner retain normal bookmark validation. No persisted credential, bookmark, projection, recovery or fork records change.

Regression coverage: held success/error after invalidation remains fenced; automatic restore after readiness false→true makes no replacement capture; server hydration makes no settings/message calls; explicit reopening succeeds; existing cold bookmark validation and end-to-end account/fork isolation remain intact.

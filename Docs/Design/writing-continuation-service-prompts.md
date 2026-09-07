# Writing Continuation Service Prompts

Approved bounded slice, tracked by TASK-13216. Follows merged PR #2930.

## Scope

Expose two independent literal `system` parts through existing Service Prompts:

| Definition | Default |
| --- | --- |
| `writing.continuation.predict` | `Continue the text from the prompt. Respond with only the continuation.` |
| `writing.continuation.fill` | `Fill in the missing text between the prefix and suffix. Respond with only the missing text.` |

Use the existing registry, owner-specific storage, Settings editor, validation,
revision handling, reset operation and packaged-default fallback. No new API,
database schema, settings system or dependency. Both clients use shared UI code.

These prompts apply only to non-chat Generate. Chat mode continues to use its
existing messages and explicit context; it must not fetch either definition.
Revision actions, Writing Agent, feedback and manuscript annotations are not
part of this slice. Braces in authored instructions are literal, not variables.

## Request behavior

Load exactly one selected definition per non-chat request before dispatch or
placeholder removal. Pass its `requestScope` and `scopeSignal` to the existing
`TldwChatService` options for both send and stream. Validate the system part
before using it. Keep current message construction, context ordering, fill
template/fallback, prefix/suffix, stop strings, model/provider settings,
logprobs, cursor insertion and undo/redo behavior.

Older-server catalog/detail 404 compatibility uses existing packaged defaults.
Authentication, scope, validation and non-404 failures do not become defaults.

## Async ownership

Give each generation a synchronous request identity and caller controller before
the first await. Check that identity, current session/scene binding and scope
before accepting tokens, logprobs, final text, history, errors or cleanup.
Cancelled old requests cannot finalize or clear a newer request. Cancel lookup
as well as transport on Stop, binding changes and unmount. A late snapshot must
be released even if its caller was cancelled. Release the request lease and
listeners on every terminal path.

Manual Stop preserves already accepted partial output and its existing undo
behavior. Scope/binding invalidation is different: do not commit partial text to
history or persistence, and never restore an old manuscript into a new binding.
Any rollback of this request's provisional text must verify both unchanged
binding and that the editor still contains this request's last emitted text.
Do not erase unrelated user edits or introduce account-wide manuscript cleanup.
Same-account token refresh remains supported by the existing scope lease.

## Verification

Test both definitions and both transport modes, defaults and overrides, literal
braces, Settings save/reset, older-server fallback, explicit chat precedence,
lookup cancellation, stale chunks/responses/logprobs/errors/finalizers,
session/scene changes, manual Stop, unmount and cleanup. Preserve existing
Writing Playground and TldwChat regression suites. Run backend registry/API
tests, touched-scope lint and Bandit, then independent code review.

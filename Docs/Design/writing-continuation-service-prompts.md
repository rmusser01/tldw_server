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

## Verification results

### Rebased code revision `5265b6b90f` (2026-09-12)

These results supersede the pre-rebase verification below. The branch was rebased
onto `dev` at `0a5d0d6e0a`. Qodo's scene-refresh finding was reproduced by two
regression tests, then fixed by including `activeSceneVersion` in continuation
ownership. Both streaming and non-streaming requests now discard a late response
after a newer saved scene version replaces their starting binding.

- The 13-file client regression union passed **390/390** tests; the backend
  Service Prompts registry/API pair passed **101/101** tests.
- The shared UI typecheck reported **192 diagnostics**, identical to latest
  `dev` after line/column normalization. No new diagnostics were introduced.
- Changed-file ESLint reported zero errors and 27 baseline warnings; Ruff
  passed and production-registry Bandit reported zero findings/errors.
- Independent review approved the scene-version fix. Qodo subsequently marked
  the scene finding resolved and dismissed the cancellation finding: the local
  controller already aborts the snapshot transport signal before lease release.
- The production WebUI build completed successfully in
  [container-build-check](https://github.com/rmusser01/tldw_server/actions/runs/34701380058/job/103573643658),
  running `bun run build:prod` through `Dockerfiles/Dockerfile.webui`. The log
  confirms a fresh compilation, not merely a restored build artifact.
- The production Chrome extension build, `bun run build:chrome:prod`, completed
  successfully in the
  [frontend-required job](https://github.com/rmusser01/tldw_server/actions/runs/34701380036/job/103574310043).
  Both CI runs identify `5265b6b90f` as their head revision.
- As checked at 15:34 UTC, backend, container, coverage, E2E, security and trusted
  frontend-license gates had passed. The overall frontend-required job was still
  running later lifecycle/admin checks; its successful extension build does not
  imply that the entire job had finished. Live-browser smoke was not run locally.

Any later documentation-only revision retains this code-verification provenance;
merge readiness still requires the GitHub gates on that revision, rather than
assuming these earlier check results apply to a new head.

### Historical pre-rebase verification

At `3f67f71fa4`, the then-final 13-file shared UI union passed 388/388 tests. The backend
registry/API pair passed 101/101 tests with 14 existing environment/deprecation
warnings. Ruff passed for the registry and its two test modules. Bandit reported
zero findings and zero errors across the 748-line production registry module.

Repository-pinned ESLint passed the changed TypeScript/TSX files with zero
errors. It retained 37 baseline warnings: 27 unused-variable/React-hook warnings
in the pre-existing Writing Playground component and 10 `no-explicit-any`
warnings in `tldw-server.ts`, plus the existing root-invocation Next pages
directory notice. The fresh shared UI typecheck at this head produced 158
diagnostics; after line/column normalization its diagnostic multiset exactly
matched the 158-diagnostic pre-change baseline, so this patch adds no TypeScript
diagnostic. All five new English locale values match between the nested WebUI
tree and flattened extension `.message` entries.

The registry/default/fallback/Settings and scoped-generation patches each
received independent approval. Full-branch review found that revision preset
and queue mutations could autosave provisional continuation text. The final fix
adds disabled controls and synchronous mutation-admission guards; regression
tests inspect actual saved payloads after scope invalidation and debounce.
The fix passed 134 Writing regression tests and scoped re-review with no new
blocking findings, followed by the 388-test combined run above. Full
frontend builds and live-browser checks were not run. The broad accidental TTS
suite's two failures were reproduced at the original base revision with the same
dependencies and are recorded as baseline; no TTS changes were made.

Known runner output remains visible rather than suppressed: Ant Design Drawer
deprecation messages, Node experimental `localStorage` warnings, jsdom
navigation notices, expected logged abort/timeout errors, pytest configuration
warnings and the legacy single-user API-key warning.

An adjacent unchanged idle revision Apply path can persist previous manuscript
text alongside applied status: the editor update precedes revision persistence
through a callback capturing old editor text. This was documented for separate
follow-up in TASK-13217, not fixed by changing continuation or session persistence here.

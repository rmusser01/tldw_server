# Writing Agent Service Prompts

Approved bounded slice, tracked by TASK-13209.

Expose literal `system` parts for `writing.agent.quick`,
`writing.agent.planning`, and `writing.agent.brainstorm` in the existing shared
Service Prompts editor. Keep current default bytes, model/provider settings,
temperatures, token limits, and manuscript context formatting/bounds.

Load one owner-scoped snapshot per send before manuscript reads, and pass its
scope through those reads and model dispatch.
The three manuscript GET paths join the existing exact scoped-transport
allowlist and enforce the existing optional expected-user header on the server.
Unscoped clients remain compatible. Retain the scope lease while
conversation history is visible so an account/server boundary clears completed
history too. Cancel and discard stale context, replies, and errors when the
account/server, project, or mode changes or the component unmounts. Ordinary
context-fetch failures remain best-effort; scope failures must stop generation.

Reuse packaged-default fallback for older servers that lack Service Prompts or
these definitions. Authentication, validation, and non-404 failures must not
silently become defaults. Existing unscoped manuscript service callers retain
their API behavior. No new storage, endpoints, jobs, or configuration system.

Tests cover mode selection, save/reset, default compatibility, literal braces,
bounded context, scoped transport, invalid configuration, and asynchronous
boundaries. Run focused backend/frontend suites, lint, Bandit and code review.

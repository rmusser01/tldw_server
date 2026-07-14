# Chat Workspace Hydration and Offline Follow-up Design

## Scope

Address the two remaining PR #2600 review findings without expanding the Chat Workspace runtime model:

1. A persisted workspace ID must not make chat or its rails ready before the workspace store finishes hydration.
2. Browser coverage must prove that the visible rails transition from connected/streaming to offline without leaving stale streaming status behind.

## Design

`ChatWorkspacePage` owns the readiness decision because it already reads workspace identity, connection state, and passes the send gate into `ChatWorkspaceConsole`. It will read `storeHydrated`, derive `workspaceReady` from both hydration and the normalized workspace ID, and pass that boolean into the console. The console will stop recomputing readiness from the ID so `WorkspaceChatPanel`, `WorkspaceStatusStrip`, and `InspectorRail` all consume the same decision.

The existing live-backend Playwright test will transition the real browser-side connection store through its existing `window.__tldw_useConnectionStore` exposure. No production test hook or new runtime abstraction is needed. The test will begin during active streaming, set the connection to an unreachable error state, and assert that both rails show server-unavailable recovery state while no longer showing `Streaming`.

## Error and State Precedence

- Store hydration and workspace identity jointly gate chat readiness.
- Backend unavailable continues to take precedence over hydration, send failure, and streaming in both rails.
- The offline browser test changes only connection state; it does not alter or replace the chat stream implementation.

## Testing

- Vitest regression: a non-empty workspace ID with `storeHydrated: false` keeps the chat backend disabled and both rails in loading state; rerendering with hydration complete enables the backend.
- Playwright regression: active streaming transitions to server-unavailable rail state after the real connection store becomes unreachable, and stale `Streaming` labels disappear.
- Re-run the full Chat Workspace Vitest folder, focused live-backend smoke, TypeScript, diff checks, and touched-scope lint where supported.

## Non-goals

- No new readiness enum or selector abstraction.
- No changes to global connection retry thresholds.
- No changes to workspace persistence or hydration internals.

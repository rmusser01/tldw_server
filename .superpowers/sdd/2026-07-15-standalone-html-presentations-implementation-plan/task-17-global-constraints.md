# Task 17 binding global constraints

Source: `Docs/superpowers/specs/2026-07-15-standalone-html-presentations-design.md`, the approved Task17 plan, `PRODUCT.md`, `DESIGN.md`, and controller/reviewer preflight rulings.

1. V1 has no execution or preview surface. Standalone source is executable, untrusted, opaque text only. It may be generated, stored, versioned, edited inertly, validated structurally, downloaded, and exported as an attachment; it is never sanitized into safety, rendered, previewed, interpreted, imported, navigated to, or executed inside tldw.

2. Backend integration must use the real owner-scoped Slides database, real Jobs database and manager, real standalone generation service and validator pool, real HTTP router, and real `process_standalone_html_generation_job`. Mock only source and provider adapters for prompt, chat, media, notes, and RAG. Verify the owner Jobs envelope, acquire/drive/finalize/poll lifecycle, one allowed provider call per normal attempt, completed replay without a second provider call, and default-off rejection without loss of saved-document readability.

3. Verify legacy source-free filtering and opt-in list/search, exact detail, version history/content, explicit strong-ETag save, exact attachment export, database reopen, and no `text/html` response. Existing structured routes retain their weak ETags and synchronous generation semantics; standalone uses strong ETags and asynchronous Jobs.

4. Chromium owns both planned E2E specs. `standalone-html-firefox` and `standalone-html-webkit` select only `presentation-studio-standalone-html.security.spec.ts`, use their native desktop devices, and set `retries: 0`. `--list` must prove no accidental suite multiplication. Test/project/replay IDs are unique by engine plus run entropy.

5. Security tests install context-level observability before any navigation for requests, pages/popups, service workers, dedicated/shared workers, and the relevant application sinks. Instrument only bounded, source-correlated behavior; do not globally disable `innerHTML`, `eval`, `Worker`, or other native APIs and then claim product safety.

6. Use a unique validator-accepted source sentinel that mutates a global only if executed. Prove the sentinel stays inert and does not reach execution, navigation, resource, popup, worker, service-worker, DOM-HTML, URL, or storage sinks. Use a separate corrupt/mock detail URL payload to prove the client rejects source before adoption without depending on the execution sentinel.

7. The hung outline test intercepts only the application-owned standalone outline worker URL and leaves Monaco workers intact. Monaco navigation tests first assert a real `.monaco-editor` is mounted; fallback editor behavior is covered separately. No global Monaco service/provider mutation is permitted.

8. A separate real protocol server on `127.0.0.1` receives requests from the WebUI on `localhost`. It, rather than `route.fulfill`, provides evidence for real CORS preflight, response header, CSP, direct source-response, save, and attachment behavior. CSP must remain unchanged.

9. Download evidence permits exactly one source-bearing Blob URL, created from a Blob whose exact type is `application/octet-stream`, assigned only to a temporary anchor with the fixed filename `presentation.html`, no opener/target behavior, anchor removal, and URL revocation no later than one second. No HTML response is rendered or opened.

10. Browser coverage includes generate, Stop/Resume, edit, trusted safe outline, explicit save, lost response, all conflict choices, reopen, attachment download, keyboard and mobile flow, 44px targets, visible focus, no horizontal overflow, same-principal pagehide/Back restoration, expired/other-principal clearing, bfcache account switch, malformed and hung outline work, and URL/modifier-click/context-menu inertness.

11. Documentation must cover exact capabilities/errors/limits/headers without embedding executable sample documents. It must explain default-off generation; closed adapters and exact provider tuple allowlisting; server-owned HMAC key source and rotation; egress kill; source-free logs/status/errors; per-user isolation; worker/reconciler health; 24-hour generation-input and 30-day receipt retention; 32-day key retirement; at-least-once provider-call risk on precommit crashes; and saved-document readability after generation is disabled.

12. Rollout documentation requires schema v2, backup-first forward migration, old-binary incompatibility, drain-first rollback, no database downgrade, guarded MCP WebSocket launch, omission of Slides tools on unguarded WebSockets, extension metadata-only WebUI handoff, and the human-requester Change summary merge caveat.

13. The PRD change is narrowly scoped: standalone JavaScript is allowed only as opaque text for generate/store/edit/version/download. The established product prohibition on arbitrary execution remains absolute everywhere in tldw.

14. Tests are complete and RED before Playwright config, product docs, release docs, or core README changes. After three failed attempts at one root cause, stop and report. A product gap proven by a real test requires controller approval before editing any runtime path not listed in Task17.

15. The original scope is the thirteen Task17 plan paths plus
`task-17-brief.md`, `task-17-global-constraints.md`, and `task-17-report.md`.
After genuine browser/unit RED, the controller additionally authorized only:
`route-leave-prompt.tsx` and `router-utils.tsx`; `PresentationStudioNew.tsx`,
`PresentationStudioPage.tsx`, `StandaloneHtmlGenerationForm.tsx`,
`StandaloneHtmlWorkspace.tsx`, `useStandaloneHtmlGeneration.ts`, and
`standalone-html-outline-client.ts`; and each path's existing directly
associated regression test. The deviations fix Next/runtime import isolation,
authority and bfcache lifecycle fencing, same-tab generation authority,
keyboard tab behavior, and exact outline-worker identity. They do not broaden
the product feature. The controller explicitly superseded the plan's Backlog
step: make no Backlog calls or edits. Preserve the protected
`apps/packages/ui/node_modules/antd` and two Watchlist templates, use
`apply_patch`, stage explicitly, commit only after controller audit, do not
install, and do not push.

16. The security harness may observe native direct-eval execution only through
the sentinel property and surrounding code/DOM/worker/network sinks. It must
not replace `window.eval` with a Proxy because that converts direct eval to
indirect eval and changes Next/webpack semantics. This observability limitation
must be recorded rather than hidden by semantics-changing instrumentation.

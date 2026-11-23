---
id: TASK-15620
title: Quick-launch serves a production WebUI build by default
status: In Progress
labels:
- webui
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
quick-launch.sh/.ps1 start the WebUI with next dev (Turbopack over the shared UI package). Measured on dev 2aa1f81c37 the next-server dev process holds 5.4-7.5 GB RSS after three pages and grows per page; a production build serves the same UI in ~150 MB. On an ordinary laptop the dev server alone exhausts RAM, matching the report of the machine freezing while the app is left open. Make the launchers build the WebUI once (rebuild only when the source revision or the baked API origin changes) and serve it with next start in quickstart networking, bootstrapping single-user auth the way the Docker quickstart does; keep the dev server behind an opt-in.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 quick-launch.sh and quick-launch.ps1 serve a production WebUI build by default and start next dev only when TLDW_WEBUI_DEV=1
- [x] #2 The build runs only when no build exists, the git revision changed, or the API origin/URL baked into the build changed, with a clear first-launch message
- [x] #3 A first-time browser connects with no manual key step (quickstart networking + runtime single-user auth), and a user-set NEXT_PUBLIC_API_URL still works
- [x] #4 Launcher contract tests and onboarding docs tests pass; docs describe the new default and opt-in
- [x] #5 Real launch verified: /chat served, health reachable via the WebUI origin, RSS recorded, second launch does not rebuild, dev opt-in starts next dev
- [x] #6 Review fixes for #3574: Windows-safe Next spawn, frontend .env override, PowerShell env restore, sign-in opt-out from env file, loopback cookie defaults, /health wait, apps/ tree rebuild key, and their tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: not required. Searched Docs/ADR/README.md (no launcher/WebUI-serving decision). This is a bug fix in the no-Docker launchers that reuses the existing quickstart networking and runtime-auth contract the Docker WebUI already follows (Dockerfile.webui, runtime-auth-policy.ts); no new durable architecture rule.
Design: default path runs build:prod (quickstart networking) with TLDW_INTERNAL_API_ORIGIN=http://<api host>:<port>, then 'bun run start' (next start) bound to 127.0.0.1 with AUTH_MODE/SINGLE_USER_API_KEY (read from the env file via python-dotenv, as the backend does; never passed to the build) and TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=1, which is the Docker quickstart contract (runtime-auth-policy.ts cookie session). A user-set NEXT_PUBLIC_API_URL keeps advanced networking via build:dev (still an optimized next build) bound to 0.0.0.0. TLDW_WEBUI_DEV=1 keeps the old next dev path. Rebuild key = '<build script> <api url> <git HEAD>' written to .next/quick-launch-build-key after a successful build. next start chosen over the standalone server.js: it runs from the checkout with no copy of .next/static/public into the standalone tree and needs only bun (bun supplies node for next's shebang, as for next dev); its standalone warning is cosmetic.
TDD: red = 5 failed (launcher ran 'run dev -- -p 8097' instead of build:prod + start; ps1 lacked TLDW_WEBUI_DEV); green = 40 passed (launcher + onboarding docs tests). Sweep: tests/Docs + Utils launcher/release helper + lint env-vars ratchet + media_db repo reference guards = 321 passed.
Live verification (ports 8027/8097, dev 2aa1f81c37): first launch built in ~38 s, /chat 200 at +41 s; /chat, /settings/data, /document-workspace 200; /health via WebUI origin 200; runtime-config available=true; session bootstrap 200 sets HttpOnly cookie; /api/v1/auth/me via WebUI 200 with cookie only, 401 without. next-server RSS 182,752 KB after the three pages, 201,280 KB after serving all 1,889 static assets (71 MB) + 6 pages (vs 5.4-7.5 GB for next dev). Second launch: 'Reusing the existing WebUI build', /chat 200 at +3 s. TLDW_WEBUI_DEV=1 ran 'next dev -p 8097' (Ready in 6.2 s) and left the production build intact (.next/dev). NEXT_PUBLIC_API_URL override rebuilt with build:dev and served /chat in advanced mode. All processes stopped by PID; .next, logs and runtime files removed. quick-launch.ps1 not executed (no PowerShell on this machine).
PR: https://github.com/rmusser01/tldw_server/pull/3574 (base dev).
Review round 1 (#3574, verdict 'with fixes'). (1) build-with-profile.mjs ran execFileSync('next.cmd') on win32 without a shell (Node EINVAL since CVE-2024-27980; bun may not create .cmd shims). It now runs process.execPath with require.resolve('next/dist/bin/next') (resolves through bun's apps/node_modules/.bun layout). getNextBinary had no other callers; the extension has its own script. vitest red 'getNextBuildCommand is not a function' -> green 13 passed (build-profile-resolver + frontend-dev-config); eslint clean. A real build:prod through the launcher used the new spawn. (2) Reproduced for real: previous launcher + .env.local copied from the example -> 'Invalid WebUI networking config: quickstart mode must not set NEXT_PUBLIC_API_URL...', build:prod exit 1. Launchers now read NEXT_PUBLIC_API_URL from .env.production.local/.env.local/.env.production/.env (Next's order), switch to advanced networking and name the file; real run built build:dev and served /chat. (3) .ps1 saves and restores every $env: it sets (run_webui and the API window start) in finally; null restores unset. (4) AUTH_MODE, SINGLE_USER_API_KEY, SINGLE_USER_SESSION_COOKIE_NAME (the name runtime-auth-policy.ts reads) and TLDW_WEBUI_EXPOSE_RUNTIME_AUTH come from the shell, else the env file, else the default (1 for the flag); 0 opts out with a printed note. (5) The backend sets Secure on both cookies (single_user_session.py:139, csrf_protection.py:248; SESSION_COOKIE_SECURE defaults True). Chromium keeps them over http://127.0.0.1. WebKit was not testable: the only cached build is webkit-2248, while Playwright 1.65.1 expects 2388, and newPage fails with "Protocol error (Console.enable): 'Console' domain was not found". No browsers were downloaded. So all mode now mirrors the Docker overlay on loopback hosts only: SESSION_COOKIE_SECURE=0 and CSRF_ENABLED=1, explicit shell or env-file values win. webui mode prints what to set. A real run's Set-Cookie headers have no Secure, and Chromium signs in. (6) The new fake-bun + fake-venv-python sandbox covers all mode as well: failed build, TLDW_HOST=0.0.0.0, shell-over-file precedence, opt-out, dev/override env with no key or flag, cookie defaults, and an API that crashes or never turns healthy. pwsh is not installed, so .ps1 coverage is a string contract only. (7) The rebuild key uses git rev-parse HEAD:apps. (8-13) docs/usage cover zip updates and the sign-in scope; the failed-build hint, the standalone-warning line, and the missing-venv explanation are printed. TLDW_API_START_DELAY is now the max /health wait (default 120 s), replacing the fixed sleep. Python tests: red 19 failed / 32 passed -> green 51 passed; sweep tests/Docs + launcher + release helper + env-vars ratchet + media_db guards + CI licensing policy = 352 passed.
Real runs, review round 1 (ports 8027/8097). B (first, all mode): /health wait 16 s, build ~33 s, /chat at +49 s; /chat, /settings/data, /document-workspace 200; proxied /health 200; runtime-config available; Set-Cookie with no Secure; auth/me 200 with cookie, 401 without; Chromium auto sign-in OK (authMe 200, no connect prompt); next-server RSS 199,312 KB. C (second): 'Reusing the existing WebUI build', /chat at +18 s (WebUI starts after API health), first proxied /health 200, no ECONNREFUSED. D: TLDW_WEBUI_DEV=1 ran next dev (Ready 1.8 s); the prod build was kept. E: .env.local override ran build:dev, start -H 0.0.0.0, /chat 200, advanced. Everything stopped by PID; .next, .env.local, logs and runtime files removed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
quick-launch.sh/.ps1 build the WebUI once (build:prod, quickstart networking) and serve it with next start on 127.0.0.1, with Docker-quickstart runtime sign-in configured from the shell or the env file. TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=0 opts out. They rebuild only when the build script, the API address, or the git tree of apps/ changes. A NEXT_PUBLIC_API_URL in the shell or in apps/tldw-frontend/.env* keeps advanced networking via build:dev; TLDW_WEBUI_DEV=1 keeps next dev. All mode waits for the API's /health and, on loopback, mirrors the Docker overlay cookie settings. build-with-profile.mjs runs Next's CLI entry with process.execPath so Windows can build. Measured WebUI RSS ~178-199 MB vs 5.4-7.5 GB for next dev. The PowerShell script has not been run (no pwsh here): it needs one real Windows run before merge.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

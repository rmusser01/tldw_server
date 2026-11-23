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
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: not required. Searched Docs/ADR/README.md (no launcher/WebUI-serving decision). This is a bug fix in the no-Docker launchers that reuses the existing quickstart networking and runtime-auth contract the Docker WebUI already follows (Dockerfile.webui, runtime-auth-policy.ts); no new durable architecture rule.
Design: default path runs build:prod (quickstart networking) with TLDW_INTERNAL_API_ORIGIN=http://<api host>:<port>, then 'bun run start' (next start) bound to 127.0.0.1 with AUTH_MODE/SINGLE_USER_API_KEY (read from the env file via python-dotenv, as the backend does; never passed to the build) and TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=1, which is the Docker quickstart contract (runtime-auth-policy.ts cookie session). A user-set NEXT_PUBLIC_API_URL keeps advanced networking via build:dev (still an optimized next build) bound to 0.0.0.0. TLDW_WEBUI_DEV=1 keeps the old next dev path. Rebuild key = '<build script> <api url> <git HEAD>' written to .next/quick-launch-build-key after a successful build. next start chosen over the standalone server.js: it runs from the checkout with no copy of .next/static/public into the standalone tree and needs only bun (bun supplies node for next's shebang, as for next dev); its standalone warning is cosmetic.
TDD: red = 5 failed (launcher ran 'run dev -- -p 8097' instead of build:prod + start; ps1 lacked TLDW_WEBUI_DEV); green = 40 passed (launcher + onboarding docs tests). Sweep: tests/Docs + Utils launcher/release helper + lint env-vars ratchet + media_db repo reference guards = 321 passed.
Live verification (ports 8027/8097, dev 2aa1f81c37): first launch built in ~38 s, /chat 200 at +41 s; /chat, /settings/data, /document-workspace 200; /health via WebUI origin 200; runtime-config available=true; session bootstrap 200 sets HttpOnly cookie; /api/v1/auth/me via WebUI 200 with cookie only, 401 without. next-server RSS 182,752 KB after the three pages, 201,280 KB after serving all 1,889 static assets (71 MB) + 6 pages (vs 5.4-7.5 GB for next dev). Second launch: 'Reusing the existing WebUI build', /chat 200 at +3 s. TLDW_WEBUI_DEV=1 ran 'next dev -p 8097' (Ready in 6.2 s) and left the production build intact (.next/dev). NEXT_PUBLIC_API_URL override rebuilt with build:dev and served /chat in advanced mode. All processes stopped by PID; .next, logs and runtime files removed. quick-launch.ps1 not executed (no PowerShell on this machine).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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

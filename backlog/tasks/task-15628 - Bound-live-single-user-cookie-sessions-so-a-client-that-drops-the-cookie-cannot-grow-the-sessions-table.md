---
id: TASK-15628
title: Bound live single-user cookie sessions so a client that drops the cookie cannot
  grow the sessions table
status: Done
labels:
- authnz
- security
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing TASK-15627. POST /api/v1/auth/single-user/session (create_single_user_cookie_session, tldw_Server_API/app/api/v1/endpoints/auth.py) mints a new session through mint_single_user_session -> SessionManager.create_session whenever the request carries no valid session cookie. A browser that drops the cookie (Safari with a Secure cookie over plain http, cookie blocking, a WebUI/API SINGLE_USER_SESSION_COOKIE_NAME mismatch) therefore gets a new 30-day row (SINGLE_USER_SESSION_EXPIRE_DAYS=30, device_id single-user-cookie:v1) on every call. Nothing caps live sessions per user: SessionManager has no max-sessions concept. The request rate is not a useful bound either. With Resource Governor enabled (the default), RGSimpleMiddleware maps /api/v1/auth/* to authnz.default (300 rpm per entity, resource_governor_policies.yaml) and sets request.state.rg_policy_id, so check_auth_rate_limit's 30-per-60-s fallback returns early (_enforce_auth_deps_ingress_guard in auth_deps.py also returns early for a single-user principal); the effective limit is about 300 mints per minute for the single user. Only with RG disabled does the 30-per-60-s per-client fallback apply. The hourly cleanup (authnz_scheduler, SESSION_CLEANUP_INTERVAL_HOURS=1; AuthnzSessionsRepo.cleanup_expired_sessions) deletes only rows expired more than 1 day ago or revoked more than 7 days ago, so these rows stay at least 31 days. Live in Safari the WebUI produced 34 rows in about 55 s before TASK-15627 bounded the client (about 37 per minute). Options to decide: (a) on mint, revoke and delete the oldest single-user-cookie sessions beyond a small cap (needs a Postgres and SQLite repo query, Redis cache invalidation, and a decision on N so a second browser profile is not signed out); (b) mint with a short expiry and extend to 30 days on the first authenticated use, so cookies that never come back expire quickly (still kept until cleanup's 1-day grace).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A client that never returns the cookie cannot leave more than a small fixed number of live single-user cookie sessions
- [x] #2 Tested on SQLite and Postgres
- [x] #3 A legitimate second browser profile keeps its session, or the chosen trade-off is recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Corrected 2026-11-23 after review: the first description said check_auth_rate_limit bounds minting at 30 per 60 s (~43,000 rows/day). By reading: with RG enabled the RG policy authnz.default (300 rpm per entity) applies instead and the auth fallback is skipped; the fallback applies only with RG disabled.
Design: Docs/Design/2026-11-23-session-cap-alert-lock-design.md section 1; plan: Docs/superpowers/plans/2026-11-23-session-cap.md; ADR-099. Verification 2026-11-24: 4478 passed, 16 skipped, 0 failed over the session-touching backend sweep plus AuthNZ_SQLite, AuthNZ_Unit, lint and Docs (PostgreSQL cap tests ran and passed); vitest 1577 (ui) and 500 passed/12 skipped (frontend). Bandit: environment skip, not installed in the venv, runs in CI. Ruff: new files clean; session_manager.py import order fixed. Limits: SSO/OIDC callback sign-ins carry no install id and share the no-id eviction group; admin E2E bootstrap (bootstrap_admin_e2e_jwt_session) creates id-less sessions, now capped per principal; only verify_jwt_and_fetch_user and /auth/refresh answer session_replaced, other revoked-token sites keep the generic 401, single-user cookie sessions get a plain 401 and re-mint; MFA challenge consumed after the code verifies, row created only on MFA completion, single-user /login creates no row; each evicted JWT session costs up to two token_blacklist writes after commit (a large pre-upgrade backlog pays once on the first capped sign-in); on PostgreSQL a session logged out at the same instant as an eviction can be stamped session_cap, the upper bound always holds. Pending CI: non-required Frontend E2E and admin-ui real-backend comparison with dev; E2E session-budget decision (brief Step 7) is the controller's. Follow-up: TASK-15642 (token_blacklist purge; TASK-15635 was claimed by another open PR).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Live sessions per user are capped (default 4, MAX_SESSIONS_PER_USER) on SQLite and PostgreSQL. Each sign-in goes through one per-user-serialized capped insert that revokes the least recently active live session, same install first (id-less sign-ins are one group). Shared live predicate COALESCE(refresh_expires_at, expires_at); evicted JWT clients get 401 session_replaced; MFA row created at completion; single-user /login creates no row. Tests on SQLite and PostgreSQL; ADR-099; follow-up TASK-15642.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

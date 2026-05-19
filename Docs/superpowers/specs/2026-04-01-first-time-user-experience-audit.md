# First-Time User Experience Audit

**Date:** 2026-04-01
**Scope:** Complete UX audit of first-run experience for self-hosted (single/multi-user), hosted SaaS, and admin personas
**Method:** Code-level analysis of setup flows, error states, onboarding, and documentation

---

## Executive Summary

**42 issues identified** across 8 journey stages, covering 3 user personas. The core product is powerful and well-engineered, but the first-time experience is fragmented:

- Docker quickstart works but is opaque — users don't know their credentials or next steps
- No LLM provider setup in any UI — users must edit `.env` files manually
- Multi-user mode can lock users out with no recovery path
- Admin-UI is undiscoverable from the standard setup flow
- Error messages are backend-facing, not user-facing
- Documentation has 6 overlapping setup guides with no decision tree

**Top 5 Critical Fixes:**
1. Provider setup UI in WebUI (#9, #18, #32)
2. Multi-user first-admin creation wizard (#4)
3. Post-quickstart "what next" guide (#2, #28)
4. Distinct error messages for connection/auth/CORS (#10, #16, #18)
5. Surface API key after Docker quickstart (#1, #8)

---

## All 42 Issues by Journey Stage

### Journey 1: Installation & First Run (7 issues)

| # | Issue | Persona | Severity |
|---|-------|---------|----------|
| 1 | Docker quickstart generates API key silently — never shown | Self-host | HIGH |
| 2 | No "what to do next" printed after quickstart | Self-host | HIGH |
| 3 | Setup wizard `/setup` hidden — not in any user docs | Self-host | HIGH |
| 4 | Multi-user mode: no admin creation wizard → locked out | Admin | CRITICAL |
| 5 | `.env.example` 150+ lines, no required/optional grouping | All | MEDIUM |
| 6 | Secret generation requires Python on host | Self-host | MEDIUM |
| 7 | Config precedence (env > .env > config.txt) confusing | Self-host | MEDIUM |

### Journey 2: Web UI Onboarding (6 issues)

| # | Issue | Persona | Severity |
|---|-------|---------|----------|
| 8 | OnboardingConnectForm asks for key user doesn't have | Self-host | HIGH |
| 9 | No provider setup in Web UI — must edit .env manually | All | CRITICAL |
| 10 | Generic "Server unreachable" — same for CORS/down/wrong URL | All | HIGH |
| 11 | No "change API URL" after initial setup | Self-host | MEDIUM |
| 12 | Onboarding identical for single/multi — should adapt | All | MEDIUM |
| 13 | No progress text during auth validation ("Connecting...") | All | LOW |

### Journey 3: Extension Setup (4 issues)

| # | Issue | Persona | Severity |
|---|-------|---------|----------|
| 14 | No localhost auto-detection for server URL | Self-host | MEDIUM |
| 15 | Extension doesn't support multi-user auth (JWT) | Admin | MEDIUM |
| 16 | Generic "connection failed" — no specific error detail | All | MEDIUM |
| 17 | No docs link from extension error state | All | LOW |

### Journey 4: First Chat Attempt (4 issues)

| # | Issue | Persona | Severity |
|---|-------|---------|----------|
| 18 | No LLM providers → cryptic "provider resolution failed" | All | CRITICAL |
| 19 | No API key validation before first use | All | HIGH |
| 20 | Provider fallback chain not explained to user | All | MEDIUM |
| 21 | No indication which model is responding in chat | Hosted | MEDIUM |

### Journey 5: First Media Ingestion (6 issues)

| # | Issue | Persona | Severity |
|---|-------|---------|----------|
| 22 | No FFmpeg → video upload silently fails | Self-host | HIGH |
| 23 | No multi-stage progress (Upload → Transcribe → Index) | All | HIGH |
| 24 | File size limits not shown before upload | All | MEDIUM |
| 25 | Large file upload not resumable | All | MEDIUM |
| 26 | No first-ingest tutorial ("Try pasting a YouTube URL") | Hosted | MEDIUM |
| 27 | Error messages from pipeline are backend-technical | All | MEDIUM |

### Journey 6: Admin-UI Setup (6 issues)

| # | Issue | Persona | Severity |
|---|-------|---------|----------|
| 28 | Admin-UI not in standard quickstart output | Admin | HIGH |
| 29 | Admin-UI requires separate login from WebUI | Admin | MEDIUM |
| 30 | No "forgot password" for admin accounts | Admin | HIGH |
| 31 | Empty dashboard with no guided first-run experience | Admin | MEDIUM |
| 32 | Provider/LLM management only via .env, not admin-UI | Admin | HIGH |
| 33 | No single → multi-user migration guide | Admin | MEDIUM |

### Journey 7: Documentation (5 issues)

| # | Issue | Persona | Severity |
|---|-------|---------|----------|
| 34 | 6 setup guides with overlap — no decision tree | All | MEDIUM |
| 35 | Troubleshooting table only 8 entries (30+ needed) | All | MEDIUM |
| 36 | No in-app help links from error screens | All | MEDIUM |
| 37 | No architecture diagram showing component relationships | Self-host | LOW |
| 38 | No `make check` sanity validation command | Self-host | LOW |

### Journey 8: Cross-Cutting (4 issues)

| # | Issue | Persona | Severity |
|---|-------|---------|----------|
| 39 | CORS defaults too restrictive for localhost dev | Self-host | MEDIUM |
| 40 | WebUI loads before API ready → connection error | All | LOW |
| 41 | Container restart loop on auth init failure | Self-host | HIGH |
| 42 | No system status page in WebUI | All | MEDIUM |

---

## Severity Summary

| Severity | Count | Examples |
|----------|-------|---------|
| CRITICAL | 3 | No provider UI (#9), multi-user lockout (#4), chat fails silently (#18) |
| HIGH | 12 | Hidden API key (#1), FFmpeg silent fail (#22), no admin recovery (#30) |
| MEDIUM | 21 | CORS issues (#39), no progress (#23), overlapping docs (#34) |
| LOW | 6 | Architecture diagram (#37), sanity check (#38) |

---

## Recommended Implementation Phases

### Phase 1: Unblock First Chat (Critical, 1-2 weeks)
- #9/#18/#32: Provider setup UI (WebUI + admin-UI) — key entry + "Test" button
- #4: Multi-user admin creation at `/setup` before auth gating
- #1/#2/#8: Print API key + next steps after `make quickstart`

### Phase 2: Error Clarity (High, 1-2 weeks)
- #10/#16: Distinct error messages (CORS vs connection vs auth)
- #22: FFmpeg detection → disable video button with hint
- #41: Clear error message on auth init failure (not restart loop)
- #19: API key validation on save, not first chat

### Phase 3: Guided Onboarding (Medium, 2-3 weeks)
- #3/#28: Surface /setup wizard + admin-UI URL in quickstart output
- #12: Adapt onboarding wizard per auth mode
- #23: Multi-stage ingestion progress (upload → transcribe → index)
- #26: First-ingest tutorial with YouTube URL example
- #31: Admin dashboard first-run checklist

### Phase 4: Documentation & Polish (Medium-Low, 2-3 weeks)
- #34: Single decision-tree setup guide replacing 6 overlapping docs
- #35: Expand troubleshooting to 30+ entries
- #36: Add "Learn more" / "How to fix" links in all error screens
- #33: Single → multi-user migration guide
- #5: Restructure .env.example with REQUIRED/OPTIONAL sections

### Phase 5: Nice-to-Haves (Low, ongoing)
- #14: Extension localhost auto-discovery
- #15: Extension multi-user JWT support
- #25: Resumable large file uploads
- #37: Architecture diagram
- #38: `make check` sanity command
- #42: WebUI system status page

---
id: TASK-172
title: Resolve VN Play visual directives into scene assets
status: Done
assignee: []
created_date: '2026-05-09 17:48'
labels:
  - vn-play
  - backend
  - api
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1432'
  - 'https://github.com/rmusser01/tldw_server/issues/1426'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
documentation:
  - Docs/API-related/VN_PLAY_API.md
  - Docs/API-related/VN_ASSET_PACKS_API.md
  - Docs/superpowers/specs/2026-05-01-vn-play-runtime-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1426: make VN Play turn completion resolve model/runtime visual directives against the approved VN asset pack manifest and expose backend-owned scene asset payloads for custom frontends and the bundled WebUI. Keep the backend as source of truth for approved-only filtering, generated-file content URLs, deterministic variants, warnings, and auditable events.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A VN Play turn with valid visual directives updates returned scene state with approved renderable asset payloads.
- [x] #2 Rejected or unresolved directives append auditable rejection events and stable scene warnings without failing the text turn.
- [x] #3 Scene replay and checkpoint restore preserve resolved background depth and sprite state.
- [x] #4 Custom frontends can consume visual state from VN Play API responses without calling VN asset-pack internals directly.
- [x] #5 Backend tests cover successful resolution missing assets unapproved assets deterministic variants replay/checkpoint behavior and API response shape.
- [x] #6 Existing turn idempotency stale-scene and in-progress behavior remain unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan file: Docs/superpowers/plans/2026-05-09-vn-play-visual-directives-runtime-implementation-plan.md

1. Normalize VN Play visual resolver asset-type aliases so approved runtime manifests resolve singular and plural directive forms.
2. Replay visual_directive_applied events into durable scene state IDs and sprite item payloads.
3. Apply model visual directives during turn completion by appending requested/applied/rejected events and merging applied assets into scene_state_changed payloads.
4. Enrich VN Play API scene responses with background depth and active_sprites payloads from the approved manifest.
5. Update API docs and run focused VN Play tests, diff checks, and Bandit on touched backend code.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added approved-manifest alias support for singular/plural VN Play visual directive asset types.
- Added turn-time directive resolution that records requested/applied/rejected events and keeps accepted narrative turns completed when visual resolution fails.
- Added scene replay and API enrichment so frontends can render `background`, `depth`, and `active_sprites` from VN Play responses.
- Rejected directives are warning-only and auditable; resolver exceptions are converted into `resolver_error` rejection warnings.
- Enriched `active_sprites` is derived from the current approved manifest and does not fall back to stale sprite payloads when an item is no longer approved.
<!-- SECTION:NOTES:END -->

## Verification

<!-- SECTION:VERIFICATION:BEGIN -->
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q` - `47 passed, 5 warnings`
- `git diff --check` - passed
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/VN_Play tldw_Server_API/app/api/v1/endpoints/vn_play.py tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py -f json -o /tmp/bandit_vn_play_visual_directives.json` - passed with 0 findings
<!-- SECTION:VERIFICATION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved VN Play visual directives against approved VN asset-pack manifests during turn completion. The backend now emits requested applied and rejected visual directive events, persists replayable scene asset state, enriches VN Play API scene responses with render-ready approved asset payloads, keeps visual misses warning-only, and documents the runtime contract for custom frontends.
<!-- SECTION:FINAL_SUMMARY:END -->

---
id: TASK-230
title: Design full VN platform API
status: Done
assignee: []
created_date: '2026-05-10 02:37'
updated_date: '2026-05-10 02:42'
labels:
  - vn
  - api
  - design
  - docs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
  - 'https://github.com/rmusser01/tldw_server/issues/1486'
documentation:
  - Docs/API-related/VN_PLAY_API.md
  - Docs/API-related/VN_ASSET_PACKS_API.md
  - Docs/superpowers/specs/2026-05-01-vn-play-runtime-design.md
  - Docs/superpowers/specs/2026-04-24-vn-asset-packs-design.md
  - Docs/superpowers/specs/2026-04-25-vn-pack-portability-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved full VN platform API design spec for the character/persona CYOA-VN effort. Scope is design/documentation only: define the canonical backend-owned resource-first REST API under `/api/v1/vn/vn-*`, covering VN assets, scripts, play/runtime, policy, audio, capabilities, cross-cutting contracts, examples, migration from existing routes, and vNext boundaries. The spec must preserve the user-approved decisions: V1-complete plus vNext sections, Story authoring in V1, canonical JSON opcode scripts with optional future DSL compiler, immutable published script versions pinned to a single asset-pack manifest snapshot, `freeform`/`story`/`scripted_story` session modes, script-specific runtime endpoints, per-session save slots, configurable policy profiles, admin-managed generation profiles, VN TTS namespace, offline-only image assets in V1, and canonical `/api/v1/vn/vn-*` routes with old `/vn-assets` and `/vn-play` routes documented as superseded rather than aliased.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents the canonical `/api/v1/vn/vn-*` namespace, auth, versioning, route migration, response/error conventions, idempotency, pagination, jobs, content endpoint rules, and ownership boundaries.
- [x] #2 Spec defines V1 endpoint inventories and representative request/response/error examples for vn-capabilities, vn-assets, vn-scripts, vn-play, vn-policy, and vn-audio.
- [x] #3 Spec captures authored script V1 semantics: mutable drafts with optimistic revisions, validation and diagnostics, immutable published versions, single primary asset pack with pinned manifest snapshot, canonical JSON opcodes, typed variables, structured conditions, seeded random replay, model generation persistence/regeneration, and vNext DSL/patch/collaboration boundaries.
- [x] #4 Spec captures runtime V1 semantics: freeform, story, and scripted_story; Story start; Freeform/model Story turns; script-specific advance/choice/regenerate/state/debug endpoints; checkpoints; per-session save slots; branch navigation/restore; synchronous runtime model calls with persisted failure state.
- [x] #5 Spec captures configurable VN policy and generation profile APIs plus VN TTS-only audio job APIs with optional output persistence and authenticated preview/content endpoints.
- [x] #6 Spec clearly marks vNext-only scope: realtime image generation, script/session portability, marketplace/sharing, multiplayer, subscriptions, full media timeline/lip sync, rich built-in gameplay systems, collaborative editing, and multiple asset packs per script version.
- [x] #7 Docs-only verification is recorded: markdown link/path sanity, git diff --check, and Bandit skip rationale for non-code changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the approved full VN platform API decisions in a single spec.
2. Include endpoint inventories and examples for vn-capabilities, vn-assets, vn-scripts, vn-play, vn-policy, and vn-audio.
3. Self-review against the approved brainstorming decisions and tighten inconsistencies.
4. Run docs-only verification, record Bandit skip rationale, then commit the spec and Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md with the canonical `/api/v1/vn/vn-*` API design. Self-review fix: changed end-to-end flow shorthand to canonical `/api/v1/vn/vn-*` paths and made model expansion from any authored scene explicit. Verification so far: referenced docs exist; git diff --check exits 0. Bandit is not applicable because the change is markdown/task metadata only.

Self-review follow-up: hardened route migration wording so deprecated aliases are not part of the V1 target API, and required VN image/archive uploads to pass through existing upload validation, storage registration, and generated-file tracking.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wrote Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md as the full backend-owned VN platform API design under `/api/v1/vn/vn-*`. The spec covers capabilities, assets, scripts, play/runtime, policy, VN audio, idempotency, ownership, Jobs linkage, content endpoints, route migration, examples, verification, vNext boundaries, and explicit risks. Verification: referenced documentation paths exist; git diff --check passed. Bandit was skipped because the touched scope is markdown and Backlog task metadata only.
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

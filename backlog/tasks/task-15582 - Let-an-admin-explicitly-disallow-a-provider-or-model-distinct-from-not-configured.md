---
id: TASK-15582
title: Let an admin explicitly disallow a provider or model, distinct from 'not configured'
status: In Progress
labels:
- llm
- byok
- policy
- admin
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up of TASK-15581, asked for by the owner on 2026-11-19. Today the provider override's allowed_models is an allowlist: a model left off it counts as explicitly disallowed (owner, 2026-11-19). The owner wants a distinct explicit "disallowed" state, so an admin can disallow a provider or model on purpose, as opposed to not having configured it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner decides the shape of an explicit disallowed state for a provider and for a model, and how it relates to is_enabled and allowed_models
- [x] #2 The owner decides whether a model left off allowed_models stays refused once the explicit state exists
- [ ] #3 The admin API and WebUI let an admin set the state the owner chose, enforcement follows it, with tests, and ADR-086 and the admin docs say so
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Owner's decision, 2026-11-21 (chose 'Server key only'). AC #1: the explicit provider disallow stays is_enabled=False, refused on every key; the explicit model disallow is a new denied_models list on the provider override, refused on every key and path with the new code model_denied (403), and it wins over allowed_models. AC #2: a model left off allowed_models is no longer refused on every key; it is only not offered on the server's key. It is refused (model_not_allowed) when the call runs on the server's key, a team or org key, or a user key sent to an operator's endpoint, and it runs on the user's own key at its vendor, as a provider with no server key does. Plan: Docs/superpowers/plans/2026-11-21-model-denylist.md.
Also done here, the four minors deferred from PR #3537's final review: the gateway runs the admin-override guard before model discovery, so a disabled provider gets no model-list request; the app's refusal handler answers with the code's own message; an LLM CLI group given no subcommand prints its help without loading the store; and two test gaps (a non-LLM CLI command run without --help, the standalone Slides branch of the app's refusal handler).
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

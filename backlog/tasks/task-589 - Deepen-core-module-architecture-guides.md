---
id: TASK-589
title: Deepen core module architecture guides
status: Done
assignee: []
created_date: '2026-06-02 00:18'
updated_date: '2026-06-02 00:26'
labels: []
dependencies:
  - TASK-588
documentation:
  - Docs/superpowers/specs/2026-06-01-core-module-readme-refresh-design.md
  - Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md
  - >-
    Docs/superpowers/plans/2026-06-02-core-module-architecture-guides-implementation-plan.md
  - >-
    Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2 follow-up to TASK-588. Review all 88 top-level tldw_Server_API/app/core modules as architecture-guide candidates, prioritize high-risk/high-complexity modules, expand only the modules that need deeper contributor guidance, and explicitly record sufficient modules without padding them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 High-risk and high-complexity core modules are reviewed first using source, endpoint, schema, config, and test evidence.
- [x] #2 Expanded architecture guides document concrete flows, boundaries, data/config surfaces, extension points, operational/security gotchas, and verification paths without speculative content.
- [x] #3 Modules that do not warrant expansion are recorded as sufficient with a concrete reason.
- [x] #4 Verification records changed files, Markdown/link sanity, placeholder scan, and docs-only Bandit skip or relevant security check.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-06-02-core-module-architecture-guides-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created after PR #2216 merged TASK-588 into dev. Worktree: .worktrees/core-architecture-guides on branch codex/core-architecture-guides from origin/dev merge commit c7b2e66400492614914e1a8f0e4abe939b031b64.

2026-06-02: Drafted Phase 2 implementation plan at Docs/superpowers/plans/2026-06-02-core-module-architecture-guides-implementation-plan.md. Plan reviews all 88 modules, prioritizes the 47 high-risk/high-complexity modules first, and records sufficient decisions for modules that do not need expansion.

2026-06-02: Stage 1 complete. Created Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md with 88 module rows. Verified 88 top-level app/core modules, no missing README files, Phase 1 priority counts of 47 high / 28 medium / 4 low / 9 sufficient, and a generated Phase 2 decision baseline of 47 expand / 28 tighten / 13 sufficient. git diff --check passed. Runtime code remains untouched.

2026-06-02: Stage 2 complete. Reviewed AuthNZ, Billing, DB_Management, Resource_Governance, Sandbox, Security, Sharing, Storage, Sync, Text2SQL, UserProfiles, and Workspaces. Expanded README architecture notes for Billing, Security, Sharing, Storage, Text2SQL, UserProfiles, and Workspaces; recorded AuthNZ, DB_Management, Resource_Governance, Sandbox, and Sync as sufficient after source/test review. Verification passed: placeholder scan returned no matches, README local-link sanity passed, Stage 2 referenced source/test paths exist, inventory has 12 reviewed rows, and git diff --check passed.

2026-06-02: Stage 3 complete. Reviewed Agent_Client_Protocol, Agent_Orchestration, Chat, Chat_Workflows, CodeGraph, Ingestion_Media_Processing, Ingestion_Sources, Jobs, MCP_unified, RAG, Scheduler, and Workflows. Expanded README architecture notes for Agent_Client_Protocol, Agent_Orchestration, Chat_Workflows, CodeGraph, Ingestion_Sources, Jobs, Scheduler, and Workflows; recorded Chat, Ingestion_Media_Processing, MCP_unified, and RAG as sufficient after source/test review. Verification passed: placeholder scan returned no matches, README local-link sanity passed, Stage 3 target README files exist, Stage 3 inventory rows are reviewed, and git diff --check passed.

2026-06-02: Stage 4 complete. Reviewed Audit, Chatbooks, Data_Tables, Embeddings, Evaluations, External_Sources, File_Artifacts, Governance, Infrastructure, Integrations, LLM_Calls, Metrics, Moderation, Monitoring, Notes, Personalization, Research_Workspace, Setup, Streaming, VN_Policy, Watchlists, WebSearch, and Web_Scraping. Expanded README architecture notes for Data_Tables, File_Artifacts, Governance, Integrations, Moderation, Personalization, Research_Workspace, Setup, Streaming, VN_Policy, and Watchlists; recorded Audit, Chatbooks, Embeddings, Evaluations, External_Sources, Infrastructure, LLM_Calls, Metrics, Monitoring, Notes, WebSearch, and Web_Scraping as sufficient after source/test review. Also rephrased one CodeGraph README absence-state line so the speculative placeholder scan remains clean. Verification passed: placeholder scan returned no matches, README local-link sanity passed, speculative placeholder scan returned no matches, Stage 4 inventory rows are reviewed with 47 total reviewed rows, and git diff --check passed.

2026-06-02: Stage 5 complete. Reviewed the remaining 41 medium, low, and previously sufficient modules: Audio, Audiobooks, Character_Chat, Chunking, Claims_Extraction, Collections, Flashcards, Image_Generation, Local_LLM, Logging, Meetings, Notes_Graph, Notifications, Persona, PrivilegeMaps, Prompt_Management, Prototype_Workspaces, RateLimiting, Reminders, Research, Search_and_Research, Skills, Slides, StudyPacks, StudySuggestions, TTS, Telegram, Templating, Third_Party, Tools, Usage, Utils, VN_Assets, VN_Platform, VN_Play, VN_Scripts, VoiceAssistant, WebClipper, Writing, config_sections, and deprecations. Recorded all as sufficient after README plus representative source/test review; no README changes were needed in this batch. Verification passed: all 88 inventory rows are reviewed, 41 representative Stage 5 source anchors exist, placeholder scans returned no matches, README local-link sanity passed, and git diff --check passed.

2026-06-02: Final verification complete after merging current origin/dev. Verified all 88 top-level core modules have README files, placeholder scans returned no matches, README local-link sanity passed, Phase 2 inventory reports 88 reviewed rows, git diff --check passed, and branch diff scope against origin/dev is 29 docs/backlog files only. Bandit skipped because TASK-589 changed Markdown documentation and Backlog records only; no Python or runtime source files were modified.

Final summary: Completed the Phase 2 architecture-guide pass for all 88 core modules. Expanded high-risk and cross-module guides where additional flow, state, security/operations, or extension-checklist detail was useful, and recorded source-backed sufficiency decisions for modules that did not warrant more README content.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

# Core Module README Refresh Design

## Context

The `tldw_Server_API/app/core` package contains 88 top-level module directories. At the start of this work, 48 top-level modules already had a `README.md` and 40 did not. Existing READMEs vary widely: some are useful contributor guides, some are long architecture references, and some are scaffolded placeholders.

The Stage 1 inventory at `Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md` verifies this baseline and records the initial red README coverage and placeholder checks.

This work improves contributor orientation without changing runtime code. It covers all immediate non-cache directories under `tldw_Server_API/app/core`, including lowercase helper packages such as `config_sections` and `deprecations`. Nested READMEs may be linked from parent modules, but this pass does not expand every nested package into its own guide.

Backlog tracking: `TASK-588`.

## Goals

- Ensure every top-level `tldw_Server_API/app/core/<module>` directory has a `README.md`.
- Replace placeholder or thin READMEs with source-informed contributor orientation.
- Preserve strong existing READMEs where they already help contributors, tightening only where needed.
- Make module ownership, entry points, dependencies, extension points, and tests easy to find.
- Set up a follow-up path for deeper architecture guides after the orientation pass lands.

## Non-Goals

- No application behavior changes.
- No broad source refactors.
- No generated API reference replacement.
- No exhaustive line-by-line documentation for every file.
- No nested README expansion beyond targeted parent-module links.

## Chosen Approach

Phase 1 uses a source-informed orientation pass across all 88 top-level core modules. Each README is based on the actual module contents, related API endpoints, schemas, configuration hooks, tests, and integration points visible in the repository.

Phase 2 is a follow-up deep-guide pass corresponding to the deeper architecture-guide approach. All 88 top-level core modules remain candidates for deeper documentation, but the work should be prioritized by risk and complexity. Broad or operationally sensitive modules such as `AuthNZ`, `Chat`, `DB_Management`, `Ingestion_Media_Processing`, `Jobs`, `LLM_Calls`, `MCP_unified`, `RAG`, `Scheduler`, and `Sync` should be evaluated first. Small modules should not be padded with speculative architecture content; their Phase 2 treatment can be a concise confirmation that the Phase 1 README is already deep enough.

## README Structure

Each README should use a consistent but lightweight structure:

1. Title and short purpose statement.
2. "Start Here" or "Responsibilities" section for quick contributor orientation.
3. "Module Map" section listing important files and subpackages.
4. "Primary Flows" or "How It Connects" section explaining data/control flow and adjacent modules.
5. "Configuration and Data" section when the module has env vars, config files, DB tables, external services, storage paths, or feature flags.
6. "Extension Points" section for common contributor tasks.
7. "Testing" section with focused test paths or verification guidance.
8. "Gotchas" or "Security Notes" when the module has notable risk, tenant scoping, credentials, file paths, sandboxing, or network behavior.

Small or simple modules can combine sections. Large modules can keep additional sections when they are already valuable. The README should be useful at a glance rather than forced into a rigid template.

## Content Rules

- Derive statements from source files, API endpoints, schemas, tests, and existing docs in the repo.
- Remove placeholder text such as "Replace placeholders with accurate details".
- Prefer concrete file paths and public symbols over vague descriptions.
- Keep endpoint and schema references concise; link to the owning file rather than duplicating complete API docs.
- Use plain Markdown and ASCII text unless the file already intentionally uses non-ASCII.
- Avoid stale promises, speculative roadmap detail, or marketing copy.
- Do not expose secrets, test credentials, or environment-specific local data.

## Implementation Strategy

1. Inventory all top-level core module directories and current README coverage.
2. Classify existing READMEs as keep/tighten, replace scaffold, or create missing.
3. Maintain an implementation inventory that records README status, source evidence inspected, related endpoint/schema/test paths, and whether the module should be a Phase 2 deep-guide candidate.
4. For each module, inspect representative source files, public exports, endpoint imports, schema imports, tests, and existing docs.
5. Write or refresh README content in small batches, prioritizing missing and placeholder files first.
6. Preserve useful existing long-form guides; add top-level orientation and links if needed.
7. Consider adding a short `tldw_Server_API/app/core/README.md` index if the inventory shows it would improve navigation; this is useful but not an acceptance criterion for the per-module pass.
8. Run README coverage checks and lightweight Markdown/link sanity checks.
9. Record verification and final summary on `TASK-588`.

## Verification

Minimum verification for Phase 1:

- Confirm every top-level core module directory has `README.md`.
- Search for obvious placeholder terms in core READMEs.
- Run a Markdown heading/link sanity check with local tooling when available, or a small local fallback check when project tooling is absent.
- Run a spelling or typo-oriented scan when local tooling exists; avoid network-dependent tooling for this docs-only pass.
- Record that Bandit is skipped for README-only changes, unless code changes become necessary.

## Follow-Up Deep Guide Pass

After Phase 1 lands, Phase 2 can deepen the module docs into fuller architecture guides. All 88 top-level modules are in scope for review, but the follow-up should not expand every module equally. It should prioritize modules where contributors need more detail to safely modify behavior: security-sensitive modules, complex orchestration modules, database-heavy modules, and modules with many API consumers.

The Phase 2 deliverable should be either expanded READMEs or separate deeper docs linked from concise READMEs, depending on how large each guide becomes. For simple modules, a Phase 2 decision can explicitly record that the concise README is sufficient and no longer guide is warranted.

## Open Decisions

- Phase 1 will keep README depth concise by default. If a module is too broad to explain clearly in one concise README, the README should point to existing deeper docs or mark it as a Phase 2 priority.
- Existing nested README files remain in place. Parent READMEs can reference them where useful.

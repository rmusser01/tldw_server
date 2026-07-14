# Service Prompt Inventory and Rollout Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the authoritative, reviewable list of internal prompts that may be user-customized and the exact domain migration worklist.

**Architecture:** Inventory prompt assets and hard-coded LLM instructions, classify every candidate against the approved eligibility rules, assign stable IDs only to eligible content-generation prompts, and record exact call sites. This is a documentation-first gate; it introduces no runtime scanner or speculative registry entries.

**Tech Stack:** ripgrep, existing YAML/Markdown prompt assets, Markdown design documentation, pytest collection for affected source references.

---

Before every commit below, satisfy the umbrella plan's mandatory per-commit gate.

## Task 1: Create the inventory skeleton

**Files:**

- Create: `Docs/Design/service-prompt-inventory.md`
- Reference: `Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md`
- Reference: `tldw_Server_API/app/core/Utils/prompt_loader.py`

- [ ] Create a Backlog implementation task referencing `TASK-12956` and this plan before editing repository files.
- [ ] Add the document purpose, review date, reviewer, eligibility rubric, and the required columns: `candidate`, `source`, `runtime consumer/call sites`, `data owner`, `workflow owner`, `explicit-field literal/template semantics`, `variables`, `assembly`, `locked fragments/visibility`, `output dependency`, `contract sensitivity`, `decision`, `reason`, `service_prompt_id`, `parts`, and `rollout slice`.
- [ ] State that missing rows are a release blocker and that the matrix, not filename location, controls eligibility.
- [ ] Commit: `docs: scaffold service prompt inventory (<task-id>)`.

## Task 2: Inventory file-backed prompts

**Files:**

- Modify: `Docs/Design/service-prompt-inventory.md`
- Inspect: `tldw_Server_API/Config_Files/Prompts/*`
- Inspect: `tldw_Server_API/app/core/Utils/prompt_loader.py`

- [ ] Run `find tldw_Server_API/Config_Files/Prompts -maxdepth 2 -type f | sort` and paste the command plus reviewed file list into the task notes, not the design document.
- [ ] Run `rg -n "load_prompt\\(" tldw_Server_API/app --glob "*.py"` and map every result to its YAML/Markdown key and fallback literal.
- [ ] Record every key, including embeddings, ingestion, RAG retrieval/reranking, evaluator, and MCP keys that are expected to be excluded.
- [ ] Verify that paired system/user prompts are one atomic definition with two parts; do not assign separate definition IDs.
- [ ] Assign eligible IDs with lowercase dotted names and stable semantic meaning, for example `media.document.summary`, never module paths or row numbers.
- [ ] Mark output-schema-sensitive extraction prompts `deferred` until schema enforcement is structurally independent; mark routing, ranking, evaluator, moderation, auth, and tool-policy prompts `excluded`.
- [ ] Commit: `docs: inventory file-backed service prompt candidates (<task-id>)`.

## Task 3: Inventory hard-coded prompt candidates

**Files:**

- Modify: `Docs/Design/service-prompt-inventory.md`
- Inspect: `tldw_Server_API/app/core/**/*.py`
- Inspect: `tldw_Server_API/app/services/**/*.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/**/*.py`
- Inspect: `apps/packages/ui/src/**/*.{ts,tsx}`
- Inspect: `apps/tldw-frontend/**/*.{ts,tsx}`

- [ ] Search construction sites with `rg -n "system_prompt|user_prompt|prompt_template|instruction" tldw_Server_API/app/core tldw_Server_API/app/services tldw_Server_API/app/api/v1/endpoints --glob "*.py"`.
- [ ] Search frontend/extension content-generation instructions with `rg -n "systemPrompt|userPrompt|promptTemplate|instruction" apps/packages/ui/src apps/tldw-frontend --glob "*.ts" --glob "*.tsx"`; distinguish prompts sent to services/models from UI labels, examples, and local setting names.
- [ ] Narrow each hit to text sent to an LLM; discard schemas, API request fields, comments, tests, and provider adapters that merely forward caller text.
- [ ] Record all content-generation candidates in the five approved rollout domains: summarization/media/audio; documents/web; RAG generation; reports/digests/watchlists/outputs; extraction/chunking.
- [ ] Record every security/control candidate as an explicit excluded row so future contributors do not re-propose it accidentally.
- [ ] For each eligible row, name every current call site, whether execution is synchronous, Jobs-backed, Scheduler-backed, or mixed, and the exact no-override provider-message golden that will prove byte-equivalent migration.
- [ ] Run `rg -n "You are|Return (only|valid)|ignore previous|moderation|judge|rerank|route" tldw_Server_API/app apps/packages/ui/src apps/tldw-frontend --glob "*.py" --glob "*.ts" --glob "*.tsx"` as a second-pass omission check.
- [ ] Commit: `docs: inventory hard-coded service prompt candidates (<task-id>)`.

## Task 4: Validate IDs, contracts, and rollout units

**Files:**

- Modify: `Docs/Design/service-prompt-inventory.md`
- Create after matrix approval: `Docs/superpowers/plans/<date>-service-prompts-domain-<slug>.md` (one per rollout domain)

- [ ] Sort eligible IDs and fail review on duplicates, ambiguous names, or two definitions sharing a call site.
- [ ] For each eligible definition, specify part order, locked/visible flags, `literal` or `template` mode, declared placeholders, per-variable maximum expansion, and total rendered budget.
- [ ] Confirm that every multipart call site can consume an atomic bundle; otherwise mark it deferred and record the prerequisite.
- [ ] Confirm every async call site names its job creation adapter and worker entry point; otherwise mark it blocked on plan 5.
- [ ] Obtain human approval of the matrix before authoring domain plans.
- [ ] Create one Backlog task and one exact-file implementation plan per rollout domain. Each plan must cite matrix rows, migrate one coherent domain, preserve literal/template explicit-request semantics, add provenance and byte-equivalent default-provider-message goldens, and name all sync/job call sites.
- [ ] Do not create placeholder domain plans for empty or still-deferred domains.

## Task 5: Verify and finalize

- [ ] Rerun both backend and frontend searches from Tasks 2–3 and confirm every candidate sent to a model/service has a matrix row.
- [ ] Run `git diff --check`.
- [ ] Review the inventory against the approved spec's eligibility and locked-part rules.
- [ ] Update the Backlog task with the matrix path, counts by decision, approval evidence, generated domain task/plan links, and verification commands.
- [ ] Commit: `docs: approve service prompt inventory and rollout matrix (<task-id>)`.

Bandit is not required for this documentation-only plan. Domain implementation plans must include Bandit for their touched Python scopes.

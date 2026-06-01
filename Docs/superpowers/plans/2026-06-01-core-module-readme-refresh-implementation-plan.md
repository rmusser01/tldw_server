# Core Module README Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create or refresh contributor-oriented `README.md` files for all 88 top-level modules under `tldw_Server_API/app/core`, then record verification and follow-up deep-guide candidates.

**Architecture:** Treat the work as documentation coverage plus source-backed orientation. Build an inventory first, write missing READMEs in batches, then tighten existing READMEs without flattening useful long-form guides. Verification is local and docs-focused: coverage, placeholder scan, link sanity, optional spelling, and Backlog closeout.

**Tech Stack:** Markdown, `rg`, `find`, existing Backlog.md CLI, local Python only for read-only Markdown sanity checks.

---

## File Structure

- Create: `Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md`
- Create: 40 missing top-level module README files listed in Stages 2-4.
- Modify: existing top-level module README files only when they are scaffolded, thin, stale, or missing contributor orientation.
- Modify: `Docs/superpowers/specs/2026-06-01-core-module-readme-refresh-design.md` only for verified inventory corrections.
- Modify: `backlog/tasks/task-588 - Refresh-core-module-README-developer-docs.md` for progress, verification, and final summary.

Do not change runtime code for this task. If source behavior appears wrong while documenting it, record it as a follow-up in the inventory instead of refactoring.

---

## Stage 1: Inventory And Red Checks

**Goal:** Establish the exact module set, source-evidence tracking, and failing documentation checks before writing README content.

**Success Criteria:** The implementation inventory exists, lists all 88 top-level modules, marks 48 existing READMEs and 40 missing READMEs, and records the initial failing checks.

**Tests:** README coverage check fails with the 40 missing modules. Placeholder scan fails because `Writing/README.md` is scaffold text.

**Status:** Not Started

### Task 1: Create The Implementation Inventory

**Files:**
- Create: `Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md`
- Modify: `Docs/superpowers/specs/2026-06-01-core-module-readme-refresh-design.md`
- Modify: `backlog/tasks/task-588 - Refresh-core-module-README-developer-docs.md`

- [ ] **Step 1: Confirm clean isolated workspace**

Run:

```bash
git status --short
```

Expected: only this plan, the inventory, the corrected spec count, and `TASK-588` changes appear. If unrelated changes appear in `.worktrees/core-module-readmes`, stop and inspect before editing.

- [ ] **Step 2: Record current module coverage**

Run:

```bash
find tldw_Server_API/app/core -mindepth 1 -maxdepth 1 -type d ! -name '__pycache__' | sort | wc -l
find tldw_Server_API/app/core -mindepth 1 -maxdepth 1 -type d ! -name '__pycache__' | sort | while read d; do if test -f "$d/README.md"; then basename "$d"; fi; done | wc -l
find tldw_Server_API/app/core -mindepth 1 -maxdepth 1 -type d ! -name '__pycache__' | sort | while read d; do test -f "$d/README.md" || basename "$d"; done | wc -l
```

Expected:

```text
88
48
40
```

- [ ] **Step 3: Create inventory file**

Create `Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md` with a table containing these exact columns:

```markdown
# Core Module README Refresh Inventory

Backlog: TASK-588
Design: Docs/superpowers/specs/2026-06-01-core-module-readme-refresh-design.md

## Legend

- README status: `existing`, `missing`, `created`, `refreshed`, `kept`
- Evidence status: `not inspected`, `inspected`
- Phase 2 priority: `high`, `medium`, `low`, `sufficient`

## Inventory

| Module | README status | Evidence status | Evidence inspected | Related endpoints/schemas/tests | Phase 2 priority | Notes |
| --- | --- | --- | --- | --- | --- | --- |
```

Add one row per top-level module. Use the 88 module names from `find tldw_Server_API/app/core -mindepth 1 -maxdepth 1 -type d ! -name '__pycache__' | sort`. Initialize README status as `existing` or `missing`, evidence status as `not inspected`, and phase priority using the design spec: high for broad/security/orchestration/data modules, medium for normal feature modules, low for small helper modules.

- [ ] **Step 4: Run red coverage check**

Run:

```bash
missing="$(find tldw_Server_API/app/core -mindepth 1 -maxdepth 1 -type d ! -name '__pycache__' | sort | while read d; do test -f "$d/README.md" || basename "$d"; done)"
printf '%s\n' "$missing"
test -z "$missing"
```

Expected: command exits non-zero and prints 40 missing modules.

- [ ] **Step 5: Run red placeholder check**

Run:

```bash
rg -n "Replace placeholders|scaffolded from the core template|Link API routes and files|Planned improvements|T[B]D|F[I]XME" tldw_Server_API/app/core --glob 'README.md'
```

Expected: command exits zero and reports `tldw_Server_API/app/core/Writing/README.md`.

- [ ] **Step 6: Update Backlog with inventory path**

Run:

```bash
backlog task edit TASK-588 --append-notes "Implementation inventory created at Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md. Initial red checks: 40 top-level core modules missing README.md; Writing README contains scaffold placeholder text." --plain
```

Expected: `TASK-588` notes include the inventory path and initial failing checks.

- [ ] **Step 7: Commit inventory**

Run:

```bash
git add Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md "backlog/tasks/task-588 - Refresh-core-module-README-developer-docs.md" Docs/superpowers/specs/2026-06-01-core-module-readme-refresh-design.md
git commit -m "docs: inventory core module readme coverage"
```

Expected: commit includes only the inventory, Backlog note, and corrected spec count.

---

## Stage 2: Create Missing README Files, Batch A-F

**Goal:** Create source-backed READMEs for the first eight missing modules.

**Success Criteria:** The listed modules have non-placeholder README files with purpose, start points, module map, connections, extension points, testing notes, and gotchas where relevant.

**Tests:** Coverage check missing count drops from 40 to 32.

**Status:** Not Started

### Task 2: Write Missing READMEs For A-F Modules

**Files:**
- Create: `tldw_Server_API/app/core/Agent_Client_Protocol/README.md`
- Create: `tldw_Server_API/app/core/Agent_Orchestration/README.md`
- Create: `tldw_Server_API/app/core/Audio/README.md`
- Create: `tldw_Server_API/app/core/Audiobooks/README.md`
- Create: `tldw_Server_API/app/core/Chat_Workflows/README.md`
- Create: `tldw_Server_API/app/core/CodeGraph/README.md`
- Create: `tldw_Server_API/app/core/Data_Tables/README.md`
- Create: `tldw_Server_API/app/core/File_Artifacts/README.md`
- Modify: `Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md`

- [ ] **Step 1: Inspect source evidence for each module**

Run the evidence command for each exact module name in this task:

```bash
MODULE=Agent_Client_Protocol
find "tldw_Server_API/app/core/$MODULE" -maxdepth 2 -type f | sort
rg -n "$MODULE|core\\.$MODULE" tldw_Server_API/app/api/v1 tldw_Server_API/tests Docs tldw_Server_API/app/core -g '!**/__pycache__/**'
```

Expected: source files, related tests, and related docs/endpoints are identified. Repeat with `Agent_Orchestration`, `Audio`, `Audiobooks`, `Chat_Workflows`, `CodeGraph`, `Data_Tables`, and `File_Artifacts`. Use additional searches for common aliases such as `ACP`, `Agent Client Protocol`, `FileArtifacts`, and `DataTables` when the direct module search is sparse.

- [ ] **Step 2: Create README content**

Each README must include:

- `# Module Name`
- A one-paragraph source-backed purpose statement.
- `## Start Here` with primary source files, related API surface, and related tests.
- `## Responsibilities` with concrete responsibilities found in source.
- `## Module Map` with paths and roles.
- `## How It Connects` describing adjacent modules, endpoints, schemas, DB, Jobs, Scheduler, AuthNZ, storage, providers, or external services when present.
- `## Extension Points` with concrete contributor tasks and files to inspect first.
- `## Testing` with focused test paths when found, or an explicit note that no direct module-specific tests were found in this pass.
- `## Gotchas` only when a concrete risk is found from source.

Expected: no endpoint, schema, or test path is invented. If evidence is absent, the README says so directly.

- [ ] **Step 3: Update inventory rows**

For each module row, set `README status` to `created`, `Evidence status` to `inspected`, and fill `Evidence inspected` plus `Related endpoints/schemas/tests` with files actually checked.

- [ ] **Step 4: Verify batch coverage**

Run:

```bash
missing="$(find tldw_Server_API/app/core -mindepth 1 -maxdepth 1 -type d ! -name '__pycache__' | sort | while read d; do test -f "$d/README.md" || basename "$d"; done)"
printf '%s\n' "$missing" | wc -l
printf '%s\n' "$missing"
rg -n "Replace placeholders|scaffolded from the core template|Link API routes and files|Planned improvements|T[B]D|F[I]XME" tldw_Server_API/app/core --glob 'README.md'
```

Expected: missing count is `32`. Placeholder scan still reports only unresolved pre-existing README placeholder content.

- [ ] **Step 5: Commit batch A-F**

Run:

```bash
git add tldw_Server_API/app/core/Agent_Client_Protocol/README.md tldw_Server_API/app/core/Agent_Orchestration/README.md tldw_Server_API/app/core/Audio/README.md tldw_Server_API/app/core/Audiobooks/README.md tldw_Server_API/app/core/Chat_Workflows/README.md tldw_Server_API/app/core/CodeGraph/README.md tldw_Server_API/app/core/Data_Tables/README.md tldw_Server_API/app/core/File_Artifacts/README.md Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md
git commit -m "docs: add core readmes batch a-f"
```

Expected: commit includes only the eight README files and inventory updates.

---

## Stage 3: Create Missing README Files, Batch G-R

**Goal:** Create source-backed READMEs for the next eleven missing modules.

**Success Criteria:** The listed modules have non-placeholder README files and inventory rows with source evidence.

**Tests:** Coverage check missing count drops from 32 to 21.

**Status:** Not Started

### Task 3: Write Missing READMEs For G-R Modules

**Files:**
- Create: `tldw_Server_API/app/core/Governance/README.md`
- Create: `tldw_Server_API/app/core/Image_Generation/README.md`
- Create: `tldw_Server_API/app/core/Ingestion_Sources/README.md`
- Create: `tldw_Server_API/app/core/Integrations/README.md`
- Create: `tldw_Server_API/app/core/Meetings/README.md`
- Create: `tldw_Server_API/app/core/Notes_Graph/README.md`
- Create: `tldw_Server_API/app/core/Personalization/README.md`
- Create: `tldw_Server_API/app/core/Prototype_Workspaces/README.md`
- Create: `tldw_Server_API/app/core/Reminders/README.md`
- Create: `tldw_Server_API/app/core/Research/README.md`
- Create: `tldw_Server_API/app/core/Research_Workspace/README.md`
- Modify: `Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md`

- [ ] **Step 1: Inspect source evidence for each module**

Run the evidence command for each exact module name in this task:

```bash
MODULE=Governance
find "tldw_Server_API/app/core/$MODULE" -maxdepth 2 -type f | sort
rg -n "$MODULE|core\\.$MODULE" tldw_Server_API/app/api/v1 tldw_Server_API/tests Docs tldw_Server_API/app/core -g '!**/__pycache__/**'
```

Expected: evidence is recorded. Use additional searches for aliases such as `ImageGeneration`, `Ingestion Sources`, `PrototypeWorkspaces`, `Research Workspace`, and `research_workspace` when needed.

- [ ] **Step 2: Create README content**

Use the required README sections from Stage 2. Keep simple modules short; use tables only when they improve scanability.

- [ ] **Step 3: Update inventory rows**

Expected: all eleven rows are marked `created` and `inspected`, with evidence paths filled.

- [ ] **Step 4: Verify batch coverage**

Run:

```bash
missing="$(find tldw_Server_API/app/core -mindepth 1 -maxdepth 1 -type d ! -name '__pycache__' | sort | while read d; do test -f "$d/README.md" || basename "$d"; done)"
printf '%s\n' "$missing" | wc -l
printf '%s\n' "$missing"
rg -n "Replace placeholders|scaffolded from the core template|Link API routes and files|Planned improvements|T[B]D|F[I]XME" tldw_Server_API/app/core --glob 'README.md'
```

Expected: missing count is `21`. Placeholder scan still reports only unresolved pre-existing README placeholder content.

- [ ] **Step 5: Commit batch G-R**

Run:

```bash
git add tldw_Server_API/app/core/Governance/README.md tldw_Server_API/app/core/Image_Generation/README.md tldw_Server_API/app/core/Ingestion_Sources/README.md tldw_Server_API/app/core/Integrations/README.md tldw_Server_API/app/core/Meetings/README.md tldw_Server_API/app/core/Notes_Graph/README.md tldw_Server_API/app/core/Personalization/README.md tldw_Server_API/app/core/Prototype_Workspaces/README.md tldw_Server_API/app/core/Reminders/README.md tldw_Server_API/app/core/Research/README.md tldw_Server_API/app/core/Research_Workspace/README.md Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md
git commit -m "docs: add core readmes batch g-r"
```

Expected: commit includes only these eleven README files and inventory updates.

---

## Stage 4: Create Missing README Files, Batch S-Z And Helper Packages

**Goal:** Create source-backed READMEs for the remaining 21 missing modules.

**Success Criteria:** No top-level core module is missing a README.

**Tests:** README coverage check passes with zero missing modules.

**Status:** Not Started

### Task 4: Write Missing READMEs For S-Z And Lowercase Helper Modules

**Files:**
- Create: `tldw_Server_API/app/core/Sharing/README.md`
- Create: `tldw_Server_API/app/core/Skills/README.md`
- Create: `tldw_Server_API/app/core/Slides/README.md`
- Create: `tldw_Server_API/app/core/Storage/README.md`
- Create: `tldw_Server_API/app/core/Streaming/README.md`
- Create: `tldw_Server_API/app/core/StudyPacks/README.md`
- Create: `tldw_Server_API/app/core/StudySuggestions/README.md`
- Create: `tldw_Server_API/app/core/Telegram/README.md`
- Create: `tldw_Server_API/app/core/Templating/README.md`
- Create: `tldw_Server_API/app/core/Text2SQL/README.md`
- Create: `tldw_Server_API/app/core/UserProfiles/README.md`
- Create: `tldw_Server_API/app/core/VN_Assets/README.md`
- Create: `tldw_Server_API/app/core/VN_Platform/README.md`
- Create: `tldw_Server_API/app/core/VN_Play/README.md`
- Create: `tldw_Server_API/app/core/VN_Policy/README.md`
- Create: `tldw_Server_API/app/core/VN_Scripts/README.md`
- Create: `tldw_Server_API/app/core/VoiceAssistant/README.md`
- Create: `tldw_Server_API/app/core/WebClipper/README.md`
- Create: `tldw_Server_API/app/core/Workspaces/README.md`
- Create: `tldw_Server_API/app/core/config_sections/README.md`
- Create: `tldw_Server_API/app/core/deprecations/README.md`
- Modify: `Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md`

- [ ] **Step 1: Inspect source evidence for each module**

Run the evidence command for each exact module name in this task:

```bash
MODULE=Sharing
find "tldw_Server_API/app/core/$MODULE" -maxdepth 2 -type f | sort
rg -n "$MODULE|core\\.$MODULE" tldw_Server_API/app/api/v1 tldw_Server_API/tests Docs tldw_Server_API/app/core -g '!**/__pycache__/**'
```

Expected: evidence is recorded. Use additional searches for aliases such as `StudySuggestions`, `UserProfile`, `VN Assets`, `VN_Play`, `Voice Assistant`, `WebClipper`, `config_sections`, and `deprecations`.

- [ ] **Step 2: Create README content**

Use the required README sections from Stage 2. For `config_sections` and `deprecations`, explain that they are support packages rather than user-facing feature modules.

- [ ] **Step 3: Update inventory rows**

Expected: all 21 rows are marked `created` and `inspected`, with evidence paths filled.

- [ ] **Step 4: Verify zero missing READMEs**

Run:

```bash
missing="$(find tldw_Server_API/app/core -mindepth 1 -maxdepth 1 -type d ! -name '__pycache__' | sort | while read d; do test -f "$d/README.md" || basename "$d"; done)"
printf '%s\n' "$missing"
test -z "$missing"
```

Expected: no output and exit code `0`.

- [ ] **Step 5: Commit batch S-Z**

Run:

```bash
git add tldw_Server_API/app/core/Sharing/README.md tldw_Server_API/app/core/Skills/README.md tldw_Server_API/app/core/Slides/README.md tldw_Server_API/app/core/Storage/README.md tldw_Server_API/app/core/Streaming/README.md tldw_Server_API/app/core/StudyPacks/README.md tldw_Server_API/app/core/StudySuggestions/README.md tldw_Server_API/app/core/Telegram/README.md tldw_Server_API/app/core/Templating/README.md tldw_Server_API/app/core/Text2SQL/README.md tldw_Server_API/app/core/UserProfiles/README.md tldw_Server_API/app/core/VN_Assets/README.md tldw_Server_API/app/core/VN_Platform/README.md tldw_Server_API/app/core/VN_Play/README.md tldw_Server_API/app/core/VN_Policy/README.md tldw_Server_API/app/core/VN_Scripts/README.md tldw_Server_API/app/core/VoiceAssistant/README.md tldw_Server_API/app/core/WebClipper/README.md tldw_Server_API/app/core/Workspaces/README.md tldw_Server_API/app/core/config_sections/README.md tldw_Server_API/app/core/deprecations/README.md Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md
git commit -m "docs: add remaining core readmes"
```

Expected: commit includes only these 21 README files and inventory updates.

---

## Stage 5: Refresh Existing READMEs And Verify

**Goal:** Replace scaffolded/thin README content, preserve strong long-form guides, and complete verification.

**Success Criteria:** All 88 module READMEs are source-backed, placeholder scan passes, local link sanity check passes, and `TASK-588` records verification plus docs-only Bandit skip.

**Tests:** Coverage, placeholder, link sanity, optional spelling scan.

**Status:** Not Started

### Task 5: Tighten Existing READMEs

**Files:**
- Modify as needed: `tldw_Server_API/app/core/Audit/README.md`
- Modify as needed: `tldw_Server_API/app/core/AuthNZ/README.md`
- Modify as needed: `tldw_Server_API/app/core/Billing/README.md`
- Modify as needed: `tldw_Server_API/app/core/Character_Chat/README.md`
- Modify as needed: `tldw_Server_API/app/core/Chat/README.md`
- Modify as needed: `tldw_Server_API/app/core/Chatbooks/README.md`
- Modify as needed: `tldw_Server_API/app/core/Chunking/README.md`
- Modify as needed: `tldw_Server_API/app/core/Claims_Extraction/README.md`
- Modify as needed: `tldw_Server_API/app/core/Collections/README.md`
- Modify as needed: `tldw_Server_API/app/core/DB_Management/README.md`
- Modify as needed: `tldw_Server_API/app/core/Embeddings/README.md`
- Modify as needed: `tldw_Server_API/app/core/Evaluations/README.md`
- Modify as needed: `tldw_Server_API/app/core/External_Sources/README.md`
- Modify as needed: `tldw_Server_API/app/core/Flashcards/README.md`
- Modify as needed: `tldw_Server_API/app/core/Infrastructure/README.md`
- Modify as needed: `tldw_Server_API/app/core/Ingestion_Media_Processing/README.md`
- Modify as needed: `tldw_Server_API/app/core/Jobs/README.md`
- Modify as needed: `tldw_Server_API/app/core/LLM_Calls/README.md`
- Modify as needed: `tldw_Server_API/app/core/Local_LLM/README.md`
- Modify as needed: `tldw_Server_API/app/core/Logging/README.md`
- Modify as needed: `tldw_Server_API/app/core/MCP_unified/README.md`
- Modify as needed: `tldw_Server_API/app/core/Metrics/README.md`
- Modify as needed: `tldw_Server_API/app/core/Moderation/README.md`
- Modify as needed: `tldw_Server_API/app/core/Monitoring/README.md`
- Modify as needed: `tldw_Server_API/app/core/Notes/README.md`
- Modify as needed: `tldw_Server_API/app/core/Notifications/README.md`
- Modify as needed: `tldw_Server_API/app/core/Persona/README.md`
- Modify as needed: `tldw_Server_API/app/core/PrivilegeMaps/README.md`
- Modify as needed: `tldw_Server_API/app/core/Prompt_Management/README.md`
- Modify as needed: `tldw_Server_API/app/core/RAG/README.md`
- Modify as needed: `tldw_Server_API/app/core/RateLimiting/README.md`
- Modify as needed: `tldw_Server_API/app/core/Resource_Governance/README.md`
- Modify as needed: `tldw_Server_API/app/core/Sandbox/README.md`
- Modify as needed: `tldw_Server_API/app/core/Scheduler/README.md`
- Modify as needed: `tldw_Server_API/app/core/Search_and_Research/README.md`
- Modify as needed: `tldw_Server_API/app/core/Security/README.md`
- Modify as needed: `tldw_Server_API/app/core/Setup/README.md`
- Modify as needed: `tldw_Server_API/app/core/Sync/README.md`
- Modify as needed: `tldw_Server_API/app/core/TTS/README.md`
- Modify as needed: `tldw_Server_API/app/core/Third_Party/README.md`
- Modify as needed: `tldw_Server_API/app/core/Tools/README.md`
- Modify as needed: `tldw_Server_API/app/core/Usage/README.md`
- Modify as needed: `tldw_Server_API/app/core/Utils/README.md`
- Modify as needed: `tldw_Server_API/app/core/Watchlists/README.md`
- Modify as needed: `tldw_Server_API/app/core/WebSearch/README.md`
- Modify as needed: `tldw_Server_API/app/core/Web_Scraping/README.md`
- Modify as needed: `tldw_Server_API/app/core/Workflows/README.md`
- Replace: `tldw_Server_API/app/core/Writing/README.md`
- Modify: `Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md`

- [ ] **Step 1: Classify existing README files**

Run:

```bash
for md in tldw_Server_API/app/core/*/README.md; do printf '%4s %s\n' "$(wc -l < "$md")" "$md"; done | sort -n
rg -n "Replace placeholders|scaffolded from the core template|Link API routes and files|Planned improvements|T[B]D|F[I]XME" tldw_Server_API/app/core --glob 'README.md'
```

Expected: short files and explicit placeholders are identified. `Writing/README.md` must be replaced. Long guides such as `Chat`, `MCP_unified`, `Metrics`, `Evaluations`, and `Chunking` should be preserved unless source inspection shows material inaccuracies.

- [ ] **Step 2: Refresh scaffolded or thin files**

For each existing README that is thin or scaffolded, inspect source first:

```bash
MODULE=Writing
find "tldw_Server_API/app/core/$MODULE" -maxdepth 2 -type f | sort
rg -n "$MODULE|core\\.$MODULE" tldw_Server_API/app/api/v1 tldw_Server_API/tests Docs tldw_Server_API/app/core -g '!**/__pycache__/**'
```

Expected: refreshed README content is source-backed. Start with `Writing/README.md`, then review short files under roughly 70 lines: `Tools`, `Utils`, `PrivilegeMaps`, `Setup`, `Usage`, `Logging`, `RateLimiting`, `Flashcards`, `Local_LLM`, `Moderation`, `Notifications`, `Billing`, `Watchlists`, `Security`, `Search_and_Research`, `Collections`, `Claims_Extraction`, and `Jobs`.

- [ ] **Step 3: Preserve and lightly tighten strong READMEs**

For long READMEs, only add a concise `Start Here` orientation or update stale references if needed. Do not rewrite entire long guides unless source inspection shows material inaccuracies.

Expected: `Chat`, `RAG`, `MCP_unified`, `Metrics`, `Evaluations`, `Chunking`, `Sandbox`, `Character_Chat`, and `Chatbooks` keep their useful detail.

- [ ] **Step 4: Update inventory rows**

Set existing README rows to `refreshed` or `kept`. Fill evidence columns for every row. Set Phase 2 priority to `sufficient` when the concise README is enough; otherwise keep `high`, `medium`, or `low` with a short note.

- [ ] **Step 5: Commit existing README refresh**

Run:

```bash
git add tldw_Server_API/app/core/*/README.md Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md
git commit -m "docs: refresh existing core readmes"
```

Expected: commit contains only README files and inventory changes.

### Task 6: Final Verification And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-588 - Refresh-core-module-README-developer-docs.md`
- Modify if needed: `Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md`

- [ ] **Step 1: Run final README coverage check**

Run:

```bash
missing="$(find tldw_Server_API/app/core -mindepth 1 -maxdepth 1 -type d ! -name '__pycache__' | sort | while read d; do test -f "$d/README.md" || basename "$d"; done)"
test -z "$missing"
```

Expected: exit code `0`.

- [ ] **Step 2: Run final placeholder check**

Run:

```bash
rg -n "Replace placeholders|scaffolded from the core template|Link API routes and files|Planned improvements|T[B]D|F[I]XME" tldw_Server_API/app/core --glob 'README.md'
```

Expected: no matches and exit code `1`.

- [ ] **Step 3: Run local Markdown link sanity check**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re
import sys

root = Path("tldw_Server_API/app/core")
errors = []
for md in root.glob("*/README.md"):
    text = md.read_text(encoding="utf-8")
    if not text.startswith("# "):
        errors.append(f"{md}: missing top-level heading")
    for match in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = match.strip()
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target_path = target.split("#", 1)[0]
        if not target_path:
            continue
        candidates = [(md.parent / target_path), Path(target_path)]
        if not any(candidate.exists() for candidate in candidates):
            errors.append(f"{md}: broken local link {target}")

if errors:
    print("\n".join(errors))
    sys.exit(1)
print("core README markdown sanity checks passed")
PY
```

Expected:

```text
core README markdown sanity checks passed
```

- [ ] **Step 4: Run optional spelling scan if installed**

Run:

```bash
if command -v codespell >/dev/null 2>&1; then
  codespell tldw_Server_API/app/core --skip='*.py,*.json,*.yaml,*.yml,*.db,*.sqlite'
else
  echo "codespell not installed; spelling scan skipped"
fi
```

Expected: either `codespell` passes or the skip message is recorded in `TASK-588`.

- [ ] **Step 5: Record Bandit docs-only skip**

If only Markdown/backlog files changed, do not run Bandit. Record this note in Backlog:

```bash
backlog task edit TASK-588 --append-notes "Verification: Bandit skipped because this task changed Markdown documentation and Backlog metadata only; no Python/source code was modified." --plain
```

If any Python/source file changed, stop and revise the plan before closeout. Source changes are outside this documentation-only task.

- [ ] **Step 6: Check acceptance criteria and final summary**

Run:

```bash
backlog task edit TASK-588 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 6 --append-final-summary "Completed source-informed README orientation pass for all 88 top-level app/core modules. Added missing README files, refreshed scaffolded/thin existing READMEs, preserved strong long-form guides, and recorded verification results plus docs-only Bandit skip." --plain
backlog task edit TASK-588 --check-dod 5 --plain
```

Expected: Backlog acceptance criteria and Definition of Done items are checked.

- [ ] **Step 7: Commit verification closeout**

Run:

```bash
git add "backlog/tasks/task-588 - Refresh-core-module-README-developer-docs.md" Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md
git commit -m "docs: verify core readme refresh"
```

Expected: commit contains Backlog closeout and any final inventory verification updates.

---

## Self-Review Notes

- Spec coverage: all design goals map to stages: inventory (Stage 1), missing README creation (Stages 2-4), existing README preservation/tightening (Stage 5), verification/backlog closeout (Task 6), and Phase 2 candidate tracking (inventory fields).
- Placeholder scan: no incomplete work markers are required as plan content. Search patterns include stale placeholder text to detect in module READMEs.
- Path consistency: all module names are exact top-level directory names under `tldw_Server_API/app/core` as of the approved spec inventory.

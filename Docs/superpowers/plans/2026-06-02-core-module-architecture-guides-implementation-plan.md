# Core Module Architecture Guides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Phase 2 architecture-guide pass for all 88 top-level `tldw_Server_API/app/core` modules by expanding high-risk/high-complexity README guidance and recording sufficiency decisions for modules that do not need more depth.

**Architecture:** Start with a source-evidence review matrix derived from the Phase 1 inventory. Expand README architecture sections only when source, endpoint, schema, config, DB, test, or operational evidence shows contributors need deeper guidance. Keep simple modules concise and record the reason they are sufficient rather than padding them.

**Tech Stack:** Markdown, Backlog.md CLI, `rg`, `find`, local Python for read-only inventory/link sanity checks, existing project virtual environment for optional local tooling.

---

## File Structure

- Create: `Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md`
- Create: `Docs/superpowers/plans/2026-06-02-core-module-architecture-guides-implementation-plan.md`
- Modify: `backlog/tasks/task-589 - Deepen-core-module-architecture-guides.md`
- Modify: selected `tldw_Server_API/app/core/<module>/README.md` files only when Phase 2 evidence shows a real architecture-guide gap.
- Do not change runtime source for this task. If source behavior appears wrong while documenting it, record it as a follow-up note in the Phase 2 inventory.

## Architecture-Guide Standard

Expanded modules should add concrete guidance under existing headings or a new `## Architecture Notes` section. Use only sections that are supported by source evidence:

- `Core Flow`: request/job/event/data flow through the module and adjacent modules.
- `Boundaries`: what this module owns versus what callers, endpoints, DB layers, workers, or provider adapters own.
- `State And Data`: DB tables, storage paths, cache keys, config files, environment variables, feature flags, or tenant scoping.
- `Security And Operations`: auth/RBAC, egress, sandboxing, quotas, rate limits, migration/worker risks, idempotency, secret handling, and observability.
- `Extension Checklist`: specific files/tests to touch for common contributor changes.
- `Verification`: focused tests, link sanity, and manual checks when external services or optional dependencies prevent direct tests.

Small modules can remain unchanged if their current README already provides enough contributor orientation. Record those decisions in the Phase 2 inventory with one concrete reason.

---

## Stage 1: Phase 2 Inventory And Review Matrix

**Goal:** Create the review matrix for all 88 modules and confirm the initial high/medium/low/sufficient priority distribution from Phase 1.

**Success Criteria:** The Phase 2 inventory exists, lists all 88 modules, records Phase 1 priority, Phase 2 decision, evidence inspected, target action, and verification notes. Initial counts are recorded as 47 high, 28 medium, 4 low, and 9 sufficient.

**Tests:** Inventory parser reports exactly 88 modules and the expected priority counts. README coverage remains complete with zero missing top-level README files.

**Status:** Complete

### Task 1: Create Phase 2 Inventory

**Files:**
- Create: `Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md`
- Modify: `backlog/tasks/task-589 - Deepen-core-module-architecture-guides.md`

- [ ] **Step 1: Confirm clean worktree**

Run:

```bash
git status --short
```

Expected: only `TASK-589` and this implementation plan are modified before creating the inventory. If unrelated changes appear, stop and inspect before editing.

- [ ] **Step 2: Confirm Phase 1 baseline**

Run:

```bash
find tldw_Server_API/app/core -mindepth 1 -maxdepth 1 -type d ! -name '__pycache__' | sort | wc -l
find tldw_Server_API/app/core -mindepth 1 -maxdepth 1 -type d ! -name '__pycache__' | sort | while read d; do test -f "$d/README.md" || basename "$d"; done
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python - <<'PY'
from pathlib import Path
from collections import Counter
text = Path("Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md").read_text()
rows = []
for line in text.splitlines():
    if not line.startswith("| ") or line.startswith("| ---") or line.startswith("| Module"):
        continue
    cells = [c.strip() for c in line.strip("|").split("|")]
    if len(cells) >= 7:
        rows.append((cells[0], cells[5]))
counts = Counter(priority for _, priority in rows)
print(len(rows))
print(dict(sorted(counts.items())))
PY
```

Expected:

```text
88
```

The missing README command prints no module names. The Python command prints:

```text
88
{'high': 47, 'low': 4, 'medium': 28, 'sufficient': 9}
```

- [ ] **Step 3: Create inventory table**

Create `Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md` with this header and columns:

```markdown
# Core Module Architecture Guide Inventory

Backlog: TASK-589
Phase 1 task: TASK-588
Source inventory: Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md

## Legend

- Phase 1 priority: `high`, `medium`, `low`, `sufficient`
- Phase 2 decision: `expand`, `tighten`, `sufficient`, `follow-up`
- Review status: `not reviewed`, `reviewed`

## Inventory

| Module | Phase 1 priority | Phase 2 decision | Review status | Evidence inspected | Target action | Verification notes |
| --- | --- | --- | --- | --- | --- | --- |
```

Populate one row per module from the Phase 1 inventory. Initial `Phase 2 decision` should be `expand` for high-priority modules, `tighten` for medium-priority modules, and `sufficient` for low/sufficient modules unless the Phase 1 notes already indicate a deeper risk.

- [ ] **Step 4: Record Stage 1 verification**

Append to `TASK-589` implementation notes:

```text
2026-06-02: Created Phase 2 inventory and confirmed the merged TASK-588 baseline: 88 top-level core modules, no missing README files, and priority counts of 47 high, 28 medium, 4 low, and 9 sufficient. Runtime code remains untouched.
```

- [ ] **Step 5: Commit Stage 1**

Run:

```bash
git add Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md "backlog/tasks/task-589 - Deepen-core-module-architecture-guides.md"
git diff --cached --check
git commit -m "docs: inventory core architecture guide pass"
```

Expected: commit includes only the Phase 2 inventory and `TASK-589` note.

---

## Stage 2: Security, Identity, Persistence, And User Data

**Goal:** Deepen the most risk-sensitive guides first: auth, policy, storage, query generation, sharing, sync, and user data boundaries.

**Success Criteria:** Each target README either gains concrete Phase 2 architecture guidance or the inventory records a source-backed reason it is already sufficient. Security/ops claims cite actual files, tests, endpoints, or config surfaces.

**Tests:** Targeted path checks for every referenced file, placeholder scan for changed READMEs, Markdown link sanity, and `git diff --check`.

**Status:** Complete

### Task 2: Expand Security And Data Boundary Guides

**Files:**
- Modify as needed: `tldw_Server_API/app/core/AuthNZ/README.md`
- Modify as needed: `tldw_Server_API/app/core/Billing/README.md`
- Modify as needed: `tldw_Server_API/app/core/DB_Management/README.md`
- Modify as needed: `tldw_Server_API/app/core/Resource_Governance/README.md`
- Modify as needed: `tldw_Server_API/app/core/Sandbox/README.md`
- Modify as needed: `tldw_Server_API/app/core/Security/README.md`
- Modify as needed: `tldw_Server_API/app/core/Sharing/README.md`
- Modify as needed: `tldw_Server_API/app/core/Storage/README.md`
- Modify as needed: `tldw_Server_API/app/core/Sync/README.md`
- Modify as needed: `tldw_Server_API/app/core/Text2SQL/README.md`
- Modify as needed: `tldw_Server_API/app/core/UserProfiles/README.md`
- Modify as needed: `tldw_Server_API/app/core/Workspaces/README.md`
- Modify: `Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md`
- Modify: `backlog/tasks/task-589 - Deepen-core-module-architecture-guides.md`

- [ ] **Step 1: Inspect source evidence for the batch**

For each target module, run:

```bash
module=AuthNZ
find "tldw_Server_API/app/core/$module" -maxdepth 2 -type f | sort | sed -n '1,80p'
rg -n "$module|auth|rbac|tenant|quota|policy|sandbox|storage|sync|text2sql|workspace" tldw_Server_API/app/api/v1 tldw_Server_API/tests tldw_Server_API/app/core --glob '*.py' | sed -n '1,120p'
```

Repeat with the exact module names listed in this task. Use source-specific search terms for each module when the generic terms are noisy. Record the concrete evidence paths in the inventory row before editing a README.

- [ ] **Step 2: Update each README only where the current guide lacks Phase 2 depth**

For each target README, add or tighten architecture guidance using this exact Markdown shape when the README lacks equivalent content:

```markdown
## Architecture Notes

### Core Flow

- Describe the main request/job/data path through concrete files.
- Name the adjacent endpoint, schema, dependency, DB, worker, or service boundaries.

### State And Data

- List concrete stores, paths, config sections, environment variables, caches, or tenant/user scoping rules that contributors must preserve.

### Security And Operations

- List RBAC, policy, egress, sandboxing, quotas, idempotency, migrations, observability, or failure modes that are visible in source/tests.

### Extension Checklist

- For a contributor change, name the source file, API/schema/dependency file, and focused test path that should move together.
```

Do not duplicate a strong existing section. Tighten stale claims instead of adding parallel text.

- [ ] **Step 3: Update inventory decisions**

For every target module, set `Review status` to `reviewed`. Set `Phase 2 decision` to:

- `expand` if the README gained new architecture guidance.
- `tighten` if only stale links, headings, or concise clarifications changed.
- `sufficient` if the README already had enough architecture depth.
- `follow-up` only if a deeper design task is needed outside README scope.

The `Verification notes` cell must name at least one focused test path or explain why docs-only verification is enough.

- [ ] **Step 4: Run Stage 2 verification**

Run:

```bash
rg -n "Replace placeholders|scaffolded from the core template|Link API routes and files|Planned improvements|T[B]D|F[I]XME" tldw_Server_API/app/core --glob 'README.md'
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python - <<'PY'
from pathlib import Path
import re, sys
errors = []
for md in sorted(Path("tldw_Server_API/app/core").glob("*/README.md")):
    text = md.read_text(encoding="utf-8")
    if not text.startswith("# "):
        errors.append(f"{md}: missing top-level heading")
    for match in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = match.strip()
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target_path = target.split("#", 1)[0]
        if target_path and not any(candidate.exists() for candidate in (md.parent / target_path, Path(target_path))):
            errors.append(f"{md}: broken local link {target}")
if errors:
    print("\n".join(errors))
    sys.exit(1)
print("core README markdown sanity checks passed")
PY
git diff --check
```

Expected: placeholder scan exits `1` with no matches; Python prints `core README markdown sanity checks passed`; `git diff --check` exits `0`.

- [ ] **Step 5: Commit Stage 2**

Run:

```bash
git add tldw_Server_API/app/core/AuthNZ/README.md tldw_Server_API/app/core/Billing/README.md tldw_Server_API/app/core/DB_Management/README.md tldw_Server_API/app/core/Resource_Governance/README.md tldw_Server_API/app/core/Sandbox/README.md tldw_Server_API/app/core/Security/README.md tldw_Server_API/app/core/Sharing/README.md tldw_Server_API/app/core/Storage/README.md tldw_Server_API/app/core/Sync/README.md tldw_Server_API/app/core/Text2SQL/README.md tldw_Server_API/app/core/UserProfiles/README.md tldw_Server_API/app/core/Workspaces/README.md Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md "backlog/tasks/task-589 - Deepen-core-module-architecture-guides.md"
git diff --cached --check
git commit -m "docs: deepen core security data guides"
```

Expected: commit includes only target README files that actually changed, the Phase 2 inventory, and `TASK-589`.

---

## Stage 3: Orchestration, Jobs, RAG, And Execution Boundaries

**Goal:** Deepen guides for modules where contributors must understand multi-step workflows, worker lifecycles, tool execution, retrieval, and scheduler/job contracts.

**Success Criteria:** Target READMEs explain control flow, ownership boundaries, worker/scheduler/job semantics, and extension/testing checklists with concrete source references.

**Tests:** Same docs sanity checks as Stage 2 plus targeted reference existence checks for all mentioned endpoints, workers, schemas, and tests.

**Status:** Complete

### Task 3: Expand Orchestration And Execution Guides

**Files:**
- Modify as needed: `tldw_Server_API/app/core/Agent_Client_Protocol/README.md`
- Modify as needed: `tldw_Server_API/app/core/Agent_Orchestration/README.md`
- Modify as needed: `tldw_Server_API/app/core/Chat/README.md`
- Modify as needed: `tldw_Server_API/app/core/Chat_Workflows/README.md`
- Modify as needed: `tldw_Server_API/app/core/CodeGraph/README.md`
- Modify as needed: `tldw_Server_API/app/core/Ingestion_Media_Processing/README.md`
- Modify as needed: `tldw_Server_API/app/core/Ingestion_Sources/README.md`
- Modify as needed: `tldw_Server_API/app/core/Jobs/README.md`
- Modify as needed: `tldw_Server_API/app/core/MCP_unified/README.md`
- Modify as needed: `tldw_Server_API/app/core/RAG/README.md`
- Modify as needed: `tldw_Server_API/app/core/Scheduler/README.md`
- Modify as needed: `tldw_Server_API/app/core/Workflows/README.md`
- Modify: `Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md`
- Modify: `backlog/tasks/task-589 - Deepen-core-module-architecture-guides.md`

- [ ] **Step 1: Inspect source and tests**

Run targeted file discovery and searches for each module:

```bash
module=Jobs
find "tldw_Server_API/app/core/$module" -maxdepth 3 -type f | sort | sed -n '1,120p'
rg -n "class .*Job|def .*worker|enqueue|scheduler|workflow|rag|mcp|agent|orchestrat|ingest" tldw_Server_API/app/core tldw_Server_API/app/api/v1 tldw_Server_API/tests --glob '*.py' | sed -n '1,160p'
```

Repeat with source-specific terms for each target module. Record concrete evidence in the inventory before editing.

- [ ] **Step 2: Expand or tighten README architecture sections**

For modules with existing long-form guides (`Chat`, `MCP_unified`, `RAG`, `Scheduler`, `Workflows`, `Ingestion_Media_Processing`), prefer stale-path cleanup and contributor checklists over wholesale rewrites. For newer short READMEs (`Agent_Client_Protocol`, `Agent_Orchestration`, `Chat_Workflows`, `CodeGraph`, `Ingestion_Sources`, `Jobs`), add `Architecture Notes` if the flow or worker boundary is not already clear.

- [ ] **Step 3: Update inventory and task notes**

Mark each target module reviewed and record whether it was expanded, tightened, sufficient, or follow-up. Append a `TASK-589` note with the modules changed and the verification commands run.

- [ ] **Step 4: Run Stage 3 verification**

Run the same placeholder scan, Markdown sanity script, and `git diff --check` from Stage 2. Also run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python - <<'PY'
from pathlib import Path
targets = [
    "tldw_Server_API/app/core/Agent_Client_Protocol/README.md",
    "tldw_Server_API/app/core/Agent_Orchestration/README.md",
    "tldw_Server_API/app/core/Chat/README.md",
    "tldw_Server_API/app/core/Chat_Workflows/README.md",
    "tldw_Server_API/app/core/CodeGraph/README.md",
    "tldw_Server_API/app/core/Ingestion_Media_Processing/README.md",
    "tldw_Server_API/app/core/Ingestion_Sources/README.md",
    "tldw_Server_API/app/core/Jobs/README.md",
    "tldw_Server_API/app/core/MCP_unified/README.md",
    "tldw_Server_API/app/core/RAG/README.md",
    "tldw_Server_API/app/core/Scheduler/README.md",
    "tldw_Server_API/app/core/Workflows/README.md",
]
missing = [path for path in targets if not Path(path).is_file()]
if missing:
    raise SystemExit("missing target README files: " + ", ".join(missing))
print("Stage 3 target README files exist")
PY
```

Expected: target file existence script prints `Stage 3 target README files exist`.

- [ ] **Step 5: Commit Stage 3**

Run:

```bash
git add tldw_Server_API/app/core/Agent_Client_Protocol/README.md tldw_Server_API/app/core/Agent_Orchestration/README.md tldw_Server_API/app/core/Chat/README.md tldw_Server_API/app/core/Chat_Workflows/README.md tldw_Server_API/app/core/CodeGraph/README.md tldw_Server_API/app/core/Ingestion_Media_Processing/README.md tldw_Server_API/app/core/Ingestion_Sources/README.md tldw_Server_API/app/core/Jobs/README.md tldw_Server_API/app/core/MCP_unified/README.md tldw_Server_API/app/core/RAG/README.md tldw_Server_API/app/core/Scheduler/README.md tldw_Server_API/app/core/Workflows/README.md Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md "backlog/tasks/task-589 - Deepen-core-module-architecture-guides.md"
git diff --cached --check
git commit -m "docs: deepen core orchestration guides"
```

---

## Stage 4: Providers, External Boundaries, Operations, And Safety Modules

**Goal:** Deepen modules that integrate providers, external networks, observability, moderation, governance, setup, streaming, watchlists, and policy.

**Success Criteria:** Target READMEs make provider boundaries, external I/O, observability, moderation/policy behavior, and operator-facing risks concrete without duplicating API reference material.

**Tests:** Same docs sanity checks as prior stages plus targeted scans for stale claims and nonexistent local links.

**Status:** Complete

### Task 4: Expand Provider And Operations Guides

**Files:**
- Modify as needed: `tldw_Server_API/app/core/Audit/README.md`
- Modify as needed: `tldw_Server_API/app/core/Chatbooks/README.md`
- Modify as needed: `tldw_Server_API/app/core/Data_Tables/README.md`
- Modify as needed: `tldw_Server_API/app/core/Embeddings/README.md`
- Modify as needed: `tldw_Server_API/app/core/Evaluations/README.md`
- Modify as needed: `tldw_Server_API/app/core/External_Sources/README.md`
- Modify as needed: `tldw_Server_API/app/core/File_Artifacts/README.md`
- Modify as needed: `tldw_Server_API/app/core/Governance/README.md`
- Modify as needed: `tldw_Server_API/app/core/Infrastructure/README.md`
- Modify as needed: `tldw_Server_API/app/core/Integrations/README.md`
- Modify as needed: `tldw_Server_API/app/core/LLM_Calls/README.md`
- Modify as needed: `tldw_Server_API/app/core/Metrics/README.md`
- Modify as needed: `tldw_Server_API/app/core/Moderation/README.md`
- Modify as needed: `tldw_Server_API/app/core/Monitoring/README.md`
- Modify as needed: `tldw_Server_API/app/core/Notes/README.md`
- Modify as needed: `tldw_Server_API/app/core/Personalization/README.md`
- Modify as needed: `tldw_Server_API/app/core/Research_Workspace/README.md`
- Modify as needed: `tldw_Server_API/app/core/Setup/README.md`
- Modify as needed: `tldw_Server_API/app/core/Streaming/README.md`
- Modify as needed: `tldw_Server_API/app/core/VN_Policy/README.md`
- Modify as needed: `tldw_Server_API/app/core/Watchlists/README.md`
- Modify as needed: `tldw_Server_API/app/core/WebSearch/README.md`
- Modify as needed: `tldw_Server_API/app/core/Web_Scraping/README.md`
- Modify: `Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md`
- Modify: `backlog/tasks/task-589 - Deepen-core-module-architecture-guides.md`

- [ ] **Step 1: Inspect provider, network, policy, and ops evidence**

Use targeted searches per module:

```bash
module=LLM_Calls
find "tldw_Server_API/app/core/$module" -maxdepth 3 -type f | sort | sed -n '1,140p'
rg -n "provider|adapter|egress|policy|moderation|metrics|monitor|stream|setup|watchlist|websearch|scrap|personalization|note" tldw_Server_API/app/core tldw_Server_API/app/api/v1 tldw_Server_API/tests --glob '*.py' | sed -n '1,180p'
```

Record source evidence in the inventory before editing. For external network modules, include egress and outbound policy test paths when present.

- [ ] **Step 2: Expand or tighten READMEs**

Add architecture depth only where useful. Prioritize:

- audit, chatbook import/export, data-table, and research-workspace boundaries where user data or generated artifacts cross modules;
- provider adapter boundaries and configuration surfaces for `LLM_Calls`, `Embeddings`, `External_Sources`, `WebSearch`, and `Web_Scraping`;
- moderation, governance, VN policy, and personalization safety boundaries;
- metrics/monitoring/infrastructure operational behavior and failure modes;
- setup and streaming lifecycles where contributor changes often cross endpoints/services/tests.

- [ ] **Step 3: Update inventory and task notes**

Mark all target modules reviewed, record decisions, and append a `TASK-589` note listing changed modules plus verification evidence.

- [ ] **Step 4: Run Stage 4 verification**

Run the Stage 2 placeholder scan, Markdown sanity script, and `git diff --check`. Then run:

```bash
rg -n "TODO|TBD|FIXME|does not exist|planned future|coming soon" tldw_Server_API/app/core --glob 'README.md'
```

Expected: no matches for new speculative placeholders. Existing legitimate code or quoted references must be inspected before deciding whether they are acceptable.

- [ ] **Step 5: Commit Stage 4**

Run:

```bash
git add tldw_Server_API/app/core/Audit/README.md tldw_Server_API/app/core/Chatbooks/README.md tldw_Server_API/app/core/Data_Tables/README.md tldw_Server_API/app/core/Embeddings/README.md tldw_Server_API/app/core/Evaluations/README.md tldw_Server_API/app/core/External_Sources/README.md tldw_Server_API/app/core/File_Artifacts/README.md tldw_Server_API/app/core/Governance/README.md tldw_Server_API/app/core/Infrastructure/README.md tldw_Server_API/app/core/Integrations/README.md tldw_Server_API/app/core/LLM_Calls/README.md tldw_Server_API/app/core/Metrics/README.md tldw_Server_API/app/core/Moderation/README.md tldw_Server_API/app/core/Monitoring/README.md tldw_Server_API/app/core/Notes/README.md tldw_Server_API/app/core/Personalization/README.md tldw_Server_API/app/core/Research_Workspace/README.md tldw_Server_API/app/core/Setup/README.md tldw_Server_API/app/core/Streaming/README.md tldw_Server_API/app/core/VN_Policy/README.md tldw_Server_API/app/core/Watchlists/README.md tldw_Server_API/app/core/WebSearch/README.md tldw_Server_API/app/core/Web_Scraping/README.md Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md "backlog/tasks/task-589 - Deepen-core-module-architecture-guides.md"
git diff --cached --check
git commit -m "docs: deepen core provider ops guides"
```

---

## Stage 5: Medium, Low, And Sufficient Module Review

**Goal:** Complete the all-88 review by checking the 28 medium-priority, 4 low-priority, and 9 sufficient modules for architecture gaps without adding unnecessary depth.

**Success Criteria:** Every remaining module is reviewed and recorded in the Phase 2 inventory. Medium modules receive targeted additions only when their current README lacks necessary contributor architecture guidance. Low/sufficient modules are usually recorded as sufficient with source-backed reasons.

**Tests:** Inventory parser verifies all 88 modules have `Review status` of `reviewed`, README sanity checks pass, and docs-only verification is recorded.

**Status:** Not Started

### Task 5: Review Remaining Modules And Close The Matrix

**Files:**
- Modify as needed: medium-priority README files listed in the Phase 1 inventory.
- Modify as needed: low-priority and sufficient README files only if stale or misleading.
- Modify: `Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md`
- Modify: `backlog/tasks/task-589 - Deepen-core-module-architecture-guides.md`

- [ ] **Step 1: Review remaining modules**

Review and update inventory rows for:

```text
Medium: Audio, Audiobooks, Character_Chat, Chunking, Claims_Extraction, Collections, Image_Generation, Local_LLM, Meetings, Notes_Graph, Persona, Prompt_Management, Prototype_Workspaces, Reminders, Research, Skills, Slides, StudyPacks, StudySuggestions, TTS, Telegram, VN_Assets, VN_Platform, VN_Play, VN_Scripts, VoiceAssistant, WebClipper, Writing
Low: Templating, Third_Party, config_sections, deprecations
Sufficient: Flashcards, Logging, Notifications, PrivilegeMaps, RateLimiting, Search_and_Research, Tools, Usage, Utils
```

For each module, inspect at least the README and a representative source file or test path. Mark `sufficient` when the README is already appropriately scoped.

- [ ] **Step 2: Apply targeted README edits only where justified**

Use concise additions. Do not add a long `Architecture Notes` section to helper modules unless the source evidence shows a real contributor hazard.

- [ ] **Step 3: Verify all inventory rows are reviewed**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python - <<'PY'
from pathlib import Path
text = Path("Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md").read_text()
rows = []
for line in text.splitlines():
    if not line.startswith("| ") or line.startswith("| ---") or line.startswith("| Module"):
        continue
    cells = [c.strip() for c in line.strip("|").split("|")]
    if len(cells) >= 7:
        rows.append(cells)
not_reviewed = [row[0] for row in rows if row[3] != "reviewed"]
bad_decisions = [row[0] for row in rows if row[2] not in {"expand", "tighten", "sufficient", "follow-up"}]
if len(rows) != 88:
    raise SystemExit(f"expected 88 rows, found {len(rows)}")
if not_reviewed:
    raise SystemExit("not reviewed: " + ", ".join(not_reviewed))
if bad_decisions:
    raise SystemExit("bad decisions: " + ", ".join(bad_decisions))
print("Phase 2 inventory review complete for 88 modules")
PY
```

Expected: prints `Phase 2 inventory review complete for 88 modules`.

- [ ] **Step 4: Run docs sanity checks**

Run the Stage 2 placeholder scan, Markdown sanity script, and `git diff --check`.

- [ ] **Step 5: Commit Stage 5**

Run:

```bash
git add tldw_Server_API/app/core Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md "backlog/tasks/task-589 - Deepen-core-module-architecture-guides.md"
git diff --cached --check
git commit -m "docs: complete core architecture guide review"
```

---

## Stage 6: Final Verification And Closeout

**Goal:** Prove the Phase 2 pass satisfies `TASK-589`, record verification, and prepare the branch for review.

**Success Criteria:** All acceptance criteria are checked on `TASK-589`, final verification results are recorded, no unrelated files are changed, and branch diff is docs/backlog only.

**Tests:** README coverage, placeholder scan, Markdown link sanity, inventory completion parser, `git diff --check`, branch docs-only diff review. Bandit is skipped and documented if no runtime code changed.

**Status:** Not Started

### Task 6: Verify And Close TASK-589

**Files:**
- Modify: `Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md`
- Modify: `backlog/tasks/task-589 - Deepen-core-module-architecture-guides.md`

- [ ] **Step 1: Run final verification**

Run:

```bash
find tldw_Server_API/app/core -mindepth 1 -maxdepth 1 -type d ! -name '__pycache__' | sort | while read d; do test -f "$d/README.md" || basename "$d"; done
rg -n "Replace placeholders|scaffolded from the core template|Link API routes and files|Planned improvements|T[B]D|F[I]XME" tldw_Server_API/app/core --glob 'README.md'
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python - <<'PY'
from pathlib import Path
import re, sys
errors = []
for md in sorted(Path("tldw_Server_API/app/core").glob("*/README.md")):
    text = md.read_text(encoding="utf-8")
    if not text.startswith("# "):
        errors.append(f"{md}: missing top-level heading")
    for match in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = match.strip()
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target_path = target.split("#", 1)[0]
        if target_path and not any(candidate.exists() for candidate in (md.parent / target_path, Path(target_path))):
            errors.append(f"{md}: broken local link {target}")
if errors:
    print("\n".join(errors))
    sys.exit(1)
print("core README markdown sanity checks passed")
PY
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python - <<'PY'
from pathlib import Path
text = Path("Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md").read_text()
rows = []
for line in text.splitlines():
    if not line.startswith("| ") or line.startswith("| ---") or line.startswith("| Module"):
        continue
    cells = [c.strip() for c in line.strip("|").split("|")]
    if len(cells) >= 7:
        rows.append(cells)
not_reviewed = [row[0] for row in rows if row[3] != "reviewed"]
if len(rows) != 88 or not_reviewed:
    raise SystemExit(f"inventory incomplete: rows={len(rows)}, not_reviewed={not_reviewed}")
print("Phase 2 inventory review complete for 88 modules")
PY
git diff --check
git diff --name-only origin/dev..HEAD
```

Expected:

- missing README command prints no module names;
- placeholder scan exits `1` with no matches;
- Markdown sanity prints `core README markdown sanity checks passed`;
- inventory parser prints `Phase 2 inventory review complete for 88 modules`;
- diff checks exit `0`;
- branch diff includes only Markdown docs and Backlog task files.

- [ ] **Step 2: Record final verification**

Use `backlog task edit TASK-589` to check all acceptance criteria and Definition of Done items, append verification notes, and add a final summary. Include this Bandit note if no runtime source changed:

```text
Bandit skipped because TASK-589 changed Markdown documentation and Backlog records only; no Python or runtime source files were modified.
```

- [ ] **Step 3: Commit closeout**

Run:

```bash
git add Docs/superpowers/plans/2026-06-02-core-module-architecture-guide-inventory.md "backlog/tasks/task-589 - Deepen-core-module-architecture-guides.md"
git diff --cached --check
git commit -m "docs: verify core architecture guide pass"
```

- [ ] **Step 4: Request final review and finish branch**

Run a final branch diff review against `origin/dev`. If the review passes, use `superpowers:finishing-a-development-branch` to offer merge/PR/keep/discard options.

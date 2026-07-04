# API Module Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a user-facing API module capability guide and align OpenAPI tag metadata so `/docs` and `/redoc` are easier to browse.

**Architecture:** Keep detailed guidance in markdown and concise browsing metadata in `tldw_Server_API/app/main.py`. Treat each OpenAPI tag as a module; group tags by user goal and label admin-only or experimental surfaces clearly. Preserve route behavior, schemas, security, and router registration.

**Tech Stack:** Markdown documentation, FastAPI OpenAPI tag metadata, ReDoc `x-tagGroups`, Python stdlib validation scripts, Bandit.

---

## File Structure

- `Docs/API-related/API_Tags_Index.md`: source module capability guide for API users.
- `Docs/Published/API-related/API_Tags_Index.md`: published mirror of the same guide.
- `tldw_Server_API/app/main.py`: concise OpenAPI tag descriptions and ReDoc tag groups.
- `Docs/superpowers/specs/2026-07-04-api-module-documentation-design.md`: approved design spec.
- `Docs/superpowers/plans/2026-07-04-api-module-documentation.md`: this implementation plan.
- `backlog/tasks/task-12027 - Improve-user-facing-API-module-documentation.md`: Backlog.md task record.

## Task 1: Create Inventory Checks

**Files:**
- Read: `tldw_Server_API/app/api/v1/router_groups/core.py`
- Read: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Read: `tldw_Server_API/app/api/v1/router_groups/admin.py`
- Read: `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Read: `tldw_Server_API/app/main.py`

- [x] **Step 1: Generate the router tag inventory**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python - <<'PY'
import ast
from pathlib import Path

files = [
    Path("tldw_Server_API/app/api/v1/router_groups/core.py"),
    Path("tldw_Server_API/app/api/v1/router_groups/content.py"),
    Path("tldw_Server_API/app/api/v1/router_groups/admin.py"),
    Path("tldw_Server_API/app/api/v1/router_groups/minimal.py"),
]
tags: set[str] = set()
for path in files:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "tags":
            value = node.value
            if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
                for item in value.elts:
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        tags.add(item.value)

print("\n".join(sorted(tags)))
print(f"\ncount={len(tags)}")
PY
```

Expected: a sorted list of router tags and a count near `100`. Use this list as the source of truth for the module guide.

- [x] **Step 2: Generate the curated OpenAPI tag inventory**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python - <<'PY'
import ast
from pathlib import Path

tree = ast.parse(Path("tldw_Server_API/app/main.py").read_text())
names: list[str] = []
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "OPENAPI_TAGS":
                for item in node.value.elts:
                    if not isinstance(item, ast.Dict):
                        continue
                    for key, value in zip(item.keys, item.values):
                        if (
                            isinstance(key, ast.Constant)
                            and key.value == "name"
                            and isinstance(value, ast.Constant)
                            and isinstance(value.value, str)
                        ):
                            names.append(value.value)

print("\n".join(names))
print(f"\ncount={len(names)}")
PY
```

Expected: a sorted-by-file-order list of curated OpenAPI tags. Compare it to the router tag inventory to decide which missing tags need curated descriptions.

- [x] **Step 3: Inspect existing docs links**

Run:

```bash
find Docs/API-related Docs/API Docs/User_Guides/Server Docs/MCP Docs/RAG -maxdepth 2 -type f \( -name '*.md' -o -name '*.yaml' \) | sort
```

Expected: existing API docs such as `Docs/API-related/Chat_API_Documentation.md`, `Docs/API-related/RAG-API-Guide.md`, `Docs/API/Voice_Assistant.md`, and `Docs/MCP/Unified/Developer_Guide.md`.

## Task 2: Replace the API Tag Index With a Capability Guide

**Files:**
- Modify: `Docs/API-related/API_Tags_Index.md:1-40`
- Modify: `Docs/Published/API-related/API_Tags_Index.md:1-40`

- [x] **Step 1: Replace the source guide content**

Edit `Docs/API-related/API_Tags_Index.md` so it starts with this structure:

```markdown
# API Module Guide

This guide explains what each API module makes possible. It is organized by user goal rather than by source file, and each module name maps to an OpenAPI tag in `/docs` and `/redoc`.

Use this page when you are deciding which API area to explore first. Use the linked module docs and live OpenAPI pages for request and response details.

## How To Read This Guide

| Column | Meaning |
|--------|---------|
| Module | The OpenAPI tag shown in Swagger/ReDoc. |
| What it lets you do | The capability in user-facing terms. |
| Common uses | Typical workflows or products you can build with it. |
| Docs | The most relevant existing guide, or an inline note when a dedicated guide does not exist yet. |
```

Then add grouped sections for:

```markdown
## Start, Auth, And Configuration
## Media, Documents, And Ingestion
## Audio, Voice, And Speech
## Chat, Characters, And Persona
## Search, RAG, Embeddings, And Evaluation
## Notes, Prompts, Study, And Generated Work
## Automation, Jobs, And Integrations
## Admin, Governance, And Operations
## Experimental And Advanced Surfaces
```

Each section should use this table shape:

```markdown
| Module | What it lets you do | Common uses | Docs |
|--------|---------------------|-------------|------|
| `chat` | Run OpenAI-compatible chat completions and manage conversations. | Build chat clients, provider-backed assistants, and conversation workflows. | [Chat API](Chat_API_Documentation.md) |
```

- [x] **Step 2: Cover all primary router tags**

Include at least these module rows in the source guide:

```text
health, setup, authentication, users, user keys via users, organizations, invites, config, consent, media, media-embeddings, audio, audio-websocket, audio-jobs, audiobooks, voice-assistant, voice-assistant-ws, chat, messages, chat-dictionaries, chat-documents, chat-workflows, characters, character-chat-sessions, character-memory, character-messages, persona, personalization, rag-unified, rag-health, research, research-discovery, research-runs, research-workspace, paper-search, embeddings, vector-stores, claims, feedback, evaluations, benchmarks, ocr, notes, reading, reading-highlights, prompts, prompt-studio, chatbooks, flashcards, quizzes, study-suggestions, writing, manuscripts, slides, outputs, outputs-templates, data-tables, files, storage, items, tasks, notifications, workflows, scheduler, jobs, scheduled-tasks, integrations, ingestion-sources, connectors, web-scraping, collections-feeds, collections-websub, email, meetings, slack, discord, telegram, admin, audit, monitoring, metrics, billing, privileges, resource-governor, llm, llamacpp, mcp-unified, mcp-hub, mcp-catalogs, tools, moderation, acp, acp-schedules, acp-triggers, acp-permissions, acp-multiplex, agent-orchestration, sandbox, prototype-workspaces, sharing, workspaces, companion, guardian, self-monitoring, vn-capabilities, vn-assets, vn-scripts, vn-policy, vn-play
```

Expected: tags without dedicated docs still have a useful capability explanation and a docs note such as `Live OpenAPI only` or `Covered by [API examples](../API/api-examples.md)`.

- [x] **Step 3: Use local links that resolve from `Docs/API-related`**

Use relative links like:

```markdown
[RAG API guide](RAG-API-Guide.md)
[Voice Assistant API](../API/Voice_Assistant.md)
[MCP Unified guide](../MCP/Unified/Developer_Guide.md)
[Server authentication guide](../User_Guides/Server/Authentication_Setup.md)
```

Expected: links resolve from `Docs/API-related/API_Tags_Index.md`.

- [x] **Step 4: Mirror the source guide to Published**

Run:

```bash
cp Docs/API-related/API_Tags_Index.md Docs/Published/API-related/API_Tags_Index.md
```

Expected: `cmp Docs/API-related/API_Tags_Index.md Docs/Published/API-related/API_Tags_Index.md` exits with status `0`.

- [x] **Step 5: Commit the guide**

Run:

```bash
git add Docs/API-related/API_Tags_Index.md Docs/Published/API-related/API_Tags_Index.md
git commit -m "docs: expand API module capability guide"
```

Expected: commit succeeds with only the two tag index docs staged for this task.

## Task 3: Align OpenAPI Tag Metadata

**Files:**
- Modify: `tldw_Server_API/app/main.py:1300-1585`
- Modify: `tldw_Server_API/app/main.py:2038-2077`

- [x] **Step 1: Update concise tag descriptions**

Edit `OPENAPI_TAGS` in `tldw_Server_API/app/main.py` so the curated tags use user-facing descriptions. Keep entries as dictionaries in the existing list. Examples of the target tone:

```python
{"name": "health", "description": "Check whether the API server and core dependencies are reachable."}
{"name": "media", "description": "Ingest, inspect, search, and manage videos, audio, documents, web pages, and other source material."}
{"name": "rag-unified", "description": "Search indexed knowledge with hybrid keyword, vector, reranking, and context-building controls."}
{"name": "prompt-studio", "description": "Build, test, compare, and optimize prompts as reusable projects and runs."}
```

Expected: descriptions answer what users can do, not just what implementation component exists.

- [x] **Step 2: Add curated metadata for important missing tags**

Add concise entries for common router tags currently auto-filled without descriptions. Include at least:

```python
{"name": "setup", "description": "Complete first-run setup and inspect onboarding readiness."}
{"name": "moderation", "description": "Check text against moderation policies, rules, review queues, and enforcement helpers."}
{"name": "messages", "description": "Use Anthropic-style message endpoints and conversion helpers."}
{"name": "audiobooks", "description": "Create and manage audiobook projects, chapters, narration, alignment, and subtitles."}
{"name": "voice-assistant", "description": "Send voice-assistant commands and manage real-time assistant interactions."}
{"name": "workspaces", "description": "Create and manage research workspaces, memberships, migrations, and active context."}
{"name": "storage", "description": "Manage user files, folders, downloads, quotas, trash, and storage usage."}
{"name": "outputs", "description": "Create, inspect, and retrieve generated outputs and artifacts."}
{"name": "watchlists", "description": "Track recurring sources, runs, alert rules, and monitored topics."}
{"name": "mcp-hub", "description": "Manage MCP hub profiles, external servers, tools, and user-facing MCP connections."}
{"name": "acp", "description": "Experimental Agent Client Protocol sessions, permissions, schedules, triggers, and multiplexing."}
{"name": "sandbox", "description": "Experimental sandbox runs, artifacts, diagnostics, and execution controls."}
```

Expected: the most visible modules in the guide also have descriptions in `/docs` and `/redoc`.

- [x] **Step 3: Update ReDoc tag groups**

Replace the existing `x-tagGroups` categories with groups that mirror the guide:

```python
openapi_schema["x-tagGroups"] = [
    {
        "name": "Start, Auth, And Configuration",
        "tags": ["health", "setup", "authentication", "users", "organizations", "invites", "config", "consent"],
    },
    {
        "name": "Media, Documents, And Ingestion",
        "tags": ["media", "media-embeddings", "chunking", "chunking-templates", "ocr", "web-scraping", "ingestion-sources", "connectors", "collections-feeds", "collections-websub"],
    },
    {
        "name": "Audio, Voice, And Speech",
        "tags": ["audio", "audio-websocket", "audio-jobs", "audiobooks", "voice-assistant", "voice-assistant-ws"],
    },
    {
        "name": "Chat, Characters, And Persona",
        "tags": ["chat", "messages", "chat-dictionaries", "chat-documents", "chat-workflows", "characters", "character-chat-sessions", "character-memory", "character-messages", "persona", "personalization"],
    },
    {
        "name": "Search, RAG, Embeddings, And Evaluation",
        "tags": ["rag-unified", "rag-health", "research", "research-discovery", "research-runs", "research-workspace", "paper-search", "embeddings", "vector-stores", "claims", "feedback", "evaluations", "benchmarks"],
    },
    {
        "name": "Knowledge, Study, And Generated Work",
        "tags": ["notes", "reading", "reading-highlights", "prompts", "prompt-studio", "chatbooks", "flashcards", "quizzes", "study-suggestions", "writing", "manuscripts", "slides", "outputs", "outputs-templates", "data-tables", "files", "storage"],
    },
    {
        "name": "Automation, Jobs, And Integrations",
        "tags": ["jobs", "workflows", "scheduler", "scheduled-tasks", "items", "tasks", "notifications", "email", "meetings", "slack", "discord", "telegram", "integrations"],
    },
    {
        "name": "Admin, Governance, And Operations",
        "tags": ["admin", "audit", "monitoring", "metrics", "billing", "privileges", "resource-governor", "llm", "llamacpp", "mcp-unified", "mcp-hub", "mcp-catalogs", "tools", "moderation"],
    },
    {
        "name": "Experimental And Advanced Surfaces",
        "tags": ["acp", "acp-schedules", "acp-triggers", "acp-permissions", "acp-multiplex", "agent-orchestration", "sandbox", "prototype-workspaces", "sharing", "workspaces", "companion", "guardian", "self-monitoring", "vn-capabilities", "vn-assets", "vn-scripts", "vn-policy", "vn-play"],
    },
]
```

Expected: ReDoc navigation follows the guide's user-facing categories.

- [x] **Step 4: Commit OpenAPI metadata**

Run:

```bash
git add tldw_Server_API/app/main.py
git commit -m "docs: align OpenAPI tag metadata"
```

Expected: commit succeeds with only `main.py` staged.

## Task 4: Verify Documentation And OpenAPI Output

**Files:**
- Read: `Docs/API-related/API_Tags_Index.md`
- Read: `Docs/Published/API-related/API_Tags_Index.md`
- Read: `tldw_Server_API/app/main.py`

- [x] **Step 1: Confirm the published mirror matches**

Run:

```bash
cmp Docs/API-related/API_Tags_Index.md Docs/Published/API-related/API_Tags_Index.md
```

Expected: no output and exit status `0`.

- [x] **Step 2: Run markdown link sanity for local links**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python - <<'PY'
import re
from pathlib import Path

paths = [
    Path("Docs/API-related/API_Tags_Index.md"),
    Path("Docs/Published/API-related/API_Tags_Index.md"),
]
pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
missing: list[str] = []
for doc in paths:
    text = doc.read_text()
    for target in pattern.findall(text):
        if "://" in target or target.startswith("#") or target.startswith("mailto:"):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        resolved = (doc.parent / path_part).resolve()
        if not resolved.exists():
            missing.append(f"{doc}: {target} -> {resolved}")

if missing:
    print("\n".join(missing))
    raise SystemExit(1)
print("local markdown links resolve")
PY
```

Expected: `local markdown links resolve`.

- [x] **Step 3: Run OpenAPI import/schema smoke check**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python - <<'PY'
from tldw_Server_API.app.main import app

schema = app.openapi()
tags = {
    tag.get("name")
    for tag in schema.get("tags", [])
    if isinstance(tag, dict)
}
required = {
    "chat",
    "media",
    "rag-unified",
    "embeddings",
    "audio",
    "notes",
    "jobs",
    "admin",
    "mcp-unified",
    "acp",
    "sandbox",
}
missing = sorted(required - tags)
if missing:
    raise SystemExit(f"missing required tags: {missing}")
groups = schema.get("x-tagGroups", [])
if not groups:
    raise SystemExit("x-tagGroups missing")
print(f"openapi tags={len(tags)} groups={len(groups)}")
PY
```

Expected: prints tag and group counts; no exception.

- [x] **Step 4: Run Bandit on touched Python**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit -r tldw_Server_API/app/main.py -f json -o /tmp/bandit_api_module_docs.json
```

Expected: Bandit completes. Review `/tmp/bandit_api_module_docs.json`; if it reports only pre-existing unrelated findings, record that. Fix any new findings caused by this task.

- [x] **Step 5: Commit verification notes if task record is updated**

Use Backlog.md MCP or CLI to add verification notes to `TASK-12027`. If the available MCP surface does not expose task editing, use:

```bash
backlog task edit TASK-12027 --notes "Verification: cmp mirror passed; markdown local links resolve; OpenAPI schema smoke passed; Bandit run recorded at /tmp/bandit_api_module_docs.json."
```

Expected: the task records the verification results.

## Task 5: Finalize

**Files:**
- Modify: `backlog/tasks/task-12027 - Improve-user-facing-API-module-documentation.md`

- [x] **Step 1: Review final diff**

Run:

```bash
git status --short
git diff --stat HEAD~2..HEAD
git diff -- Docs/API-related/API_Tags_Index.md Docs/Published/API-related/API_Tags_Index.md tldw_Server_API/app/main.py
```

Expected: diff contains only documentation guide changes and OpenAPI metadata descriptions/grouping.

- [x] **Step 2: Update Backlog final summary**

Use Backlog.md MCP or CLI to mark acceptance criteria and add a final summary. If using CLI:

```bash
backlog task edit TASK-12027 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --final-summary "Expanded the API tag index into a grouped module capability guide and aligned OpenAPI tag descriptions/ReDoc groups so users can browse by goal. Endpoint behavior was unchanged."
```

Expected: task file records completed acceptance criteria and final summary.

- [x] **Step 3: Commit task finalization**

Run:

```bash
git add "backlog/tasks/task-12027 - Improve-user-facing-API-module-documentation.md"
git commit -m "docs: finalize API module documentation task"
```

Expected: commit succeeds if the task file changed. If Backlog.md tooling did not modify the task, skip this commit and record why in the final response.

## Self-Review Notes

- Spec coverage: Tasks 2 and 3 implement the two deliverables; Task 4 covers required verification; Task 5 covers task finalization.
- Placeholder scan: This plan contains no TBD/TODO/fill-in placeholders. It includes exact commands, paths, and expected outputs.
- Scope control: The plan documents modules by OpenAPI tag and does not require endpoint behavior changes or full rewrites of existing API guides.

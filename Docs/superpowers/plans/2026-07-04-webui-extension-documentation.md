# WebUI And Extension Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a top-level user-facing WebUI & Extension documentation section that explains available pages, feature sets, and larger systems.

**Architecture:** Author source docs under `Docs/User_Guides/WebUI/` so the existing `Docs/User_Guides` publishing path includes the section. Use `apps/packages/ui/src/routes/route-metadata.ts`, `apps/tldw-frontend/pages`, and `apps/packages/ui/src/routes/sidepanel-route-registry.tsx` as the route truth. Keep `Docs/Published` generated and unchanged.

**Tech Stack:** Markdown, MkDocs navigation, Backlog.md MCP tracking, Python stdlib verification scripts.

---

## File Structure

- `Docs/User_Guides/WebUI/index.md`: landing page for WebUI and extension documentation.
- `Docs/User_Guides/WebUI/Page_Feature_Index.md`: route and feature index grouped by user goal.
- `Docs/User_Guides/WebUI/Start_Account_Settings.md`: setup, auth/account, settings, health, and configuration surfaces.
- `Docs/User_Guides/WebUI/Chat_Characters_Assistants.md`: chat, characters, persona, companion, agents, dictionaries, world books, chat workflows, and chat workspace.
- `Docs/User_Guides/WebUI/Knowledge_Media_Sources.md`: media, sources, connectors, collections, reading, notes, knowledge, research/document workspaces, sharing, and public links.
- `Docs/User_Guides/WebUI/Audio_Speech_Audiobooks.md`: speech overview, STT, TTS, audio alias, and audiobook studio.
- `Docs/User_Guides/WebUI/Study_Writing_Artifacts.md`: evaluations, flashcards, quiz, prompts, prompt studio, chatbooks, writing, presentations, data tables, kanban, repo2txt, and content review.
- `Docs/User_Guides/WebUI/Automation_Admin_Operations.md`: scheduled tasks, watchlists, integrations, workflow editor, MCP hub, ACP playground, model playground, skills, moderation, claims review, notifications, and admin/operator pages.
- `Docs/User_Guides/WebUI/Extension_Sidepanel.md`: browser extension options, sidepanel, clipper, page chat, persona/companion/agent sidepanel, flashcards, settings, and extension-specific setup.
- `Docs/User_Guides/WebUI/Experimental_And_Specialized.md`: visual novel tools, prototype workspaces, hosted-only account/billing pages, legacy aliases, and internal QA/debug pages.
- `Docs/User_Guides/index.md`: update the user guide map to point at the new section.
- `Docs/mkdocs.yml`: add a top-level `WebUI & Extension` navigation section.
- `Docs/superpowers/specs/2026-07-04-webui-extension-documentation-design.md`: approved design spec.
- `Docs/superpowers/plans/2026-07-04-webui-extension-documentation.md`: this implementation plan.
- `backlog/tasks/task-12028 - Improve-WebUI-and-extension-user-facing-documentation.md`: Backlog.md task record.

## Task 1: Confirm Route And Docs Inputs

**Files:**
- Read: `apps/packages/ui/src/routes/route-metadata.ts`
- Read: `apps/packages/ui/src/routes/route-registry.tsx`
- Read: `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
- Read: `apps/tldw-frontend/pages`
- Read: `Docs/User_Guides/WebUI_Extension`
- Read: `apps/extension/docs/index.md`
- Read: `apps/extension/docs/sidebar/index.md`
- Read: `apps/extension/docs/shortcuts.md`

- [ ] **Step 1: List WebUI pages**

Run:

```bash
find apps/tldw-frontend/pages -maxdepth 4 -type f \( -name '*.tsx' -o -name '*.ts' \) | sort
```

Expected: routes include `/chat`, `/knowledge`, `/media`, `/settings`, `/admin`, `/tts`, `/stt`, `/flashcards`, `/vn-assets`, `/vn-play`, and debug/internal pages.

- [ ] **Step 2: List shared route metadata labels**

Run:

```bash
rg -n "path:|label:|group:|surface:|availability:|rationale:" apps/packages/ui/src/routes/route-metadata.ts
```

Expected: output includes route groups `start`, `chat`, `knowledge`, `media_library`, `settings`, `operations`, `workspace`, `audio`, `study`, `safety`, `specialized`, `documentation`, `account`, and `extension`.

- [ ] **Step 3: List extension sidepanel routes**

Run:

```bash
sed -n '1,220p' apps/packages/ui/src/routes/sidepanel-route-registry.tsx
```

Expected: sidepanel routes include `/`, `/chat`, `/agent`, `/companion`, `/companion/conversation`, `/clipper`, `/persona`, `/flashcards`, `/settings`, and `/error-boundary-test`.

- [ ] **Step 4: Confirm stable docs to link**

Run:

```bash
find Docs/User_Guides/WebUI_Extension apps/extension/docs -maxdepth 2 -type f \( -name '*.md' -o -name 'README.md' \) | sort
```

Expected: stable docs include WebUI extension user guides, Knowledge QA, chat pages, STT/TTS, chatbooks, flashcards, extension index, sidebar docs, and shortcuts. WIP PRDs are not copied.

## Task 2: Add WebUI Section Landing And Feature Index

**Files:**
- Create: `Docs/User_Guides/WebUI/index.md`
- Create: `Docs/User_Guides/WebUI/Page_Feature_Index.md`

- [ ] **Step 1: Create the landing page**

Create `Docs/User_Guides/WebUI/index.md` with:

```markdown
# WebUI And Extension Guide

This section explains what the WebUI and browser extension let you do. Use it to choose the right page, understand which surface a feature belongs to, and find deeper setup or workflow docs.

## Surfaces

| Surface | What it means |
| --- | --- |
| WebUI | The full browser application served by the Next.js app. |
| Extension options | The browser extension's full-page options UI, usually using the same shared route components as the WebUI. |
| Extension sidepanel | Compact browser-adjacent tools for chat, clipping, persona, companion, agent, and flashcard review workflows. |
| Shared UI | A route or feature implemented in the shared UI package and reused by multiple surfaces. |
| Admin/operator | Deployment, server, org, model runtime, monitoring, usage, billing, and governance pages. |
| Hosted-only | Account or billing pages that mainly apply to hosted or multi-user deployments. |
| Experimental/labs | Beta, specialized, or advanced workflows that may require extra server capability. |
| Legacy alias | A compatibility route that redirects or points users to a newer canonical page. |
| Internal QA/debug | Test or preview pages that normal users should ignore. |

## Start Here

| Need | Start with |
| --- | --- |
| Find a page or feature | [Page and feature index](Page_Feature_Index.md) |
| Connect to a server or configure auth | [Start, account, and settings](Start_Account_Settings.md) |
| Chat with models, characters, personas, or assistants | [Chat, characters, and assistants](Chat_Characters_Assistants.md) |
| Add sources, search knowledge, or manage media | [Knowledge, media, and sources](Knowledge_Media_Sources.md) |
| Use transcription, TTS, or audiobook workflows | [Audio, speech, and audiobooks](Audio_Speech_Audiobooks.md) |
| Study, write, generate artifacts, or review content | [Study, writing, and artifacts](Study_Writing_Artifacts.md) |
| Automate, integrate, moderate, or administer a server | [Automation, admin, and operations](Automation_Admin_Operations.md) |
| Use browser-sidepanel workflows | [Extension sidepanel](Extension_Sidepanel.md) |
| Understand advanced, hosted, alias, or debug pages | [Experimental and specialized pages](Experimental_And_Specialized.md) |

## Related Existing Guides

- [Current WebUI user guide](../WebUI_Extension/User_Guide.md)
- [Knowledge QA guide](../WebUI_Extension/Knowledge_QA_Guide.md)
- [Chat pages](../WebUI_Extension/Chat_Pages.md)
- [Getting started with STT and TTS](../WebUI_Extension/Getting-Started-STT_and_TTS.md)
- [Browser extension user docs](../../../apps/extension/docs/index.md)
```

Expected: file introduces the new section and links every new feature-set page.

- [ ] **Step 2: Create the page and feature index**

Create `Docs/User_Guides/WebUI/Page_Feature_Index.md` with grouped tables for:

```text
Start, Account, And Settings
Chat, Characters, And Assistants
Knowledge, Media, And Sources
Audio, Speech, And Audiobooks
Study, Writing, And Artifacts
Automation, Admin, And Operations
Extension Sidepanel
Experimental, Specialized, Hosted, Alias, And Debug
```

Each table must use:

```markdown
| Page or feature | Surface/status | What it lets you do | Common uses | More docs |
| --- | --- | --- | --- | --- |
```

Expected: the index covers at least the audited root routes from `ROUTE_METADATA` and labels hidden/gated routes clearly.

- [ ] **Step 3: Commit landing and index**

Run:

```bash
git add Docs/User_Guides/WebUI/index.md Docs/User_Guides/WebUI/Page_Feature_Index.md
git commit -m "docs: add WebUI extension documentation index"
```

Expected: commit succeeds with only the two new WebUI index docs staged.

## Task 3: Add Core Feature-Set Pages

**Files:**
- Create: `Docs/User_Guides/WebUI/Start_Account_Settings.md`
- Create: `Docs/User_Guides/WebUI/Chat_Characters_Assistants.md`
- Create: `Docs/User_Guides/WebUI/Knowledge_Media_Sources.md`
- Create: `Docs/User_Guides/WebUI/Audio_Speech_Audiobooks.md`

- [ ] **Step 1: Create Start, Account, And Settings**

Cover:

```text
/, /setup, /login, /signup, /account, /profile, /privileges, /config, /settings,
/settings/tldw, /settings/model, /settings/provider-keys, /settings/chat,
/settings/prompt, /settings/knowledge, /settings/rag, /settings/speech,
/settings/evaluations, /settings/health, /settings/ui, /settings/splash,
/settings/quick-ingest, /settings/image-generation, /settings/share,
/settings/processed, /settings/about, /billing, /404
```

Expected: explain setup/readiness, auth modes, profiles, health diagnostics, provider/model settings, quick ingest settings, and hosted-only account/billing routes.

- [ ] **Step 2: Create Chat, Characters, And Assistants**

Cover:

```text
/chat, /chat/agent, /quick-chat-popout, /persona, /companion,
/companion/conversation, /characters, /agents, /agent-tasks,
/chat-workflows, /chat-workspace, /dictionaries, /world-books,
/settings/chat, /settings/chat-dictionaries, /settings/characters,
/settings/world-books
```

Expected: explain normal chat, character/persona workflows, companion/agent routes, chat dictionaries, world books, chat workflows, and workspace chat.

- [ ] **Step 3: Create Knowledge, Media, And Sources**

Cover:

```text
/knowledge, /search, /research, /workspaces, /research-workspace,
/document-workspace, /media, /media-multi, /media/[id]/view,
/media-trash, /review, /items, /collections, /reading, /notes,
/sources, /sources/new, /sources/[sourceId], /connectors,
/connectors/browse, /connectors/jobs, /connectors/sources,
/shared, /share/[token]
```

Expected: explain media library, source intake, RAG/Knowledge QA, research workspaces, document workspaces, reading/collections, notes, sharing, and connector status.

- [ ] **Step 4: Create Audio, Speech, And Audiobooks**

Cover:

```text
/speech, /stt, /tts, /audio, /audiobook-studio, /settings/speech
```

Expected: explain speech overview, transcription, TTS, legacy audio alias, voice/provider readiness, and audiobook production.

- [ ] **Step 5: Commit core feature-set pages**

Run:

```bash
git add Docs/User_Guides/WebUI/Start_Account_Settings.md Docs/User_Guides/WebUI/Chat_Characters_Assistants.md Docs/User_Guides/WebUI/Knowledge_Media_Sources.md Docs/User_Guides/WebUI/Audio_Speech_Audiobooks.md
git commit -m "docs: document core WebUI feature sets"
```

Expected: commit succeeds with only those four feature-set docs staged.

## Task 4: Add Study, Operations, Extension, And Specialized Pages

**Files:**
- Create: `Docs/User_Guides/WebUI/Study_Writing_Artifacts.md`
- Create: `Docs/User_Guides/WebUI/Automation_Admin_Operations.md`
- Create: `Docs/User_Guides/WebUI/Extension_Sidepanel.md`
- Create: `Docs/User_Guides/WebUI/Experimental_And_Specialized.md`

- [ ] **Step 1: Create Study, Writing, And Artifacts**

Cover:

```text
/evaluations, /flashcards, /quiz, /prompts, /prompt-studio,
/chatbooks, /chatbooks-playground, /writing-playground,
/presentation-studio, /presentation-studio/new,
/presentation-studio/[projectId], /data-tables, /kanban,
/repo2txt, /content-review
```

Expected: explain study workflows, prompt library/studio, chatbooks, writing, slides, data tables, kanban, repo2txt, and content review.

- [ ] **Step 2: Create Automation, Admin, And Operations**

Cover:

```text
/integrations, /scheduled-tasks, /scheduled-tasks/results,
/watchlists, /workflow-editor, /mcp-hub, /acp-playground,
/model-playground, /skills, /notifications, /moderation,
/moderation/rules, /moderation-playground, /claims-review,
/admin, /admin/server, /admin/api-keys, /admin/billing,
/admin/data-ops, /admin/integrations, /admin/llamacpp,
/admin/maintenance, /admin/mlx, /admin/monitoring,
/admin/orgs, /admin/rate-limiting, /admin/rbac,
/admin/sources, /admin/usage, /admin/watchlists-items,
/admin/watchlists-runs
```

Expected: explain automation/integration pages, MCP/ACP, moderation/claims, notifications, and admin/operator surfaces.

- [ ] **Step 3: Create Extension Sidepanel**

Cover:

```text
Extension options page, background proxy, context menu actions,
sidepanel /, /chat, /agent, /companion, /companion/conversation,
/clipper, /persona, /flashcards, /settings, /error-boundary-test,
copilot popup, web clipper content script, HF pull content script
```

Expected: explain what is extension-specific, which workflows are compact sidepanel versions of shared WebUI routes, and what setup failures are extension-specific.

- [ ] **Step 4: Create Experimental And Specialized**

Cover:

```text
/vn-assets, /vn-play, /vn-play/sessions/[sessionId]/generations,
/vn-scripts, /prototype-workspaces, /composer-variants-preview,
/onboarding-test, /__debug__/sidepanel-chat,
/__debug__/mermaid-chat-cards, /__debug__/sidepanel-error-boundary,
hosted-only account/billing routes, legacy aliases /audio, /search,
/prompt-studio, /review, /moderation-playground
```

Expected: explain that these routes can be beta, deployment-gated, compatibility aliases, or internal QA/debug routes.

- [ ] **Step 5: Commit remaining feature pages**

Run:

```bash
git add Docs/User_Guides/WebUI/Study_Writing_Artifacts.md Docs/User_Guides/WebUI/Automation_Admin_Operations.md Docs/User_Guides/WebUI/Extension_Sidepanel.md Docs/User_Guides/WebUI/Experimental_And_Specialized.md
git commit -m "docs: document advanced WebUI and extension surfaces"
```

Expected: commit succeeds with only those four docs staged.

## Task 5: Update Discovery And Navigation

**Files:**
- Modify: `Docs/User_Guides/index.md`
- Modify: `Docs/mkdocs.yml`

- [ ] **Step 1: Update User Guides index**

In `Docs/User_Guides/index.md`, update the "Choose Your Surface" table so the WebUI row points to `WebUI/index.md` and mentions browser extension sidepanel coverage. Add a common workflow link to `WebUI/Page_Feature_Index.md`.

Expected: users starting from the user guide map can find the new section.

- [ ] **Step 2: Update MkDocs navigation**

In `Docs/mkdocs.yml`, add a top-level nav section after `Getting Started`:

```yaml
  - WebUI & Extension:
      - Overview: User_Guides/WebUI/index.md
      - Page And Feature Index: User_Guides/WebUI/Page_Feature_Index.md
      - Start, Account, And Settings: User_Guides/WebUI/Start_Account_Settings.md
      - Chat, Characters, And Assistants: User_Guides/WebUI/Chat_Characters_Assistants.md
      - Knowledge, Media, And Sources: User_Guides/WebUI/Knowledge_Media_Sources.md
      - Audio, Speech, And Audiobooks: User_Guides/WebUI/Audio_Speech_Audiobooks.md
      - Study, Writing, And Artifacts: User_Guides/WebUI/Study_Writing_Artifacts.md
      - Automation, Admin, And Operations: User_Guides/WebUI/Automation_Admin_Operations.md
      - Extension Sidepanel: User_Guides/WebUI/Extension_Sidepanel.md
      - Experimental And Specialized: User_Guides/WebUI/Experimental_And_Specialized.md
```

Keep the existing `User Guides -> WebUI and Extension` subsection as a compact legacy/deep-links area unless the final docs make it redundant enough to simplify safely.

- [ ] **Step 3: Commit navigation updates**

Run:

```bash
git add Docs/User_Guides/index.md Docs/mkdocs.yml
git commit -m "docs: expose WebUI extension section in docs nav"
```

Expected: commit succeeds with only the user-guide map and MkDocs nav staged.

## Task 6: Verify Documentation

**Files:**
- Read: `Docs/User_Guides/WebUI/*.md`
- Read: `Docs/User_Guides/index.md`
- Read: `Docs/mkdocs.yml`
- Read: `Docs/Published`

- [ ] **Step 1: Run local markdown link check**

Run:

```bash
python - <<'PY'
import re
from pathlib import Path

paths = sorted(Path("Docs/User_Guides/WebUI").glob("*.md")) + [
    Path("Docs/User_Guides/index.md"),
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

- [ ] **Step 2: Run MkDocs navigation source-target check**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

mkdocs = Path("Docs/mkdocs.yml").read_text()
targets = re.findall(r": ([A-Za-z0-9_./-]+\.md)", mkdocs)
missing = []
for target in targets:
    source = Path("Docs") / target
    if not source.exists():
        missing.append(f"{target} -> {source}")

if missing:
    print("\n".join(missing))
    raise SystemExit(1)
print(f"mkdocs source nav targets exist: {len(targets)}")
PY
```

Expected: prints a target count and exits `0`.

- [ ] **Step 3: Confirm generated Published docs are unchanged**

Run:

```bash
git diff --quiet dev -- Docs/Published && printf 'Docs/Published unchanged against dev\n'
```

Expected: `Docs/Published unchanged against dev`.

- [ ] **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit status `0`.

- [ ] **Step 5: Record Bandit skip**

No Python files should be changed in this task. Record in Backlog that Bandit is not applicable because the touched files are Markdown and MkDocs YAML only.

## Task 7: Finalize Backlog Task

**Files:**
- Modify: `backlog/tasks/task-12028 - Improve-WebUI-and-extension-user-facing-documentation.md`

- [ ] **Step 1: Update Backlog verification and acceptance criteria**

Use Backlog MCP:

```text
task_edit TASK-12028:
- check AC 1-5
- check DoD 1-5
- notes: verification results
- final summary: what changed and why
```

Expected: task shows completed acceptance criteria and a concise final summary.

- [ ] **Step 2: Commit task finalization**

Run:

```bash
git add "backlog/tasks/task-12028 - Improve-WebUI-and-extension-user-facing-documentation.md"
git commit -m "docs: finalize WebUI extension documentation task"
```

Expected: commit succeeds if the task file changed.

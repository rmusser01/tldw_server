# First-Run Assistant Setup — Design Document

**Date:** 2026-04-07
**Status:** Approved
**Approach:** Template-Seeded Wizard (Approach A)

## Summary

Add a guided "build your assistant" setup flow on top of Persona Garden. The flow combines two capabilities:

1. **Archetype templates** — pre-built assistant profiles (Research Assistant, Study Buddy, etc.) that pre-fill all wizard steps
2. **First-launch gate** — a soft gate that intercepts new users before the main UI, guiding them through creating their first persona + connecting tools

Users can skip at any time and resume later. Skipped users get a bare platform with periodic gentle nudges.

---

## Archetype Template Data Model

Each archetype is a YAML file under `Config_Files/persona_archetypes/`. The schema:

```yaml
archetype:
  key: "research_assistant"
  label: "Research Assistant"
  tagline: "Helps you find, analyze, and organize research"
  icon: "search"

  persona:
    name: "Research Assistant"
    system_prompt: "You are a focused research assistant..."
    personality_traits: ["thorough", "analytical", "concise"]

  mcp_modules:
    enabled: ["media", "knowledge", "notes"]
    disabled: ["flashcards", "quizzes", "kanban", "slides"]

  suggested_external_servers:
    - "arxiv"
    - "github"
    - "zotero"

  policy:
    confirmation_mode: "destructive_only"
    tool_overrides:
      - tool: "media.ingest"
        requires_confirmation: true
      - tool: "notes.search"
        requires_confirmation: false

  voice_defaults:
    tts_provider: null
    tts_voice: null
    stt_language: "en"

  scope_rules: []

  buddy:
    species: "owl"
    palette: "warm"
    silhouette: "round"

  starter_commands:
    - template_key: "media-search"
    - template_key: "notes-search"
    - template_key: "note-create"
    - custom:
        name: "Summarize source"
        phrases: ["summarize {url}", "tldr {url}"]
        tool_name: "media.ingest"
        slot_map: { source: "url" }
        requires_confirmation: true
```

### Starter Archetypes (6)

| Archetype | Key Modules | Buddy |
|---|---|---|
| Research Assistant | media, knowledge, notes, web search | owl |
| Study Buddy | flashcards, quizzes, knowledge, notes | — |
| Writing Coach | notes, prompts, media | — |
| Project Manager | kanban, notes, filesystem, chats | — |
| Roleplayer | characters, personas, exemplars, chats | — |
| Custom / Blank Canvas | user picks | — |

Archetypes are loaded at startup, validated via Pydantic, cached in memory. Malformed files are logged and skipped.

---

## First-Launch Gate & Wizard Flow

### Route Guard Logic

- On app load, check `GET /api/v1/persona/profiles` — if zero profiles AND no dismissed flag in localStorage, redirect to `/setup/assistant`
- If user previously started but didn't finish (`setup.status == "in_progress"`), show a banner: "Continue setting up your assistant?" linking to the wizard at the saved step
- "Skip for now" sets a localStorage flag → bare platform, no persona active
- Periodic nudges in chat input area, voice button tooltip, and Persona Garden empty state

### Wizard Steps (6)

```
1. "Pick a starting point"   — Archetype picker grid (NEW)
2. "Choose persona"          — Pre-filled from archetype, editable
3. "Voice defaults"          — Pre-filled, review & tweak
4. "Starter commands"        — Pre-filled templates, add/remove
5. "Tools & connections"     — Built-in toggles + external MCP catalog + access controls (ENHANCED)
6. "Test and finish"         — Dry run / live session (existing)
```

### Step 1: Archetype Picker

- Grid of cards showing icon, label, tagline
- "Blank Canvas" always last, visually distinct
- Selecting pre-fills all downstream steps and advances to step 2
- Going back to step 1 and picking a different archetype resets downstream (with confirmation warning)

### Step 5: Tools & Connections (Enhanced)

Three sections:

**Section A — Built-in modules:** Toggle grid of internal MCP modules, pre-set by archetype.

**Section B — External MCP servers:** Hybrid catalog:
- Curated tiles for popular integrations with one-click "Connect" expanding an inline auth form
- "Add custom MCP server" for manual URL entry
- Archetype's `suggested_external_servers` highlighted with "Recommended" badge
- "Test Connection" button validates connectivity and reports discovered tools

**Section C — Access controls (tiered):** Per-server confirmation mode selector (always / destructive-only / never). "Fine-tune per-tool permissions" link opens full policy editor in Persona Garden post-setup.

### Resume Behavior

- `PersonaSetupState` tracks `current_step` and `completed_steps`
- New step value `"archetype"` added to `PersonaSetupStep`
- On resume, wizard opens at saved step with completed steps' data intact

---

## Backend API Changes

### New Endpoints

```
GET  /api/v1/persona/archetypes
     → List of archetype summaries (key, label, tagline, icon)

GET  /api/v1/persona/archetypes/{key}
     → Full archetype template bundle

GET  /api/v1/persona/archetypes/{key}/preview
     → Pre-filled PersonaProfileCreate from template

GET  /api/v1/mcp/catalog
     → Curated external MCP server catalog
     → Sourced from Config_Files/mcp_server_catalog.yaml

POST /api/v1/mcp/catalog/test-connection
     → Test connectivity to external MCP server URL
     → Returns { reachable, tools_discovered, error }
```

### Modified Endpoints

```
POST /api/v1/persona/profiles
     → New optional field: archetype_key: str | None
     → If provided, merges archetype defaults with user overrides
     → Stored on profile for analytics
```

### Schema Changes

```python
PersonaSetupStep = Literal["archetype", "persona", "voice", "commands", "safety", "test"]

# New field on PersonaProfileCreate/Response
archetype_key: str | None = None

# New models
class ArchetypeSummary(BaseModel):
    key: str
    label: str
    tagline: str
    icon: str

class ArchetypeTemplate(ArchetypeSummary):
    persona: ArchetypePersonaDefaults
    mcp_modules: ArchetypeMCPConfig
    suggested_external_servers: list[str]
    policy: ArchetypePolicyDefaults
    voice_defaults: PersonaVoiceDefaults
    scope_rules: list[dict]
    buddy: ArchetypeBuddyDefaults
    starter_commands: list[ArchetypeStarterCommand]

class MCPCatalogEntry(BaseModel):
    key: str
    name: str
    description: str
    url_template: str
    auth_type: str
    category: str
    logo_key: str | None
```

---

## External MCP Server Catalog

Catalog data lives in `Config_Files/mcp_server_catalog.yaml`:

```yaml
catalog:
  - key: "github"
    name: "GitHub"
    description: "Repositories, issues, PRs, and code search"
    url_template: "https://api.github.com"
    auth_type: "bearer"
    category: "development"
    logo_key: "github"
    suggested_for: ["research_assistant", "project_manager"]

  - key: "arxiv"
    name: "arXiv"
    description: "Academic paper search and retrieval"
    url_template: "https://export.arxiv.org/api"
    auth_type: "none"
    category: "research"
    logo_key: "arxiv"
    suggested_for: ["research_assistant", "study_buddy"]

  # ... more entries
```

Connected servers are persisted as persona connections (existing `connections.py` infrastructure). Full per-tool policy editing available from Persona Garden post-setup.

---

## Frontend Component Architecture

### New Components

```
PersonaGarden/
  FirstRunGate.tsx              — Route guard
  FirstRunBanner.tsx            — Resume/nudge banners
  ArchetypePickerStep.tsx       — Grid of archetype cards (step 1)
  ArchetypeCard.tsx             — Individual archetype card
  ToolsConnectionsStep.tsx      — 3-section tools & connections (step 5)
  MCPModuleToggleGrid.tsx       — Built-in module toggles
  MCPExternalCatalog.tsx        — Curated catalog + custom server entry
  MCPServerCatalogCard.tsx      — Individual catalog entry
  MCPAccessControlTier.tsx      — Per-server confirmation selector
```

### Modified Components

```
AssistantSetupWizard.tsx        — Add "archetype" step, accept archetypeKey, pre-fill logic
personaSetupProgress.ts         — Add "archetype" step label and summaries
SetupStarterCommandsStep.tsx    — Accept pre-filled commands from archetype
```

### New Hooks

```
useArchetypeCatalog()           — Fetch + cache archetype list
useArchetypePreview(key)        — Fetch pre-fill state for selected archetype
useFirstRunCheck()              — Profile count + setup state + localStorage flag
useMCPServerCatalog()           — Fetch external server catalog
useMCPConnectionTest(url)       — Test external server connectivity
```

### State Flow

```
ArchetypePickerStep selects key
  → useArchetypePreview(key) returns pre-filled PersonaProfileCreate
  → Wizard holds pre-fill in local state
  → Each step receives pre-fill as defaultValues
  → User edits override (spread semantics, user values win)
  → On finish, POST /profiles with archetype_key
```

### Nudge Placement (skipped users)

- Chat input area: inline banner
- Voice button: tooltip
- Persona Garden tab: empty state CTA
- All dismissible, all link to `/setup/assistant`

---

## Analytics Extensions

### New Event Types

```
archetype_selected           — User picks an archetype
archetype_changed            — User goes back, picks different archetype
external_server_connected    — Successful external MCP connection
external_server_failed       — Connection test failed
connection_test_initiated    — User clicked "Test Connection"
setup_skipped                — User clicked "Skip for now"
setup_resumed                — User resumed from banner
```

### Event Metadata

```json
{
  "archetype_key": "research_assistant",
  "server_key": "github",
  "tools_discovered": 12,
  "error_reason": "timeout"
}
```

---

## Edge Cases

| Scenario | Behavior |
|---|---|
| Pick archetype → edit → go back → pick different | Reset downstream, warn "This will replace your customizations" |
| External MCP server down after setup | Circuit breaker handles runtime; shows "unreachable" in Persona Garden |
| Malformed archetype YAML | Skipped on startup with loguru warning |
| Profiles exist but none completed setup | Banner "Continue setting up?" — no full gate |
| Multiple browser tabs | localStorage flag shared; one tab triggers gate |
| Auth mode switch (single→multi-user) | Per-user first-run check |
| Archetype YAML updated between restarts | Existing profiles unaffected (archetype_key is historical) |

---

## Testing Strategy

- **Unit tests:** Archetype YAML loading + validation, template merge, preview generation, catalog validation
- **Integration tests:** Full wizard API flow (create from archetype, verify defaults), connection test with mock server
- **Frontend tests:** Archetype picker selection/reset, pre-fill propagation, first-run gate logic, resume banner
- **Analytics tests:** Event emission for new types, funnel metrics include archetype step

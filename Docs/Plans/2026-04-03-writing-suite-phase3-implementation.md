# Writing Suite Phase 3: Analysis — AI Tools + Agent Chat

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add AI-powered manuscript analysis (pacing, plot holes, consistency), visualization tools (Story Pulse, Plot Tracker, Event Line, Connection Web), and a context-aware AI agent chat panel to the writing suite.

**Architecture:** New `manuscript_ai_analyses` table (migration V42→V43) stores cached analysis results per scope. Analysis endpoints gather manuscript content, construct structured LLM prompts, call via the existing `perform_chat_api_call_async()` adapter, parse structured JSON responses, and cache results. Frontend adds 4 lazy-loaded analysis modals (using Ant Design components + existing Cytoscape.js) and an AI agent inspector tab. Scene updates mark matching analyses as stale.

**Tech Stack:** FastAPI, SQLite, `perform_chat_api_call_async` (LLM adapter), Cytoscape.js (already installed), Ant Design Progress/Timeline/Table, React Query

---

## Task 1: DB Migration V42→V43 — AI Analyses Table

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`

**Step 1: Add migration SQL + method + wiring**

Same pattern as V41→V42. One new table:

```sql
CREATE TABLE IF NOT EXISTS manuscript_ai_analyses (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  scope_type    TEXT NOT NULL CHECK(scope_type IN ('scene','chapter','part','project')),
  scope_id      TEXT NOT NULL,
  analysis_type TEXT NOT NULL,
  provider      TEXT,
  model         TEXT,
  result_json   TEXT NOT NULL DEFAULT '{}',
  score         REAL,
  stale         BOOLEAN NOT NULL DEFAULT 0,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT 0,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_maa_scope ON manuscript_ai_analyses(scope_type, scope_id);
CREATE INDEX IF NOT EXISTS idx_maa_project_type ON manuscript_ai_analyses(project_id, analysis_type);
CREATE INDEX IF NOT EXISTS idx_maa_stale ON manuscript_ai_analyses(stale);
CREATE INDEX IF NOT EXISTS idx_maa_deleted ON manuscript_ai_analyses(deleted);
```

Plus 4 sync triggers (create/update/delete/undelete). Version bump to 43.

**Step 2: Commit**
```bash
git commit -m "feat(db): add manuscript_ai_analyses table (migration V42→V43)"
```

---

## Task 2: ManuscriptDB — Analysis CRUD + Stale Marking

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`
- Create: `tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py`

**Step 1: Write tests**

```python
class TestAnalysisCRUD:
    def test_create_and_get(self, mdb): ...
    def test_list_by_project(self, mdb): ...
    def test_list_by_scope(self, mdb): ...
    def test_mark_stale_on_scope(self, mdb): ...
    def test_delete_analysis(self, mdb): ...

class TestStaleMarking:
    def test_scene_update_marks_analyses_stale(self, mdb):
        """When scene content changes, analyses for that scene should be marked stale."""
        ...
```

**Step 2: Add CRUD methods**

- `create_analysis(project_id, scope_type, scope_id, analysis_type, result, *, score=None, provider=None, model=None, analysis_id=None) -> str`
  - `result` is a dict, serialize to `result_json`
- `get_analysis(analysis_id) -> dict | None`
  - Deserialize `result_json` to `result`
- `list_analyses(project_id, *, scope_type=None, scope_id=None, analysis_type=None, include_stale=False) -> list[dict]`
- `mark_analyses_stale(scope_type, scope_id)` — `UPDATE SET stale = 1 WHERE scope_type = ? AND scope_id = ? AND stale = 0 AND deleted = 0`
- `soft_delete_analysis(analysis_id, expected_version)`

**Step 3: Modify `update_scene`** — After updating scene content, call `self.mark_analyses_stale("scene", scene_id)` if `content_plain` changed.

**Step 4: Commit**
```bash
git commit -m "feat(manuscripts): add AI analysis CRUD with stale marking"
```

---

## Task 3: Analysis Service — LLM-Powered Content Analysis

**Files:**
- Create: `tldw_Server_API/app/core/Writing/manuscript_analysis.py`
- Create: `tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py`

**Step 1: Create analysis service**

This module provides structured LLM analysis of manuscript content. It uses `perform_chat_api_call_async` from `chat_service.py`.

```python
"""Manuscript analysis service — structured LLM analysis of writing content."""
from __future__ import annotations

import json
from typing import Any

from loguru import logger


# ── Prompt templates ────────────────────────────────────

PACING_PROMPT = """Analyze the pacing of the following text. Return a JSON object with:
- "pacing": float 0-1 (0=very slow, 1=very fast)
- "tension": float 0-1
- "atmosphere": float 0-1
- "engagement": float 0-1
- "assessment": string (1-2 sentence summary)
- "beats": list of strings (key story beats found)

Text:
{text}

Return ONLY valid JSON, no markdown."""

PLOT_HOLES_PROMPT = """Analyze the following manuscript for plot holes and inconsistencies.
Characters: {characters}
World Info: {world_info}
Text: {text}

Return a JSON object with:
- "plot_holes": list of objects, each with "title", "description", "severity" (low/medium/high/critical), "location_hint"
- "inconsistencies": list of strings

Return ONLY valid JSON, no markdown."""

CONSISTENCY_PROMPT = """Check the following manuscript for character and world consistency.
Characters: {characters}
World Info: {world_info}
Text: {text}

Return a JSON object with:
- "character_issues": list of objects with "character_name", "issue", "severity"
- "world_issues": list of objects with "entity_name", "issue", "severity"
- "timeline_issues": list of strings
- "overall_score": float 0-1 (1=perfectly consistent)

Return ONLY valid JSON, no markdown."""


async def analyze_pacing(
    text: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Analyze pacing/tension/atmosphere of a text passage."""
    return await _run_analysis(
        PACING_PROMPT.format(text=text[:8000]),
        provider=provider, model=model, api_key=api_key,
    )


async def analyze_plot_holes(
    text: str,
    characters: str = "",
    world_info: str = "",
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Detect plot holes and inconsistencies."""
    return await _run_analysis(
        PLOT_HOLES_PROMPT.format(text=text[:12000], characters=characters[:2000], world_info=world_info[:2000]),
        provider=provider, model=model, api_key=api_key,
    )


async def analyze_consistency(
    text: str,
    characters: str = "",
    world_info: str = "",
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Check character and world consistency."""
    return await _run_analysis(
        CONSISTENCY_PROMPT.format(text=text[:12000], characters=characters[:2000], world_info=world_info[:2000]),
        provider=provider, model=model, api_key=api_key,
    )


async def _run_analysis(
    prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Send analysis prompt to LLM and parse JSON response."""
    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async
    except ImportError:
        logger.warning("Chat service not available for analysis")
        return {"error": "LLM service unavailable"}

    kwargs: dict[str, Any] = {
        "input_data": prompt,
        "custom_prompt_input": "You are a literary analysis assistant. Respond only with valid JSON.",
        "temp": 0.3,
    }
    if provider:
        kwargs["api_endpoint"] = provider
    if model:
        kwargs["api_key"] = model  # model goes here per the adapter interface
    if api_key:
        kwargs["api_key"] = api_key

    try:
        response = await perform_chat_api_call_async(**kwargs)
        # Parse response — handle various formats
        content = ""
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
            elif "content" in response:
                content = response["content"]
        elif isinstance(response, str):
            content = response

        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        if content.startswith("json"):
            content = content[4:].strip()

        return json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse analysis JSON: {}", exc)
        return {"error": "Failed to parse LLM response", "raw": content[:500]}
    except Exception as exc:
        logger.error("Analysis LLM call failed: {}", exc)
        return {"error": str(exc)}
```

**Step 2: Write basic tests (mock LLM)**

```python
"""Tests for manuscript analysis service."""
import pytest
from unittest.mock import AsyncMock, patch
from tldw_Server_API.app.core.Writing.manuscript_analysis import (
    analyze_pacing,
    analyze_plot_holes,
    analyze_consistency,
)

@pytest.mark.asyncio
async def test_analyze_pacing_parses_json():
    mock_response = {
        "choices": [{"message": {"content": '{"pacing": 0.7, "tension": 0.5, "atmosphere": 0.6, "engagement": 0.8, "assessment": "Good pace", "beats": ["intro"]}'}}]
    }
    with patch("tldw_Server_API.app.core.Writing.manuscript_analysis.perform_chat_api_call_async",
               new_callable=AsyncMock, return_value=mock_response):
        result = await analyze_pacing("Some text")
        assert result["pacing"] == 0.7
        assert "assessment" in result

@pytest.mark.asyncio
async def test_analyze_pacing_handles_markdown_fences():
    mock_response = {
        "choices": [{"message": {"content": '```json\n{"pacing": 0.5}\n```'}}]
    }
    with patch("tldw_Server_API.app.core.Writing.manuscript_analysis.perform_chat_api_call_async",
               new_callable=AsyncMock, return_value=mock_response):
        result = await analyze_pacing("Text")
        assert result["pacing"] == 0.5

@pytest.mark.asyncio
async def test_analyze_pacing_handles_llm_error():
    with patch("tldw_Server_API.app.core.Writing.manuscript_analysis.perform_chat_api_call_async",
               new_callable=AsyncMock, side_effect=RuntimeError("API down")):
        result = await analyze_pacing("Text")
        assert "error" in result
```

**Step 3: Commit**
```bash
git commit -m "feat(manuscripts): add manuscript analysis service with LLM integration"
```

---

## Task 4: Pydantic Schemas + API Endpoints for Analysis

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Create: `tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py`

**Step 1: Add schemas**

```python
class ManuscriptAnalysisRequest(BaseModel):
    analysis_types: list[str] = Field(
        default=["pacing", "tension", "mood"],
        description="Types: pacing, tension, mood, plot_holes, consistency",
    )
    provider: str | None = Field(None, description="LLM provider override")
    model: str | None = Field(None, description="Model override")

class ManuscriptAnalysisResponse(BaseModel):
    id: str
    project_id: str
    scope_type: str
    scope_id: str
    analysis_type: str
    result: dict[str, Any]
    score: float | None = None
    stale: bool = False
    provider: str | None = None
    model: str | None = None
    created_at: datetime
    last_modified: datetime
    version: int

class ManuscriptAnalysisListResponse(BaseModel):
    analyses: list[ManuscriptAnalysisResponse]
    total: int
```

**Step 2: Add endpoints**

- `POST /scenes/{scene_id}/analyze` — Gather scene content, run analysis, store result
- `POST /chapters/{chapter_id}/analyze` — Gather all scenes in chapter, analyze
- `POST /projects/{project_id}/analyze/plot-holes` — Gather all content + characters + world info, detect plot holes
- `POST /projects/{project_id}/analyze/consistency` — Check character/world consistency
- `GET /projects/{project_id}/analyses` — List cached results (query: scope_type, analysis_type, include_stale)

The analysis endpoints call functions from `manuscript_analysis.py`, store results via `ManuscriptDBHelper.create_analysis()`, and return the stored analysis.

**Step 3: Write integration test (mock LLM)**

Test with mocked `perform_chat_api_call_async` to verify:
- Analysis endpoint returns structured result
- Result is cached in DB
- List endpoint returns cached results
- Scene update marks analyses as stale

**Step 4: Commit**
```bash
git commit -m "feat(manuscripts): add analysis endpoints with LLM-powered content analysis"
```

---

## Task 5: Frontend — Extend InspectorTabKey + API Service

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlayground.types.ts`
- Modify: `apps/packages/ui/src/services/writing-playground.ts`

**Step 1: Add "agent" to InspectorTabKey**

```typescript
export type InspectorTabKey = "sampling" | "context" | "setup" | "inspect" | "characters" | "research" | "agent"
```

**Step 2: Add analysis + agent API functions**

```typescript
// ── Analysis ────────────────────────────────────────────
export async function analyzeScene(sceneId: string, data?: { analysis_types?: string[]; provider?: string; model?: string }) { ... }
export async function analyzeChapter(chapterId: string, data?: Record<string, unknown>) { ... }
export async function analyzeProjectPlotHoles(projectId: string, data?: Record<string, unknown>) { ... }
export async function analyzeProjectConsistency(projectId: string, data?: Record<string, unknown>) { ... }
export async function listManuscriptAnalyses(projectId: string, params?: { scope_type?: string; analysis_type?: string; include_stale?: boolean }) { ... }
```

**Step 3: Commit**
```bash
git commit -m "feat(writing): add analysis API service and agent tab key"
```

---

## Task 6: Frontend — Analysis Modals (Story Pulse, Plot Tracker, Event Line, Connection Web)

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/WritingAnalysisModalHost.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/modals/StoryPulseModal.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/modals/PlotTrackerModal.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/modals/EventLineModal.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/modals/ConnectionWebModal.tsx`

### StoryPulseModal
Uses Ant Design `Progress` bars per metric (pacing, tension, atmosphere, engagement) for each chapter. Calls `analyzeChapter` for each chapter, displays scores as horizontal bars with labels. No external chart library needed — Progress component is clean and lightweight.

### PlotTrackerModal
Uses Ant Design `Table` showing:
- Open plot lines (from `listManuscriptPlotLines`)
- Plot holes with severity badges (from `listManuscriptPlotHoles`)
- "AI Detect" button calls `analyzeProjectPlotHoles` to auto-detect new holes

### EventLineModal
Uses Ant Design `Timeline` component showing plot events from `listManuscriptPlotLines` → `listPlotEvents`, color-coded by event_type:
- setup=blue, conflict=red, action=orange, emotional=purple, plot=default, resolution=green

### ConnectionWebModal
Reuses the Cytoscape.js pattern from `NotesGraphModal.tsx`:
- Nodes: characters (blue), factions (green), locations (orange) from world info
- Edges: relationships from `listManuscriptRelationships`
- Layout: dagre (already registered)

### WritingAnalysisModalHost
Lazy-loaded wrapper that renders the active modal based on state:
```typescript
const LazyStoryPulse = React.lazy(() => import("./modals/StoryPulseModal"))
// etc.
```

**Commit:**
```bash
git commit -m "feat(writing): add analysis modals (Story Pulse, Plot Tracker, Event Line, Connection Web)"
```

---

## Task 7: Frontend — AIAgentTab Component

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/AIAgentTab.tsx`

A compact chat interface in the inspector panel:

- **Mode selector**: Segmented control with Quick / Planning / Brainstorm
- **Context display**: Shows "Context: Scene 'X' + 3 characters + 2 world entries"
- **Message history**: Scrollable list of user/assistant messages
- **Input**: TextArea + Send button
- **Integration**: Uses `tldwChat.sendMessage()` or direct fetch to `/api/v1/chat/completions`
- **Context injection**: Prepends scene content + character sheets + world info to system message
- **Conversation storage**: Messages stored in component state (per-session, not persisted to DB in Phase 3)

System prompts by mode:
- **Quick**: "You are a writing assistant. Give brief, direct answers (3 sentences max)."
- **Planning**: "You are a story planning assistant. Help with plot structure, character arcs, and world-building. Provide structured suggestions."
- **Brainstorm**: "You are a creative brainstorming partner. Generate ideas freely, suggest alternatives, and explore possibilities."

All modes include: "The WRITER writes. You ASSIST and ADVISE. Never generate prose unless explicitly asked."

**Commit:**
```bash
git commit -m "feat(writing): add AIAgentTab with context-aware manuscript chat"
```

---

## Task 8: Frontend — Wire Analysis + Agent into WritingPlayground

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundInspectorPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
- Modify: `apps/packages/ui/src/store/writing-playground.tsx`

**Step 1: Add analysis modal state to Zustand store**

```typescript
analysisModalOpen: "pulse" | "plot" | "timeline" | "web" | null
setAnalysisModalOpen: (modal: "pulse" | "plot" | "timeline" | "web" | null) => void
```

**Step 2: Add "agent" tab to InspectorPanel**

Add to `TAB_DEFINITIONS`, props, and `panelMap`.

**Step 3: Wire in index.tsx**

- Import `AIAgentTab` and `WritingAnalysisModalHost`
- Create `agentTabContent = <AIAgentTab isOnline={isOnline} />`
- Pass to inspector panel
- Render `WritingAnalysisModalHost` (lazy-loaded) at the bottom of the component
- Add analysis buttons to toolbar (Activity, GitBranch, Clock, Share2 icons) that open modals

**Commit:**
```bash
git commit -m "feat(writing): wire analysis modals and AI agent tab into WritingPlayground"
```

---

## Verification Checklist

1. **DB migration**: Version 43 with `manuscript_ai_analyses` table
2. **Analysis CRUD**: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py -v`
3. **Analysis service**: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py -v`
4. **Analysis endpoints**: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py -v`
5. **All prior tests**: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_*.py -v` (all pass)
6. **Frontend build**: Builds without errors
7. **Inspector**: 7 tabs (sampling, context, setup, analysis, characters, research, agent)
8. **Analysis modals**: Story Pulse, Plot Tracker, Event Line, Connection Web open from toolbar
9. **AI Agent**: Chat interface sends messages with manuscript context

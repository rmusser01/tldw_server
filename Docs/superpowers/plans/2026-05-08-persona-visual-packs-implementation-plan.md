# Persona Visual Packs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build V1 animated persona visual packs: persona-owned sprite/frame assets, validated manifests, Buddy shell runtime rendering, a pack editor, Jobs-backed generated candidates, and internal `persona_visuals` MCP tools.

**Architecture:** Keep Persona Buddy as the identity facet and Persona Live as the runtime source of session state. Add a persona-scoped visual-pack service that owns metadata, upload validation, manifest validation, draft/active transitions, storage, Jobs review, and MCP durable actions; the frontend consumes that service through shared UI types and upgrades the existing Buddy shell with a sprite/frame renderer and visual-state resolver.

**Tech Stack:** FastAPI, Pydantic, ChaChaNotes `CharactersRAGDB`/`PersonaStateStore`, per-user filesystem storage through `DatabasePaths`, core Jobs, existing `Image_Generation` adapter registry, Unified MCP `BaseModule`, React 18, Vitest, pytest, Playwright.

---

## Source Inputs

- Spec: `Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md`
- Planning task: `TASK-126`
- Design tracking task: `TASK-125`
- Existing persona Buddy specs:
  - `Docs/superpowers/specs/2026-03-31-persona-buddy-facet-design.md`
  - `Docs/superpowers/specs/2026-03-31-persona-buddy-track-b-floating-shell-design.md`
- Existing wake/live spec:
  - `Docs/superpowers/specs/2026-04-30-persona-wake-word-support-design.md`

## Scope Boundaries

Included in this V1 plan:

- Persona-owned visual packs and assets.
- Versioned `sprite_frames` manifest contract.
- User-owned asset storage under the per-user database directory.
- Mutable draft packs and explicit activation.
- Existing Buddy shell as the animated render surface.
- Pack editor for upload, preview, required-state mapping, frame timing, loop mode, alignment, fallback chains, generated-candidate review, and activation.
- Explicit deactivate/revert to derived Buddy rendering.
- Jobs-backed generation candidate creation using the existing image-generation adapter registry.
- Internal `persona_visuals` MCP module for capabilities, bounded transient states, and durable draft/review changes.
- Tests for backend validation, API, frontend resolver/renderer/editor, Jobs, and MCP.

Not included in V1:

- Live2D renderer.
- Arbitrary user-supplied scripts or SVG animation runtimes.
- Shared libraries, marketplaces, or cross-persona sharing.
- Automatic active-pack replacement from MCP.
- Full crop/onion-skin/state-machine visual authoring.
- Real external generation in tests.

## Implementation Decisions Locked For V1

The spec review called out four planning decisions that must be resolved before implementation:

- **Storage adapter:** add `DatabasePaths.PERSONA_VISUALS_SUBDIR = "persona_visuals"` and `DatabasePaths.get_user_persona_visuals_dir(user_id)`. Store metadata in ChaChaNotes DB; store bytes under the user directory as `persona_visuals/<safe_persona_id>/<pack_id>/<asset_id>.<ext>`. The service is the only layer that resolves these paths.
- **Revision model:** use mutable draft packs in V1. Editing an active pack creates a new draft with `parent_pack_id` and `revision_number`; activation marks the draft `active` and archives the previous active pack for that persona.
- **Upload limits:** start with conservative constants in the service: allowed MIME types `image/png`, `image/jpeg`, `image/webp`, `image/gif`; max upload file size `10_485_760` bytes; max image dimension `4096`; max frames per animation `240`; max total persona visual bytes `104_857_600`. Keep these constants overridable later, but do not add a config surface in V1.
- **First generation-provider path:** use `tldw_Server_API.app.core.Image_Generation.adapter_registry.get_registry()` in a Jobs worker. Tests inject a fake registry and fake adapter; runtime returns a clear unavailable failure when no image-generation backend is configured.

## Planned File Structure

Create:

- `tldw_Server_API/app/core/Persona/visuals.py`
  - Manifest models, validation helpers, visual-state constants, upload limits, storage-key normalization, and service-facing pure functions.
- `tldw_Server_API/app/core/Persona/visual_service.py`
  - `PersonaVisualService` for pack/asset CRUD, upload validation, manifest validation, draft/active transitions, generated candidate review, and storage access.
- `tldw_Server_API/app/core/Persona/visual_jobs.py`
  - Job type constants, payload builders, idempotency keys, and enqueue helpers.
- `tldw_Server_API/app/core/Persona/visual_jobs_worker.py`
  - Jobs worker handler that uses `Image_Generation` adapters to create generated candidates.
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py`
  - Internal MCP module for visual pack capabilities, runtime state triggers, draft changes, and generation requests.
- `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py`
- `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`
- `tldw_Server_API/tests/Persona/test_persona_visual_jobs.py`
- `tldw_Server_API/tests/Services/test_persona_visual_jobs_worker_startup.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py`
- `apps/packages/ui/src/types/persona-visuals.ts`
- `apps/packages/ui/src/services/persona-visuals.ts`
- `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualState.ts`
- `apps/packages/ui/src/components/Common/PersonaBuddy/SpriteFrameRenderer.tsx`
- `apps/packages/ui/src/store/persona-visual-runtime.ts`
  - Session-scoped visual override store for MCP-triggered runtime states.
- `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
- `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts`
- `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx`

Modify:

- `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
  - Add persona visual asset storage directory helper.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Add migration SQL for visual pack, asset, and generated candidate tables; delegate new store methods.
- `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`
  - Add row mappers and CRUD methods for visual packs/assets/candidates.
- `tldw_Server_API/app/api/v1/schemas/persona.py`
  - Add visual pack request/response/manifest schemas.
- `tldw_Server_API/app/api/v1/endpoints/persona.py`
  - Add persona-scoped visual pack endpoints, authenticated asset serving, generated candidate review endpoints, and service wiring.
- `tldw_Server_API/app/services/startup_optional_workers.py`
  - Register persona visual generation worker behind an explicit optional-worker env flag.
- `tldw_Server_API/Config_Files/mcp_modules.yaml`
  - Add disabled `persona_visuals` module entry for controlled rollout.
- `apps/packages/ui/src/types/persona-buddy.ts`
  - Extend Buddy summary types with optional active visual-pack summary.
- `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx`
  - Load active visual pack and feed runtime state into the renderer.
- `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellDock.tsx`
  - Render animated sprite content while preserving fallback text behavior.
- `apps/packages/ui/src/components/Common/PersonaBuddy/index.ts`
  - Export new visual state/renderer pieces as needed.
- `apps/packages/ui/src/components/PersonaGarden/AssistantVoiceCard.tsx`
  - Show compact live visual-state feedback for debugging overrides, triggers, and fallback recovery.
- `apps/packages/ui/src/routes/sidepanel-persona.tsx`
  - Add lazy visual pack editor tab and pass active persona id.
- `apps/packages/ui/src/utils/persona-garden-route.ts`
  - Add tab key for visuals if the route utility enumerates allowed tabs.
- `apps/packages/ui/src/hooks/usePersonaLiveVoiceController.tsx`
  - Expose enough status for visual state resolution, if current return object does not already expose it.
- `apps/packages/ui/src/routes/sidepanel-persona.tsx`
  - Handle `visual_state_override` WebSocket payloads and publish session-scoped visual overrides.
- `apps/packages/ui/src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx`
  - Extend visual-state fixture coverage if live panel status needs to expose visual state.
- `apps/tldw-frontend/e2e/workflows/persona-live.spec.ts`
  - Add fixture coverage for visual state changes without real microphone/TTS.
- `backlog/tasks/task-126 - Write-persona-visual-packs-implementation-plan.md`
  - Track plan/review/verification.

## Data Model

Add ChaChaNotes tables in the next schema migration:

```sql
CREATE TABLE IF NOT EXISTS persona_visual_packs (
  id TEXT PRIMARY KEY,
  persona_id TEXT NOT NULL REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  title TEXT NOT NULL,
  renderer_type TEXT NOT NULL DEFAULT 'sprite_frames',
  status TEXT NOT NULL DEFAULT 'draft'
    CHECK(status IN ('draft','review','active','archived','failed')),
  manifest_version INTEGER NOT NULL DEFAULT 1,
  manifest_json TEXT NOT NULL DEFAULT '{}',
  parent_pack_id TEXT REFERENCES persona_visual_packs(id) ON DELETE SET NULL,
  revision_number INTEGER NOT NULL DEFAULT 1,
  provenance TEXT NOT NULL DEFAULT 'uploaded'
    CHECK(provenance IN ('uploaded','generated','imported','mixed')),
  active_at TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_persona_visual_packs_one_active
  ON persona_visual_packs(user_id, persona_id)
  WHERE status = 'active' AND deleted = 0;

CREATE INDEX IF NOT EXISTS idx_persona_visual_packs_persona
  ON persona_visual_packs(user_id, persona_id, deleted, status);

CREATE TABLE IF NOT EXISTS persona_visual_assets (
  id TEXT PRIMARY KEY,
  pack_id TEXT NOT NULL REFERENCES persona_visual_packs(id) ON DELETE CASCADE,
  persona_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  asset_role TEXT NOT NULL
    CHECK(asset_role IN ('frame','still_pose','sprite_sheet','preview','generated_candidate')),
  storage_key TEXT NOT NULL,
  original_filename TEXT,
  mime_type TEXT NOT NULL,
  byte_size INTEGER NOT NULL,
  checksum_sha256 TEXT NOT NULL,
  width INTEGER,
  height INTEGER,
  validation_status TEXT NOT NULL DEFAULT 'valid'
    CHECK(validation_status IN ('valid','invalid')),
  validation_error TEXT,
  provenance TEXT NOT NULL DEFAULT 'uploaded',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_persona_visual_assets_pack
  ON persona_visual_assets(user_id, pack_id, deleted);

CREATE TABLE IF NOT EXISTS persona_visual_candidates (
  id TEXT PRIMARY KEY,
  pack_id TEXT NOT NULL REFERENCES persona_visual_packs(id) ON DELETE CASCADE,
  persona_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  job_id TEXT,
  status TEXT NOT NULL DEFAULT 'review'
    CHECK(status IN ('review','accepted','rejected','failed')),
  proposed_manifest_patch_json TEXT NOT NULL DEFAULT '{}',
  generated_asset_ids_json TEXT NOT NULL DEFAULT '[]',
  prompt TEXT,
  failure_reason TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1
);
```

Postgres migration text should mirror the schema with `BOOLEAN` and `TIMESTAMP` defaults, following existing `*_POSTGRES` migration style in `ChaChaNotes_DB.py`.

## Manifest Contract

Use this V1 manifest shape. Backend validation is authoritative; frontend types mirror it.

```python
VISUAL_STATE_IDS = {
    "idle",
    "wake_armed",
    "listening",
    "thinking",
    "speaking",
    "tool_running",
    "approval_needed",
    "error",
    "offline",
}

REQUIRED_VISUAL_STATES = {"idle", "listening", "thinking", "speaking", "error"}
```

```json
{
  "manifest_version": 1,
  "renderer_type": "sprite_frames",
  "states": {
    "idle": {"animation_id": "idle"},
    "listening": {"animation_id": "listening"},
    "thinking": {"animation_id": "thinking"},
    "speaking": {"animation_id": "speaking"},
    "error": {"animation_id": "error"}
  },
  "animations": {
    "idle": {
      "frames": [
        {
          "asset_id": "asset-id",
          "duration_ms": 1000
        }
      ],
      "frame_rate": 1,
      "loop": true,
      "alignment": {"x": 0.5, "y": 1.0},
      "preview_frame": 0
    },
    "sprite-sheet-wave": {
      "frames": [
        {
          "asset_id": "sprite-sheet-id",
          "region": {"x": 0, "y": 0, "width": 256, "height": 256},
          "duration_ms": 120
        },
        {
          "asset_id": "sprite-sheet-id",
          "region": {"x": 256, "y": 0, "width": 256, "height": 256},
          "duration_ms": 120
        }
      ],
      "frame_rate": 8,
      "loop": true,
      "alignment": {"x": 0.5, "y": 1.0},
      "preview_frame": 0,
      "preview_asset_id": "sprite-sheet-id"
    }
  },
  "fallbacks": {
    "wake_armed": ["idle"],
    "tool_running": ["thinking", "idle"],
    "approval_needed": ["thinking", "idle"],
    "offline": ["idle"]
  },
  "authored_triggers": [
    {
      "id": "notes-search-pulse",
      "source": "tool_category",
      "match": "notes",
      "state": "tool_running",
      "duration_ms": 2500,
      "priority": 20
    }
  ]
}
```

Validation rules:

- `manifest_version` must be `1`.
- `renderer_type` must be `sprite_frames`.
- Every referenced `asset_id` must belong to the pack.
- `frame_rate` must be between `1` and `60`.
- An animation must define either `frames` or `asset_ids`. `asset_ids` is accepted as a shorthand and normalized to an ordered `frames` list.
- `frames` order is authoritative. Reordering frames in the editor must change the persisted frame order without relying on filename or upload order.
- An animation may reference at most `240` frames after shorthand normalization.
- `frame.duration_ms`, when present, must be between `16` and `30_000`.
- `frame.region`, when present, defines a sprite-sheet crop rectangle with non-negative `x`/`y` and positive `width`/`height`.
- If source asset dimensions are known, `frame.region` must fit inside the referenced asset's width and height.
- `preview_frame`, when present, must point to a valid frame index.
- `preview_asset_id`, when present, must reference an asset used by the animation.
- `alignment.x` and `alignment.y` must be between `0` and `1`.
- Fallback chains must not contain cycles.
- Activation requires all `REQUIRED_VISUAL_STATES` to resolve to a valid animation or fallback.
- Optional states `wake_armed`, `tool_running`, `approval_needed`, and `offline` may rely on fallback chains in V1; activation must verify each optional state resolves either to an explicit animation or to `idle`.
- `authored_triggers` is a list of rules with `id`, `source`, `match`, `state`, `duration_ms`, and `priority`.
- V1 trigger sources are `live_state`, `tool_category`, and `mcp_runtime`.
- `state` must be one of `VISUAL_STATE_IDS`.
- `duration_ms` must be between `100` and `30_000`.
- `priority` must be between `0` and `100`.
- Unknown trigger sources or invalid target states must fail manifest validation.

## Task 1: Plan And Clean-Worktree Setup

**Files:**

- Create: `Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md`
- Modify through MCP: `TASK-126`

- [ ] **Step 1: Confirm current repo conflicts before implementation**

Run:

```bash
git status --short
```

Expected: if any `U` conflict entries remain, do not implement code in this checkout.

- [ ] **Step 2: Create or switch to a clean implementation worktree**

Use the `superpowers:using-git-worktrees` skill before implementation. Create a worktree from the intended base branch, then copy or re-apply the approved spec/plan files if they are still uncommitted in the main checkout.

Expected: implementation starts from a checkout where `git status --short` has no unrelated `U` entries.

- [ ] **Step 3: Record plan path in TASK-126**

Use Backlog MCP `task_edit` with:

```text
Plan: Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
Scope: V1 staged implementation for sprite/frame visual packs, Buddy shell runtime, pack editor, Jobs-backed generated candidates, and internal persona_visuals MCP tools.
```

- [ ] **Step 4: Wait for explicit execution approval**

Expected: do not create production-code files until the user approves executing this plan.

## Task 2: Core Manifest Validation

**Files:**

- Create: `tldw_Server_API/app/core/Persona/visuals.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`

- [ ] **Step 1: Write failing manifest validation tests**

Create tests:

```python
import pytest

from tldw_Server_API.app.core.Persona.visuals import (
    PersonaVisualManifestError,
    REQUIRED_VISUAL_STATES,
    validate_visual_manifest,
)


def test_valid_manifest_resolves_required_states() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {
            "idle": {"animation_id": "idle"},
            "listening": {"animation_id": "listen"},
            "thinking": {"animation_id": "think"},
            "speaking": {"animation_id": "speak"},
            "error": {"animation_id": "error"},
        },
        "animations": {
            "idle": {"asset_ids": ["asset-idle"], "frame_rate": 1, "loop": True},
            "listen": {"asset_ids": ["asset-listen"], "frame_rate": 8, "loop": True},
            "think": {"asset_ids": ["asset-think"], "frame_rate": 8, "loop": True},
            "speak": {"asset_ids": ["asset-speak"], "frame_rate": 12, "loop": True},
            "error": {"asset_ids": ["asset-error"], "frame_rate": 1, "loop": False},
        },
        "fallbacks": {"tool_running": ["thinking", "idle"]},
    }

    result = validate_visual_manifest(
        manifest,
        available_asset_ids={
            "asset-idle",
            "asset-listen",
            "asset-think",
            "asset-speak",
            "asset-error",
        },
        require_activatable=True,
    )

    assert set(result.resolved_required_states) == REQUIRED_VISUAL_STATES


def test_activation_rejects_missing_required_state() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {"idle": {"asset_ids": ["asset-idle"], "frame_rate": 1}},
        "fallbacks": {},
    }

    with pytest.raises(PersonaVisualManifestError, match="listening"):
        validate_visual_manifest(
            manifest,
            available_asset_ids={"asset-idle"},
            require_activatable=True,
        )


def test_rejects_fallback_cycles() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {"idle": {"asset_ids": ["asset-idle"], "frame_rate": 1}},
        "fallbacks": {"thinking": ["tool_running"], "tool_running": ["thinking"]},
    }

    with pytest.raises(PersonaVisualManifestError, match="cycle"):
        validate_visual_manifest(
            manifest,
            available_asset_ids={"asset-idle"},
            require_activatable=False,
        )


def test_accepts_sprite_sheet_regions_and_preview_frame() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {
            "idle": {
                "frames": [
                    {
                        "asset_id": "sheet-asset",
                        "region": {"x": 0, "y": 0, "width": 128, "height": 128},
                        "duration_ms": 120,
                    },
                    {
                        "asset_id": "sheet-asset",
                        "region": {"x": 128, "y": 0, "width": 128, "height": 128},
                        "duration_ms": 120,
                    },
                ],
                "frame_rate": 8,
                "preview_frame": 1,
            }
        },
    }

    result = validate_visual_manifest(
        manifest,
        available_asset_ids={"sheet-asset"},
        available_asset_dimensions={"sheet-asset": (256, 128)},
        require_activatable=False,
    )

    assert result.manifest["animations"]["idle"]["frames"][1]["region"]["x"] == 128


def test_rejects_preview_frame_out_of_range() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {
            "idle": {
                "asset_ids": ["asset-idle"],
                "frame_rate": 1,
                "preview_frame": 2,
            }
        },
    }

    with pytest.raises(PersonaVisualManifestError, match="preview_frame"):
        validate_visual_manifest(
            manifest,
            available_asset_ids={"asset-idle"},
            require_activatable=False,
        )
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py -v
```

Expected: FAIL because `tldw_Server_API.app.core.Persona.visuals` does not exist.

- [ ] **Step 3: Implement the manifest core**

Create `visuals.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

VISUAL_STATE_IDS = {
    "idle",
    "wake_armed",
    "listening",
    "thinking",
    "speaking",
    "tool_running",
    "approval_needed",
    "error",
    "offline",
}
REQUIRED_VISUAL_STATES = {"idle", "listening", "thinking", "speaking", "error"}
SUPPORTED_RENDERER_TYPES = {"sprite_frames"}
MAX_FRAMES_PER_ANIMATION = 240


class PersonaVisualManifestError(ValueError):
    """Raised when a visual pack manifest is invalid."""


@dataclass(frozen=True)
class PersonaVisualManifestValidation:
    manifest: dict[str, Any]
    resolved_required_states: dict[str, str]


def validate_visual_manifest(
    manifest: dict[str, Any],
    *,
    available_asset_ids: set[str],
    available_asset_dimensions: dict[str, tuple[int, int]] | None = None,
    require_activatable: bool,
) -> PersonaVisualManifestValidation:
    normalized = _normalize_manifest_shape(manifest)
    animations = normalized["animations"]
    for animation_id, animation in animations.items():
        _validate_animation(animation_id, animation, available_asset_ids=available_asset_ids)
    _detect_fallback_cycles(normalized.get("fallbacks", {}))
    resolved_required = {
        state: _resolve_state(state, normalized)
        for state in REQUIRED_VISUAL_STATES
    }
    if require_activatable:
        missing = sorted(state for state, animation_id in resolved_required.items() if not animation_id)
        if missing:
            raise PersonaVisualManifestError(
                "Required visual states do not resolve: " + ", ".join(missing)
            )
    return PersonaVisualManifestValidation(
        manifest=normalized,
        resolved_required_states={k: v for k, v in resolved_required.items() if v},
    )
```

Implementation details:

- Keep this module pure: no DB, no filesystem, no FastAPI.
- Return a normalized manifest copy with default `fallbacks` and `authored_triggers`.
- Normalize `asset_ids` shorthand to `frames` before validation so backend, renderer, and editor all consume one ordered frame contract.
- Validate sprite-sheet `region` rectangles against known asset dimensions when width/height metadata is available.
- Raise `PersonaVisualManifestError` with field-specific messages.
- Add helper functions for `_validate_animation`, `_detect_fallback_cycles`, and `_resolve_state`.

- [x] **Step 4: Run core tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py -v
```

Expected: PASS.

- [x] **Step 5: Commit**

Run:

```bash
git add tldw_Server_API/app/core/Persona/visuals.py tldw_Server_API/tests/Persona/test_persona_visuals_core.py
git commit -m "feat: add persona visual manifest validation"
```

## Task 3: Persistence And Storage Helpers

**Files:**

- Modify: `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py`

- [x] **Step 1: Write failing DB migration and CRUD tests**

Cover:

```python
def test_migration_creates_persona_visual_tables(db_path: Path) -> None:
    seeded = CharactersRAGDB(db_path, "seed-client")
    seeded.close_connection()
    # Force schema to the previous version and drop visual tables, then reopen.
    # Assert all three persona visual tables and indexes exist after migration.


def test_create_and_list_visual_pack_for_persona(db_instance: CharactersRAGDB) -> None:
    persona_id = db_instance.create_persona_profile({"user_id": "user-1", "name": "Visual Persona"})
    pack = db_instance.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        title="Default Sprite Pack",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )
    assert pack["persona_id"] == persona_id
    assert db_instance.list_persona_visual_packs(persona_id=persona_id, user_id="user-1")


def test_only_one_active_pack_per_persona(db_instance: CharactersRAGDB) -> None:
    # Create two draft packs for one persona, activate both in sequence, and assert
    # only the second remains active while the first is archived.


def test_assets_are_scoped_to_pack_persona_and_user(db_instance: CharactersRAGDB) -> None:
    # Create two users/personas and assert listing assets for user B never returns
    # assets created for user A.


def test_candidate_accept_reject_round_trip(db_instance: CharactersRAGDB) -> None:
    # Create a candidate, mark it accepted, create another candidate, mark it
    # rejected, and assert both state transitions persist.
```

Use the existing style in `tldw_Server_API/tests/ChaChaNotesDB/test_persona_buddy_db.py`.

- [x] **Step 2: Run DB tests to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py -v
```

Expected: FAIL because storage helper and DB methods do not exist.

- [x] **Step 3: Add user storage directory helper**

In `db_path_utils.py`, add:

```python
PERSONA_VISUALS_SUBDIR = "persona_visuals"
```

and:

```python
@staticmethod
def get_user_persona_visuals_dir(user_id: Optional[UserId]) -> Path:
    """Get the path to the user's persona visual assets directory."""
    user_dir = DatabasePaths.get_user_base_directory(user_id)
    visuals_dir = user_dir / DatabasePaths.PERSONA_VISUALS_SUBDIR
    _ensure_dir(visuals_dir, label="persona visuals")
    return visuals_dir
```

- [x] **Step 4: Add schema migration**

Add a new schema migration after the current latest version in `ChaChaNotes_DB.py`:

- SQLite migration for `persona_visual_packs`, `persona_visual_assets`, `persona_visual_candidates`.
- Postgres migration equivalent.
- Bump `_CURRENT_SCHEMA_VERSION`.
- Add migration to the ordered migration path used by `_migrate_schema`.

Expected: a DB created from scratch includes the new tables; a DB forced to the previous version migrates forward.

- [x] **Step 5: Add PersonaStateStore methods**

In `persona_state_store.py`, add row mappers and these methods:

- `create_persona_visual_pack`: insert a draft pack after verifying the persona belongs to `user_id`.
- `get_persona_visual_pack`: return one pack scoped by `pack_id`, `persona_id`, and `user_id`.
- `get_active_persona_visual_pack`: return the active pack plus asset metadata for rendering, or `None`.
- `list_persona_visual_packs`: return non-deleted packs for a persona/user ordered active first then recently updated.
- `update_persona_visual_pack_manifest`: persist normalized manifest JSON and increment version with optimistic locking.
- `activate_persona_visual_pack`: archive the current active pack for the persona and mark this pack active inside one transaction.
- `deactivate_persona_visual_pack`: archive the current active pack so the persona falls back to derived Buddy rendering.
- `create_persona_visual_asset`: insert asset metadata after verifying pack/persona/user ownership.
- `list_persona_visual_assets`: return non-deleted asset metadata for one pack.
- `create_persona_visual_candidate`: create a generated-candidate review record for a draft pack.
- `update_persona_visual_candidate_status`: persist accepted/rejected/failed candidate status and failure reason.

- [x] **Step 6: Delegate methods through CharactersRAGDB**

In `ChaChaNotes_DB.py`, add the new method names to the `_persona_state_store_method` tuple.

- [x] **Step 7: Run DB tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py -v
```

Expected: PASS.

- [x] **Step 8: Commit**

Run:

```bash
git add tldw_Server_API/app/core/DB_Management/db_path_utils.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py
git commit -m "feat: persist persona visual packs"
```

## Task 4: PersonaVisualService And Upload Validation

**Files:**

- Create: `tldw_Server_API/app/core/Persona/visual_service.py`
- Modify: `tldw_Server_API/app/core/Persona/visuals.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`

- [ ] **Step 1: Write failing service tests**

Add tests for:

```python
def test_service_rejects_unsupported_mime_type(tmp_path: Path) -> None:
    # Upload text/plain bytes and assert PersonaVisualServiceError("invalid_upload").


def test_service_rejects_oversized_upload(tmp_path: Path) -> None:
    # Upload MAX_VISUAL_UPLOAD_BYTES + 1 bytes and assert no file is written.


def test_service_writes_asset_under_user_visuals_dir(tmp_path: Path) -> None:
    # Monkeypatch DatabasePaths.get_user_persona_visuals_dir to tmp_path and
    # assert the stored path remains below tmp_path.


def test_service_activation_rejects_invalid_manifest(tmp_path: Path) -> None:
    # Create a draft whose manifest only has idle and assert activate_pack raises.


def test_service_activation_archives_previous_active_pack(tmp_path: Path) -> None:
    # Activate pack A, then pack B, and assert A is archived and B is active.


def test_service_deactivate_reverts_to_derived_buddy(tmp_path: Path) -> None:
    # Activate a pack, deactivate it, and assert no active pack remains for the
    # persona while the pack row is archived rather than deleted.
```

Use `tmp_path` and monkeypatch `DatabasePaths.get_user_persona_visuals_dir` if needed.

- [ ] **Step 2: Run failing tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py -v
```

Expected: FAIL because `PersonaVisualService` does not exist.

- [ ] **Step 3: Implement service constants and exceptions**

In `visual_service.py`:

```python
ALLOWED_VISUAL_MIME_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
MAX_VISUAL_UPLOAD_BYTES = 10_485_760
MAX_VISUAL_IMAGE_DIMENSION = 4096
MAX_PERSONA_VISUAL_BYTES = 104_857_600


class PersonaVisualServiceError(RuntimeError):
    def __init__(self, code: str, *, detail: str | None = None) -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail or code
```

- [ ] **Step 4: Implement service methods**

Add:

```python
class PersonaVisualService:
    def __init__(self, *, db: CharactersRAGDB, user_id: str) -> None:
        self.db = db
        self.user_id = str(user_id)

    # Implement list_packs, get_pack_detail, create_draft_pack,
    # add_uploaded_asset, read_asset_content, validate_pack_manifest,
    # update_manifest, activate_pack, deactivate_active_pack, list_candidates,
    # get_candidate, accept_candidate, and reject_candidate. Each method must
    # call the DB store methods from Task 3 and raise PersonaVisualServiceError
    # with stable code values.
```

Implementation notes:

- Use `imghdr` only as supporting evidence if useful; prefer Pillow if already available, but do not add a new dependency for V1. If dimension parsing is unavailable for a format, store `None` and enforce byte/MIME validation.
- Generate storage keys from service-owned UUIDs, not original filenames.
- Sanitize persona and pack path components; never trust user-supplied filename as a path.
- Ensure `add_uploaded_asset` verifies the pack belongs to `persona_id` and `user_id` before writing bytes.

- [ ] **Step 5: Run service tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add tldw_Server_API/app/core/Persona/visuals.py tldw_Server_API/app/core/Persona/visual_service.py tldw_Server_API/tests/Persona/test_persona_visuals_core.py
git commit -m "feat: validate persona visual assets"
```

## Task 5: Persona Visual Pack API

**Files:**

- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`

- [ ] **Step 1: Write failing API tests**

Use `TestClient` and dependency overrides like `test_persona_buddy_api.py`.

Cover:

```python
def test_create_list_and_activate_visual_pack(persona_db: CharactersRAGDB) -> None:
    # Create persona, create draft visual pack, update manifest, activate, and
    # assert list endpoint returns exactly one active pack.


def test_upload_rejects_unsupported_mime_type(persona_db: CharactersRAGDB) -> None:
    # Upload multipart text/plain content and assert HTTP 400 with invalid_upload.


def test_activation_rejects_manifest_without_required_states(persona_db: CharactersRAGDB) -> None:
    # Attempt activation with idle-only manifest and assert HTTP 400.


def test_other_user_cannot_access_pack(persona_db: CharactersRAGDB) -> None:
    # Create pack as user 1, request it as user 2, and assert 404 or 403.


def test_accept_generated_candidate_updates_draft_not_active_pack(persona_db: CharactersRAGDB) -> None:
    # Accept a candidate against a draft and assert active pack id remains unchanged.


def test_deactivate_visual_pack_reverts_to_derived_buddy(persona_db: CharactersRAGDB) -> None:
    # Activate a pack, call deactivate, and assert active-pack lookup returns none.
```

- [ ] **Step 2: Run failing API tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -v
```

Expected: FAIL because API schemas/endpoints do not exist.

- [ ] **Step 3: Add Pydantic schemas**

In `schemas/persona.py`, add:

```python
PersonaVisualPackStatus = Literal["draft", "review", "active", "archived", "failed"]
PersonaVisualRendererType = Literal["sprite_frames"]
PersonaVisualAssetRole = Literal["frame", "still_pose", "sprite_sheet", "preview", "generated_candidate"]


class PersonaVisualPackCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    manifest: dict[str, Any] = Field(default_factory=dict)


class PersonaVisualManifestUpdate(BaseModel):
    manifest: dict[str, Any] = Field(default_factory=dict)
    expected_version: int | None = Field(default=None, ge=1)


class PersonaVisualPackResponse(BaseModel):
    id: str
    persona_id: str
    title: str
    renderer_type: PersonaVisualRendererType
    status: PersonaVisualPackStatus
    manifest_version: int = 1
    manifest: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    created_at: str
    last_modified: str


class PersonaVisualAssetResponse(BaseModel):
    id: str
    pack_id: str
    persona_id: str
    asset_role: PersonaVisualAssetRole
    url: str
    mime_type: str
    byte_size: int
    checksum_sha256: str
    width: int | None = None
    height: int | None = None
    validation_status: str
    validation_error: str | None = None


class PersonaVisualPackDetailResponse(PersonaVisualPackResponse):
    assets: list[PersonaVisualAssetResponse] = Field(default_factory=list)
    assets_by_id: dict[str, PersonaVisualAssetResponse] = Field(default_factory=dict)


class PersonaVisualPackListResponse(BaseModel):
    packs: list[PersonaVisualPackResponse] = Field(default_factory=list)
    active_pack: PersonaVisualPackDetailResponse | None = None


class PersonaVisualCandidateResponse(BaseModel):
    id: str
    pack_id: str
    persona_id: str
    status: Literal["review", "accepted", "rejected", "failed"]
    job_id: str | None = None
    proposed_manifest_patch: dict[str, Any] = Field(default_factory=dict)
    generated_assets: list[PersonaVisualAssetResponse] = Field(default_factory=list)
    prompt: str | None = None
    failure_reason: str | None = None
    created_at: str
    last_modified: str


class PersonaVisualCandidateListResponse(BaseModel):
    candidates: list[PersonaVisualCandidateResponse] = Field(default_factory=list)


class PersonaVisualGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    target_state: str | None = Field(default=None, max_length=80)
    backend: str | None = Field(default=None, max_length=80)
```

- [ ] **Step 4: Add endpoint helpers**

In `endpoints/persona.py`, import `UploadFile`, `File`, and schemas. Add helper:

```python
def _persona_visual_service(
    *, db: CharactersRAGDB, current_user: User
) -> PersonaVisualService:
    return PersonaVisualService(db=db, user_id=_require_current_user_id(current_user))
```

Map `PersonaVisualServiceError.code` to HTTP status:

- `not_found` -> 404
- `forbidden` -> 403
- `invalid_manifest`, `invalid_upload`, `quota_exceeded` -> 400
- default -> 500

Add asset URL projection:

- `PersonaVisualAssetResponse.url` must be an authenticated API URL, not a filesystem path.
- The URL shape is `/api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/assets/{asset_id}/content`.
- The endpoint must re-check user, persona, pack, and asset ownership before streaming bytes.
- The response should set the stored MIME type and a long-lived private cache header such as `Cache-Control: private, max-age=3600`.
- Frontend renderers and preview UI must only use this projected `url`.
- `PersonaVisualPackDetailResponse.assets_by_id` is the exact asset URL map consumed by `SpriteFrameRenderer`.
- `GET /visual-packs` should include `active_pack` with `assets_by_id` so the Buddy shell can render without inventing a second contract.
- `GET /visual-packs/{pack_id}` should return `PersonaVisualPackDetailResponse` for editor/detail views.

- [ ] **Step 5: Add endpoints**

Add routes under existing persona router:

```python
@router.get("/profiles/{persona_id}/visual-packs", response_model=PersonaVisualPackListResponse, tags=["persona"])
async def list_persona_visual_packs(persona_id: str, current_user: User = Depends(get_request_user), db: CharactersRAGDB = Depends(get_chacha_db_for_user)) -> PersonaVisualPackListResponse:
    service = _persona_visual_service(db=db, current_user=current_user)
    return PersonaVisualPackListResponse(packs=service.list_packs(persona_id=persona_id))


@router.post("/profiles/{persona_id}/visual-packs", response_model=PersonaVisualPackResponse, tags=["persona"])
async def create_persona_visual_pack(persona_id: str, payload: PersonaVisualPackCreate = Body(), current_user: User = Depends(get_request_user), db: CharactersRAGDB = Depends(get_chacha_db_for_user)) -> PersonaVisualPackResponse:
    service = _persona_visual_service(db=db, current_user=current_user)
    return PersonaVisualPackResponse.model_validate(service.create_draft_pack(persona_id=persona_id, title=payload.title, manifest=payload.manifest))
```

Add matching `GET`, asset upload, manifest update, validate, and activate endpoints with the same service/error-mapping pattern.

Add deactivate endpoint:

```python
@router.post("/profiles/{persona_id}/visual-packs/deactivate", response_model=PersonaVisualPackListResponse, tags=["persona"])
async def deactivate_persona_visual_pack(persona_id: str, current_user: User = Depends(get_request_user), db: CharactersRAGDB = Depends(get_chacha_db_for_user)) -> PersonaVisualPackListResponse:
    service = _persona_visual_service(db=db, current_user=current_user)
    service.deactivate_active_pack(persona_id=persona_id)
    return PersonaVisualPackListResponse(packs=service.list_packs(persona_id=persona_id), active_pack=None)
```

Add asset content and candidate review endpoints:

```python
@router.get("/profiles/{persona_id}/visual-packs/{pack_id}/assets/{asset_id}/content", tags=["persona"])
async def get_persona_visual_asset_content(persona_id: str, pack_id: str, asset_id: str, current_user: User = Depends(get_request_user), db: CharactersRAGDB = Depends(get_chacha_db_for_user)) -> Response:
    service = _persona_visual_service(db=db, current_user=current_user)
    content, mime_type = service.read_asset_content(persona_id=persona_id, pack_id=pack_id, asset_id=asset_id)
    return Response(content=content, media_type=mime_type, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/profiles/{persona_id}/visual-packs/{pack_id}/generated-candidates", response_model=PersonaVisualCandidateListResponse, tags=["persona"])
async def list_persona_visual_candidates(persona_id: str, pack_id: str, current_user: User = Depends(get_request_user), db: CharactersRAGDB = Depends(get_chacha_db_for_user)) -> PersonaVisualCandidateListResponse:
    service = _persona_visual_service(db=db, current_user=current_user)
    return PersonaVisualCandidateListResponse(candidates=service.list_candidates(persona_id=persona_id, pack_id=pack_id))
```

- [ ] **Step 6: Run API tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py
git commit -m "feat: add persona visual pack API"
```

## Task 6: Frontend Types, API Client, Resolver, And Renderer

**Files:**

- Create: `apps/packages/ui/src/types/persona-visuals.ts`
- Create: `apps/packages/ui/src/services/persona-visuals.ts`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualState.ts`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/SpriteFrameRenderer.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/index.ts`
- Test: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts`
- Test: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx`

- [ ] **Step 1: Write failing resolver tests**

```ts
import { resolvePersonaVisualState } from "../personaVisualState"

it("prefers error over live voice state", () => {
  expect(resolvePersonaVisualState({ liveVoiceState: "speaking", hasError: true })).toBe("error")
})

it("maps wake armed before idle", () => {
  expect(resolvePersonaVisualState({ liveVoiceState: "idle", wakeArmed: true })).toBe("wake_armed")
})

it("maps active tool status to tool_running", () => {
  expect(resolvePersonaVisualState({ liveVoiceState: "thinking", activeToolStatus: "Running notes.search" })).toBe("tool_running")
})
```

- [ ] **Step 2: Write failing renderer tests**

Test that `SpriteFrameRenderer`:

- renders the first frame for a state
- uses `preview_frame` when rendering an initial/static preview before the animation interval advances
- respects explicit `frames` order instead of sorting by asset id, filename, or upload order
- renders sprite-sheet `region` frames by cropping the referenced asset
- applies `data-visual-state`
- falls back to `idle` when the requested state is missing
- calls `onRenderError` when no state can resolve

- [ ] **Step 3: Run failing frontend tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx
```

Expected: FAIL because files do not exist.

- [ ] **Step 4: Add frontend types**

In `persona-visuals.ts`:

```ts
export type PersonaVisualStateId =
  | "idle"
  | "wake_armed"
  | "listening"
  | "thinking"
  | "speaking"
  | "tool_running"
  | "approval_needed"
  | "error"
  | "offline"

export type PersonaVisualRendererType = "sprite_frames"

export interface PersonaVisualRegion {
  x: number
  y: number
  width: number
  height: number
}

export interface PersonaVisualFrame {
  asset_id: string
  region?: PersonaVisualRegion | null
  duration_ms?: number
}

export interface PersonaVisualAnimation {
  frames?: PersonaVisualFrame[]
  asset_ids?: string[]
  frame_rate?: number
  loop?: boolean
  alignment?: { x: number; y: number }
  preview_frame?: number
  preview_asset_id?: string
}

export interface PersonaVisualManifest {
  manifest_version: 1
  renderer_type: PersonaVisualRendererType
  states: Partial<Record<PersonaVisualStateId, { animation_id: string }>>
  animations: Record<string, PersonaVisualAnimation>
  fallbacks?: Partial<Record<PersonaVisualStateId, PersonaVisualStateId[]>>
  authored_triggers?: PersonaVisualAuthoredTrigger[]
}

export interface PersonaVisualAuthoredTrigger {
  id: string
  source: "live_state" | "tool_category" | "mcp_runtime"
  match: string
  state: PersonaVisualStateId
  duration_ms: number
  priority: number
}

export interface PersonaVisualAsset {
  id: string
  url: string
  mime_type: string
  asset_role: string
  width?: number | null
  height?: number | null
}
```

- [ ] **Step 5: Add API client helpers**

In `services/persona-visuals.ts`, wrap `tldwClient.fetchWithAuth`:

```ts
export async function listPersonaVisualPacks(personaId: string) {
  return fetchPersonaVisualJson(`/api/v1/persona/profiles/${encodeURIComponent(personaId)}/visual-packs`)
}

export async function createPersonaVisualPack(personaId: string, payload: PersonaVisualPackCreate) {
  return fetchPersonaVisualJson(`/api/v1/persona/profiles/${encodeURIComponent(personaId)}/visual-packs`, {
    method: "POST",
    body: JSON.stringify(payload)
  })
}

export async function uploadPersonaVisualAsset(personaId: string, packId: string, file: File, role: PersonaVisualAssetRole) {
  const formData = new FormData()
  formData.append("file", file)
  formData.append("asset_role", role)
  return fetchPersonaVisualJson(`/api/v1/persona/profiles/${encodeURIComponent(personaId)}/visual-packs/${encodeURIComponent(packId)}/assets`, {
    method: "POST",
    body: formData
  })
}
```

Add matching helpers for pack detail, manifest update, activation, and deactivate/revert. The list helper should return `active_pack.assets_by_id` when the backend includes it.

- [ ] **Step 6: Add visual state resolver**

Implement `resolvePersonaVisualState` with priority:

1. `hasError` or recovery mode -> `error`
2. `approvalNeeded` -> `approval_needed`
3. active unexpired MCP/runtime override -> override state
4. highest-priority matching `authored_triggers` rule -> trigger state
5. `activeToolStatus` -> `tool_running`
6. `wakeArmed` and idle -> `wake_armed`
7. live voice state mapping
8. offline -> `offline`
9. default -> `idle`

Authored trigger matching in V1:

- `live_state` matches normalized live voice state values.
- `tool_category` matches the prefix or category parsed from active tool status, such as `notes` from `Running notes.search`.
- `mcp_runtime` matches runtime override reasons emitted by `persona_visuals.trigger_state`.

- [ ] **Step 7: Add sprite renderer**

Implement a small renderer:

```tsx
export const SpriteFrameRenderer: React.FC<SpriteFrameRendererProps> = ({
  manifest,
  assets,
  state,
  fallbackLabel,
  onRenderError
}) => {
  const resolved = resolveAnimationForState(manifest, state)
  if (!resolved) {
    onRenderError?.("missing_animation")
    return <span>{fallbackLabel}</span>
  }
  const frames = normalizePersonaVisualFrames(resolved)
  const firstFrame = frames[resolved.preview_frame ?? 0] ?? frames[0]
  const firstAsset = firstFrame ? assets[firstFrame.asset_id] : null
  if (!firstAsset) {
    onRenderError?.("missing_asset")
    return <span>{fallbackLabel}</span>
  }
  return renderPersonaVisualFrame({ frame: firstFrame, asset: firstAsset, visualState: state })
}
```

Keep animation timing simple in V1: a `setInterval` advances frames when the normalized frame list has more than one frame.

Renderer details:

- Normalize each animation to an ordered `PersonaVisualFrame[]` before rendering; `asset_ids` is only a shorthand for frames without regions.
- Initial render uses `preview_frame` when provided, otherwise frame `0`.
- For a frame with no `region`, render an `<img>` with the referenced asset URL.
- For a frame with `region`, render a fixed-size element whose `backgroundImage`, `backgroundPosition`, `backgroundSize`, `width`, and `height` crop the sprite-sheet asset to the requested rectangle.
- Tests should assert the frame order is honored by advancing timers and checking the resulting asset/region, not by inspecting implementation internals.

- [ ] **Step 8: Run frontend tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
git add apps/packages/ui/src/types/persona-visuals.ts apps/packages/ui/src/services/persona-visuals.ts apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualState.ts apps/packages/ui/src/components/Common/PersonaBuddy/SpriteFrameRenderer.tsx apps/packages/ui/src/components/Common/PersonaBuddy/index.ts apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx
git commit -m "feat: add persona visual renderer primitives"
```

## Task 7: Buddy Shell Runtime Integration

**Files:**

- Modify: `apps/packages/ui/src/types/persona-buddy.ts`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellDock.tsx`
- Create: `apps/packages/ui/src/store/persona-visual-runtime.ts`
- Modify: `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- Modify: `apps/packages/ui/src/hooks/usePersonaLiveVoiceController.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/AssistantVoiceCard.tsx`
- Test: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx`
- Test: `apps/packages/ui/src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx`
- Test: `apps/packages/ui/src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx`

- [ ] **Step 1: Write failing shell integration tests**

Add coverage:

- shell requests active visual packs for the active persona
- shell renders `SpriteFrameRenderer` when a valid pack exists
- shell falls back to existing text/dormant behavior when pack load fails
- live voice `speaking` state becomes `speaking` visual state
- active tool status becomes `tool_running` visual state
- `visual_state_override` WebSocket payload sets a bounded runtime override
- expired runtime override falls back to normal live state
- live session UI exposes compact visual-state feedback with current state, source, override reason, and fallback/missing-pack status

- [ ] **Step 2: Run failing tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx
```

Expected: FAIL because shell has no visual-pack integration.

- [ ] **Step 3: Extend Buddy summary types**

In `persona-buddy.ts`, add optional:

```ts
active_visual_pack?: PersonaVisualPackSummary | null
```

Do not require this on existing payloads.

- [ ] **Step 4: Add shell visual-pack loading**

In `BuddyShellHost.tsx`:

- when `resolvedPersona.hasTargetPersona` is true, call `listPersonaVisualPacks(activePersonaId)`
- use `response.active_pack` when present; it must include `assets_by_id`
- if `active_pack` is absent but a pack list contains an active pack, call `getPersonaVisualPack(activePersonaId, activePackId)` to fetch the detail response
- keep failures non-blocking
- pass `visualPack` and resolved visual state to `BuddyShellDock`

- [ ] **Step 5: Add runtime override store**

Create `persona-visual-runtime.ts` with a small Zustand store:

```ts
export interface PersonaVisualRuntimeOverride {
  personaId: string
  sessionId: string | null
  state: PersonaVisualStateId
  reason: string | null
  expiresAt: number
}

export const usePersonaVisualRuntimeStore = create<{
  override: PersonaVisualRuntimeOverride | null
  setOverride: (override: PersonaVisualRuntimeOverride) => void
  clearExpired: (now?: number) => void
  clearForSession: (sessionId: string | null) => void
}>((set, get) => ({
  override: null,
  setOverride: (override) => set({ override }),
  clearExpired: (now = Date.now()) => {
    const current = get().override
    if (current && current.expiresAt <= now) set({ override: null })
  },
  clearForSession: (sessionId) => {
    const current = get().override
    if (current && current.sessionId === sessionId) set({ override: null })
  }
}))
```

In `sidepanel-persona.tsx`, handle incoming WebSocket payloads with `type === "visual_state_override"` and call `setOverride` with `Date.now() + duration_ms`.

- [ ] **Step 6: Preserve fallback behavior**

In `BuddyShellDock.tsx`, render `SpriteFrameRenderer` only when:

- not dormant
- `visualPack?.renderer_type === "sprite_frames"`
- manifest and `visualPack.assets_by_id` are available

Otherwise keep the current persona name/species text.

- [ ] **Step 7: Expose needed live controller status**

If the existing `usePersonaLiveVoiceController` return object does not already expose `activeToolStatus`, `wakeArmed`, and `state`, add them without changing existing callers.

- [ ] **Step 8: Add live visual-state feedback**

In `AssistantVoiceCard.tsx`, render a compact status row only when visual pack state is relevant to the current live session:

- current resolved visual state
- source: `live`, `default`, `override`, `authored_trigger`, `fallback`, or `error_recovery`
- active override reason when an MCP-triggered override is active
- fallback reason when the requested state falls back to idle or existing derived Buddy rendering

Keep this informational and non-blocking; it should not change live voice controls.

- [ ] **Step 9: Run tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx
```

Expected: PASS.

- [ ] **Step 10: Commit**

Run:

```bash
git add apps/packages/ui/src/types/persona-buddy.ts apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellDock.tsx apps/packages/ui/src/store/persona-visual-runtime.ts apps/packages/ui/src/routes/sidepanel-persona.tsx apps/packages/ui/src/hooks/usePersonaLiveVoiceController.tsx apps/packages/ui/src/components/PersonaGarden/AssistantVoiceCard.tsx apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx apps/packages/ui/src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx
git commit -m "feat: animate persona buddy shell from live state"
```

## Task 8: Persona Garden Pack Editor

**Files:**

- Create: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Modify: `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- Modify: `apps/packages/ui/src/utils/persona-garden-route.ts`
- Test: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [ ] **Step 1: Write failing editor tests**

Cover:

- loads pack list for selected persona
- creates draft pack
- uploads file with selected role
- edits required state mapping
- edits explicit animation frame order and persists the reordered `frames` array
- edits sprite-sheet region fields for a frame
- selects and persists `preview_frame`
- adds an authored trigger row and validates its target state
- displays validation errors
- disables activation when required states are missing
- activates pack when validation succeeds
- deactivates the active pack and returns the shell to derived Buddy rendering

- [ ] **Step 2: Run failing editor tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
```

Expected: FAIL because editor does not exist.

- [ ] **Step 3: Build editor component**

Use existing Persona Garden visual language. Keep controls dense and work-focused:

- pack selector/list
- create draft button
- upload input with role select
- state mapping table for `idle`, `listening`, `thinking`, `speaking`, `error`
- optional state mappings for `wake_armed`, `tool_running`, `approval_needed`, `offline`
- ordered frame list with move up/down controls
- sprite-sheet region inputs for `x`, `y`, `width`, and `height`
- preview-frame selector and preview metadata display
- frame rate input
- loop checkbox
- alignment x/y numeric inputs
- fallback state multi-select or ordered list
- authored trigger table with source, match, target state, duration, and priority controls
- preview/test state buttons
- validation panel
- activation button
- deactivate/revert button for the active pack
- generated candidate review section reserved for Task 9

- [ ] **Step 4: Add tab route**

In `sidepanel-persona.tsx`, lazy-load:

```ts
const LazyVisualPackEditor = React.lazy(() =>
  import("@/components/PersonaGarden/VisualPackEditor").then((module) => ({
    default: module.VisualPackEditor
  }))
)
```

Add a `visuals` tab only in persona mode.

- [ ] **Step 5: Run editor tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx apps/packages/ui/src/routes/sidepanel-persona.tsx apps/packages/ui/src/utils/persona-garden-route.ts apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
git commit -m "feat: add persona visual pack editor"
```

## Task 9: Generation Jobs And Candidate Review

**Files:**

- Create: `tldw_Server_API/app/core/Persona/visual_jobs.py`
- Create: `tldw_Server_API/app/core/Persona/visual_jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/app/services/startup_optional_workers.py`
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Test: `tldw_Server_API/tests/Persona/test_persona_visual_jobs.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`
- Test: `tldw_Server_API/tests/Services/test_persona_visual_jobs_worker_startup.py`
- Test: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [ ] **Step 1: Write failing Jobs tests**

Cover:

- `create_persona_visual_generation_job` creates domain `persona_visuals`
- idempotency key includes user/persona/pack/target state
- API rejects generation job creation for a pack the user/persona does not own
- worker fails cleanly when image backend unavailable
- worker stores generated asset and candidate when fake adapter succeeds
- optional-worker startup leaves the persona visual worker disabled by default
- optional-worker startup registers or starts the persona visual worker when `PERSONA_VISUAL_GENERATION_WORKER_ENABLED=1`
- accepting candidate updates draft manifest but does not activate pack
- candidate list endpoint returns generated asset preview URLs
- candidate manifest patch merge only writes state mapping, animation definitions, and authored triggers

- [ ] **Step 2: Run failing Jobs tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_jobs.py -v
```

Expected: FAIL because Jobs helpers do not exist.

- [ ] **Step 3: Implement job helpers**

In `visual_jobs.py`:

```python
PERSONA_VISUALS_DOMAIN = "persona_visuals"
PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE = "persona_visual_generate_candidate"

def persona_visual_generation_queue() -> str:
    return (os.getenv("PERSONA_VISUAL_GENERATION_JOBS_QUEUE") or "generation").strip() or "generation"

def build_generate_candidate_payload(*, user_id: str, persona_id: str, pack_id: str, prompt: str, target_state: str | None, backend: str | None) -> dict[str, Any]:
    return {
        "user_id": str(user_id),
        "persona_id": persona_id,
        "pack_id": pack_id,
        "prompt": prompt,
        "target_state": target_state,
        "backend": backend,
    }


def create_generate_candidate_job(jobs_manager: Any, *, user_id: str, persona_id: str, pack_id: str, prompt: str, target_state: str | None = None, backend: str | None = None) -> dict[str, Any]:
    return jobs_manager.create_job(
        domain=PERSONA_VISUALS_DOMAIN,
        queue=persona_visual_generation_queue(),
        job_type=PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE,
        payload=build_generate_candidate_payload(user_id=user_id, persona_id=persona_id, pack_id=pack_id, prompt=prompt, target_state=target_state, backend=backend),
        owner_user_id=str(user_id),
        idempotency_key=f"persona_visuals:{user_id}:{persona_id}:{pack_id}:{target_state or 'pack'}",
        max_retries=1,
    )
```

- [ ] **Step 4: Implement worker**

In `visual_jobs_worker.py`:

- resolve image backend through `ImageAdapterRegistry`
- build `ImageGenRequest` with width/height from manifest defaults or `1024x1024`
- offload sync adapter generation with `asyncio.to_thread`
- save bytes through `PersonaVisualService.add_generated_asset`
- create a candidate with a proposed manifest patch shaped as:

```json
{
  "states": {"thinking": {"animation_id": "generated-thinking-candidate"}},
  "animations": {
    "generated-thinking-candidate": {
      "asset_ids": ["generated-asset-id"],
      "frame_rate": 1,
      "loop": true,
      "alignment": {"x": 0.5, "y": 1.0}
    }
  },
  "authored_triggers": []
}
```

- merge candidate patches with an allowlist of `states`, `animations`, `fallbacks`, and `authored_triggers`
- reject patches that remove existing required states or reference assets outside the same pack
- record failure reason on candidate or job failure path

Mirror the async/sync boundary style in `tldw_Server_API/app/core/VN_Assets/worker.py`.

- [ ] **Step 5: Register optional generation worker startup**

In `startup_optional_workers.py`:

- add `persona_visual_generation_stop_event` and `persona_visual_generation_task` to `OptionalWorkerStartupHandles`
- call `_start_persona_visual_generation_worker` from `start_optional_workers` with the existing service handles
- guard startup with `PERSONA_VISUAL_GENERATION_WORKER_ENABLED`; disabled is the default
- when `worker_inventory` is present, register via `_start_registered_optional_worker` with category `persona` and name `persona_visual_generation_task`
- when `worker_inventory` is absent, create an `asyncio.Event` and `asyncio.create_task` using the same legacy optional-worker pattern as nearby workers
- implement `_run_persona_visual_generation_worker_service(stop_event)` as a narrow import wrapper around `run_persona_visual_generation_worker(stop_event)` from `visual_jobs_worker.py`

In `test_persona_visual_jobs_worker_startup.py`, cover:

```python
def test_persona_visual_generation_worker_disabled_by_default(monkeypatch) -> None:
    # Assert handles contain no persona visual task/stop event when the env flag
    # is absent or false.


def test_persona_visual_generation_worker_registers_when_enabled(monkeypatch) -> None:
    # Provide a fake worker_inventory and assert name, category, and coroutine
    # factory are registered without running the real worker.
```

- [ ] **Step 6: Add API generation endpoints**

Add:

```python
@router.post("/profiles/{persona_id}/visual-packs/{pack_id}/generation-jobs", tags=["persona"])
async def create_persona_visual_generation_job(persona_id: str, pack_id: str, payload: PersonaVisualGenerationRequest = Body(), current_user: User = Depends(get_request_user), db: CharactersRAGDB = Depends(get_chacha_db_for_user)) -> dict[str, Any]:
    user_id = _require_current_user_id(current_user)
    service = _persona_visual_service(db=db, current_user=current_user)
    service.get_pack_detail(persona_id=persona_id, pack_id=pack_id)
    job = create_generate_candidate_job(jobs_manager=get_jobs_manager(), user_id=user_id, persona_id=persona_id, pack_id=pack_id, prompt=payload.prompt, target_state=payload.target_state, backend=payload.backend)
    return {"job_id": str(job.get("id")), "status": job.get("status")}
```

Add accept/reject endpoints that call `PersonaVisualService.accept_candidate` and `PersonaVisualService.reject_candidate`.

Add candidate list/detail support before the editor controls:

- `PersonaVisualService.list_candidates(persona_id, pack_id)` returns review/failed candidates with generated asset response objects and authenticated preview URLs.
- `PersonaVisualService.get_candidate(persona_id, pack_id, candidate_id)` returns one candidate for preview.
- API endpoints return `PersonaVisualCandidateListResponse` and `PersonaVisualCandidateResponse`.
- Job status polling uses the normal Jobs API by `job_id`; the visual-pack API only returns the candidate review record and the originating `job_id`.

- [ ] **Step 7: Add editor candidate review controls**

In `VisualPackEditor.tsx`:

- generation prompt field
- target state selector
- enqueue button
- candidate list
- preview candidate
- job id/status link or compact polling status using the existing Jobs status API
- accept/reject buttons

- [ ] **Step 8: Run backend and frontend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_jobs.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py tldw_Server_API/tests/Services/test_persona_visual_jobs_worker_startup.py -v
cd apps/packages/ui && bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
git add tldw_Server_API/app/core/Persona/visual_jobs.py tldw_Server_API/app/core/Persona/visual_jobs_worker.py tldw_Server_API/app/core/Persona/visual_service.py tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/app/services/startup_optional_workers.py tldw_Server_API/tests/Persona/test_persona_visual_jobs.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py tldw_Server_API/tests/Services/test_persona_visual_jobs_worker_startup.py apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
git commit -m "feat: add persona visual generation review"
```

## Task 10: Internal persona_visuals MCP Module

**Files:**

- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Modify: `apps/packages/ui/src/store/persona-visual-runtime.ts`
- Modify: `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- Modify: `tldw_Server_API/Config_Files/mcp_modules.yaml`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py`

- [ ] **Step 1: Write failing MCP tests**

Cover:

- `persona_visuals.capabilities` returns active/draft pack summary
- `persona_visuals.trigger_state` rejects unknown state names
- `persona_visuals.trigger_state` caps duration
- draft asset/manifest tools do not activate packs
- generation tool creates a Job/review candidate
- module requires persona/user context
- Persona Live emits a `visual_state_override` WebSocket payload when a persona turn calls `persona_visuals.trigger_state`
- frontend runtime store receives and expires the override

- [ ] **Step 2: Run failing MCP tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py -v
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement module tool definitions**

In `persona_visuals_module.py`, subclass `BaseModule` and expose:

```python
persona_visuals.capabilities
persona_visuals.trigger_state
persona_visuals.create_draft_pack
persona_visuals.update_manifest
persona_visuals.enqueue_generation
```

Use `create_tool_definition` from `modules.base`.

- [ ] **Step 4: Implement context and validation**

Implementation requirements:

- open `CharactersRAGDB` from `context.db_paths["chacha"]`
- require `context.persona_scope` or explicit arguments to include persona id
- validate state names with `VISUAL_STATE_IDS`
- clamp `duration_ms` to `100 <= duration_ms <= 30_000`
- return transient trigger payloads with:

```python
{
    "type": "visual_state_override",
    "persona_id": persona_id,
    "session_id": session_id,
    "state": state,
    "duration_ms": duration_ms,
    "reason": reason,
}
```

- do not persist active-pack mutation for trigger-only calls
- durable calls use `PersonaVisualService` and create drafts/review items

- [ ] **Step 5: Wire Persona Live runtime propagation**

In `endpoints/persona.py`, update the Persona Live MCP execution path:

- after `_call_mcp_tool` returns, check whether `tool_name == "persona_visuals.trigger_state"`
- if the result payload has `type == "visual_state_override"`, emit that payload over the existing `WebSocketStream`
- include `persona_id`, `session_id`, `state`, `duration_ms`, `reason`, and triggering tool name
- record the same payload in existing persona audit/log status where tool execution metadata is already surfaced

In the frontend, `sidepanel-persona.tsx` should route that payload into `usePersonaVisualRuntimeStore.setOverride`.

- [ ] **Step 6: Add disabled module config**

In `mcp_modules.yaml`:

```yaml
  - id: persona_visuals
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.persona_visuals_module:PersonaVisualsModule
    enabled: false
    department: persona
    description: Persona visual pack draft and runtime state tools
```

- [ ] **Step 7: Run MCP tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py tldw_Server_API/app/api/v1/endpoints/persona.py apps/packages/ui/src/store/persona-visual-runtime.ts apps/packages/ui/src/routes/sidepanel-persona.tsx tldw_Server_API/Config_Files/mcp_modules.yaml tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py
git commit -m "feat: add persona visuals MCP module"
```

## Task 11: E2E Coverage, Bandit, And Documentation Closeout

**Files:**

- Modify: `apps/tldw-frontend/e2e/workflows/persona-live.spec.ts`
- Modify: `Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md` only if implementation learns a design correction.
- Modify through MCP: implementation Backlog task for the code slice created by the executor.

- [ ] **Step 1: Add E2E visual-state fixture**

Extend Persona Live E2E with mocked API responses:

- active persona has active visual pack
- shell shows `data-visual-state="idle"` initially
- mocked speaking/tool/error events update state
- broken pack response falls back without blocking live controls

- [ ] **Step 2: Run focused backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Persona/test_persona_visuals_core.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py \
  tldw_Server_API/tests/Persona/test_persona_visuals_api.py \
  tldw_Server_API/tests/Persona/test_persona_visual_jobs.py \
  tldw_Server_API/tests/Services/test_persona_visual_jobs_worker_startup.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py \
  -v
```

Expected: PASS.

- [ ] **Step 3: Run focused frontend tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts \
  src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx \
  src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx \
  src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
```

Expected: PASS.

- [ ] **Step 4: Run E2E fixture**

Run:

```bash
cd apps/tldw-frontend && bunx playwright test e2e/workflows/persona-live.spec.ts --reporter=line
```

Expected: PASS or documented environment blocker.

- [ ] **Step 5: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Persona \
  tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/app/services/startup_optional_workers.py \
  tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py \
  -f json -o /tmp/bandit_persona_visuals.json
```

Expected: no new findings in touched production code. If Bandit flags test-only or pre-existing noise, document exact finding ids and why they are not new production issues.

- [ ] **Step 6: Update Backlog task final summary**

Use Backlog MCP to record:

- files changed
- test commands and results
- Bandit result path
- any known skips/blockers
- PR or branch link if available

- [ ] **Step 7: Final commit**

Run:

```bash
git add apps/tldw-frontend/e2e/workflows/persona-live.spec.ts Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md
git commit -m "test: cover persona visual pack workflow"
```

Only include the design spec in this commit if it actually changed.

## Final Verification Gate

Before declaring implementation complete, run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Persona/test_persona_visuals_core.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py \
  tldw_Server_API/tests/Persona/test_persona_visuals_api.py \
  tldw_Server_API/tests/Persona/test_persona_visual_jobs.py \
  tldw_Server_API/tests/Services/test_persona_visual_jobs_worker_startup.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py \
  -v
```

```bash
cd apps/packages/ui && bunx vitest run \
  src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts \
  src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx \
  src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx \
  src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
```

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Persona \
  tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/app/services/startup_optional_workers.py \
  tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py \
  -f json -o /tmp/bandit_persona_visuals.json
```

If frontend E2E infrastructure is available:

```bash
cd apps/tldw-frontend && bunx playwright test e2e/workflows/persona-live.spec.ts --reporter=line
```

## Execution Handoff

Plan implementation should be staged and reviewed between tasks. Recommended execution mode is subagent-driven development with a fresh worker for each task group:

1. Backend core and DB persistence.
2. Backend API and upload validation.
3. Frontend renderer and shell integration.
4. Frontend pack editor.
5. Jobs and candidate review.
6. MCP module and final verification.

Do not begin execution until the user explicitly chooses an execution mode.

# Persona Buddy Sprite Atlas V1.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make sprite atlas packs an explicit, documented, regression-tested support path under the existing `sprite_frames` Persona/Buddy renderer.

**Architecture:** Keep the PR #1608 renderer capability boundary unchanged: `sprite_frames` remains the only activatable renderer, and `sprite_sheet` remains an asset role/future label rather than a renderer. The work is mostly characterization tests and docs because backend region validation and frontend cropped rendering already exist; any code changes should be minimal and only patch behavior that fails the new tests.

**Tech Stack:** Python 3.11, pytest, FastAPI Persona visual core tests, React, Vitest, Testing Library, TypeScript, Markdown docs, Bandit.

---

## File Map

- Modify: `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`
  - Add backend regression coverage for atlas manifests under `sprite_frames`.
  - Cover no-dimension permissive validation, malformed region rejection, and required-state activation with atlas frames.
- Modify only if tests expose a gap: `tldw_Server_API/app/core/Persona/visuals.py`
  - Existing `_validate_frame_region()` should already reject malformed regions and allow missing dimensions. Keep changes minimal.
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx`
  - Add dedicated atlas preview-frame coverage where two frames share the same atlas asset.
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx`
  - Add registry coverage proving atlas packs remain renderable under `sprite_frames`.
  - Add coarse renderability coverage for malformed regions so `SpriteFrameRenderer` can emit diagnostics.
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts`
  - Add or confirm diagnostics coverage for `unsupported_region` on atlas-backed packs.
- Modify only if tests expose a gap: `apps/packages/ui/src/components/Common/PersonaBuddy/SpriteFrameRenderer.tsx`
  - Existing region rendering and unsupported-region checks should already be sufficient.
- Modify only if tests expose a gap: `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualRenderers.tsx`
  - Registry `canRender` should stay coarse and not duplicate full region validation.
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`
  - Add a "Sprite Atlas Packs" section with the supported manifest shape.
- Modify: `backlog/tasks/task-301 - Plan-Persona-Buddy-sprite-atlas-V1.1-implementation.md`
  - Close the planning task after the plan is committed.
- Create during implementation: a new Backlog task for the implementation slice, for example `TASK-302`.

## Scope Guard

- Do not add `sprite_sheet` to backend renderer capabilities.
- Do not allow `renderer_type: "sprite_sheet"` to activate or validate as a renderer.
- Do not change `manifest_version` away from `1`.
- Do not add Persona Garden atlas authoring controls.
- Do not add image generation, automatic atlas packing, Live2D, VN/CYOA behavior, external provider behavior, or marketplace semantics.
- If a proposed edit touches unrelated Persona Garden flows, stop and reassess.

### Task 1: Backend Atlas Manifest Characterization

**Files:**
- Modify: `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`
- Modify only if needed: `tldw_Server_API/app/core/Persona/visuals.py`

- [ ] **Step 1: Create the implementation Backlog task**

Run:

```bash
backlog task create "Implement Persona Buddy sprite atlas V1.1 support" \
  --status "In Progress" \
  --priority medium \
  --labels persona-buddy,visual-packs,implementation \
  --ref https://github.com/rmusser01/tldw_server/issues/1611 \
  --doc Docs/superpowers/specs/2026-05-12-persona-buddy-sprite-atlas-v1-design.md \
  --doc Docs/superpowers/plans/2026-05-12-persona-buddy-sprite-atlas-v1-implementation-plan.md \
  --description "Implement the approved Persona/Buddy sprite atlas V1.1 hardening slice under sprite_frames. Scope is focused tests, docs, and minimal fixes only if current atlas behavior has gaps."
```

Expected: a new `backlog/tasks/task-*.md` file is created.

- [ ] **Step 2: Add backend atlas regression tests**

In `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`, add tests near the existing sprite-sheet region tests:

```python
def test_accepts_atlas_regions_without_known_dimensions_for_activation() -> None:
    manifest = _activatable_manifest()
    manifest["animations"] = {
        state: {
            "frames": [
                {
                    "asset_id": "atlas",
                    "region": {"x": index * 64, "y": 0, "width": 64, "height": 64},
                    "duration_ms": 100,
                }
            ],
            "frame_rate": 8,
            "preview_frame": 0,
        }
        for index, state in enumerate(["idle", "listen", "think", "speak", "error"])
    }

    result = validate_visual_manifest(
        manifest,
        available_asset_ids={"atlas"},
        available_asset_dimensions={},
        require_activatable=True,
    )

    assert set(result.resolved_required_states) == REQUIRED_VISUAL_STATES
    assert result.manifest["animations"]["idle"]["frames"][0]["region"]["width"] == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("x", 1.5),
        ("y", "0"),
        ("width", 0),
        ("height", -1),
    ],
)
def test_rejects_malformed_atlas_region_values(field: str, value: object) -> None:
    region = {"x": 0, "y": 0, "width": 64, "height": 64}
    region[field] = value
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {
            "idle": {
                "frames": [{"asset_id": "atlas", "region": region}],
            }
        },
    }

    with pytest.raises(PersonaVisualManifestError, match="region"):
        validate_visual_manifest(
            manifest,
            available_asset_ids={"atlas"},
            available_asset_dimensions={"atlas": (256, 256)},
            require_activatable=False,
        )
```

- [ ] **Step 3: Run backend tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Persona/test_persona_visuals_core.py -q
```

Expected: tests pass. If a test fails, inspect the failure and make the smallest change in `tldw_Server_API/app/core/Persona/visuals.py` that preserves the existing validation contract.

- [ ] **Step 4: Commit backend test/validation work**

Run:

```bash
git add tldw_Server_API/tests/Persona/test_persona_visuals_core.py tldw_Server_API/app/core/Persona/visuals.py
git commit -m "test: cover persona visual sprite atlas validation"
```

Expected: commit succeeds. If `visuals.py` was not modified, omit it from `git add`.

### Task 2: WebUI Atlas Renderer And Diagnostics Coverage

**Files:**
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts`
- Modify only if needed: `apps/packages/ui/src/components/Common/PersonaBuddy/SpriteFrameRenderer.tsx`
- Modify only if needed: `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualRenderers.tsx`

- [ ] **Step 1: Add atlas preview-frame renderer test**

In `SpriteFrameRenderer.test.tsx`, add a test after `renders sprite-sheet region frames as cropped background regions`:

```tsx
it("uses preview_frame for atlas animations that share one asset", () => {
  render(
    <SpriteFrameRenderer
      manifest={baseManifest({
        animations: {
          idle: {
            preview_frame: 1,
            frames: [
              {
                asset_id: "sheet-1",
                region: { x: 0, y: 0, width: 16, height: 16 },
                duration_ms: 100
              },
              {
                asset_id: "sheet-1",
                region: { x: 16, y: 0, width: 16, height: 16 },
                duration_ms: 100
              }
            ]
          }
        }
      })}
      assets={assets}
      state="idle"
      fallbackLabel="Buddy"
    />
  )

  expect(currentFrame()).toHaveStyle({
    backgroundPosition: "-16px 0px",
    width: "16px",
    height: "16px"
  })
})
```

- [ ] **Step 2: Add registry renderability tests for atlas packs**

In `personaVisualRenderers.test.tsx`, add tests near the renderability coverage:

```tsx
it("reports atlas-backed sprite frame packs as renderable", () => {
  expect(
    canRenderPersonaVisualPack(
      buildPack({
        manifest: {
          ...buildPack().manifest,
          animations: {
            idle: {
              frames: [
                {
                  asset_id: "idle-1",
                  region: { x: 0, y: 0, width: 16, height: 16 }
                }
              ]
            }
          }
        }
      })
    )
  ).toBe(true)
})

it("keeps renderability coarse so atlas region errors can be reported", () => {
  expect(
    canRenderPersonaVisualPack(
      buildPack({
        manifest: {
          ...buildPack().manifest,
          animations: {
            idle: {
              frames: [
                {
                  asset_id: "idle-1",
                  region: { x: 0, y: 0, width: 0, height: 16 }
                }
              ]
            }
          }
        }
      })
    )
  ).toBe(true)
})
```

- [ ] **Step 3: Add or confirm diagnostics coverage**

In `personaVisualDiagnostics.test.ts`, add a focused assertion if not already present:

```ts
it("reports unsupported atlas regions from renderer errors", () => {
  expect(
    resolvePersonaVisualDiagnostics({
      pack: buildPack(),
      visualState: "idle",
      renderError: "unsupported_region"
    })[0]
  ).toEqual(
    expect.objectContaining({
      code: "unsupported_region",
      severity: "warning"
    })
  )
})
```

Use the existing local test helpers in that file rather than introducing a second pack factory if one already exists.

- [ ] **Step 4: Run focused frontend tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run \
  src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx \
  src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx \
  src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts
```

Expected: tests pass. Existing `react-i18next` warnings are only relevant if these tests import a host component that needs i18n; the three focused files should normally avoid that warning.

- [ ] **Step 5: Patch minimal frontend behavior only if tests fail**

If the preview-frame test fails, adjust `resolveInitialFrameIndex()` in `SpriteFrameRenderer.tsx` without changing `preview_asset_id` behavior for non-atlas packs.

If the coarse renderability test fails, adjust `canRender` in `personaVisualRenderers.tsx` so it checks renderer type, manifest presence, and referenced asset existence only. Do not duplicate `hasUnsupportedRegion()` in the registry.

- [ ] **Step 6: Commit frontend test/runtime work**

Run:

```bash
git add \
  apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx \
  apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx \
  apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts \
  apps/packages/ui/src/components/Common/PersonaBuddy/SpriteFrameRenderer.tsx \
  apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualRenderers.tsx
git commit -m "test: cover persona buddy atlas rendering"
```

Expected: commit succeeds. Omit runtime files from `git add` if they were not modified.

### Task 3: Persona Visual Packs Documentation

**Files:**
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`

- [ ] **Step 1: Add the Sprite Atlas Packs docs section**

In `Docs/Code_Documentation/Persona_Visual_Packs.md`, add a section after "Manifest-Backed Pack Format":

````markdown
## Sprite Atlas Packs

Sprite atlas support is part of the existing `sprite_frames` renderer. In this
slice, `sprite_sheet` is an asset role, not a separate renderer type.
`renderer_type: "sprite_sheet"` is still rejected by the V1 renderer capability
contract.

Atlas-backed animations reference one raster asset and crop individual frames
with `frames[].region`:

```json
{
  "manifest_version": 1,
  "renderer_type": "sprite_frames",
  "states": {
    "idle": { "animation_id": "idle" }
  },
  "animations": {
    "idle": {
      "frames": [
        {
          "asset_id": "idle-atlas",
          "region": { "x": 0, "y": 0, "width": 128, "height": 128 },
          "duration_ms": 120
        }
      ],
      "preview_frame": 0
    }
  }
}
```

Use `preview_frame` when an atlas animation needs a specific preview crop.
`preview_asset_id` is better suited to separate-frame animations where each
preview candidate has a distinct asset id.
````

Keep the exact wording tight and avoid implying Persona Garden can author atlas regions visually in this slice.

- [ ] **Step 2: Run docs diff check**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 3: Commit docs**

Run:

```bash
git add Docs/Code_Documentation/Persona_Visual_Packs.md
git commit -m "docs: document persona sprite atlas packs"
```

Expected: commit succeeds.

### Task 4: Final Verification And PR Prep

**Files:**
- Modify: implementation Backlog task created in Task 1
- Possibly modify: `backlog/tasks/task-301 - Plan-Persona-Buddy-sprite-atlas-V1.1-implementation.md`

- [ ] **Step 1: Run focused backend test suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Persona/test_persona_visuals_core.py \
  tldw_Server_API/tests/Persona/test_persona_visuals_api.py \
  tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q
```

Expected: all focused Persona visual tests pass.

- [ ] **Step 2: Run focused frontend test suite**

Run from `apps/packages/ui`:

```bash
bunx vitest run \
  src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx \
  src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx \
  src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts \
  src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx \
  src/services/__tests__/persona-visuals.test.ts
```

Expected: all focused tests pass. Record any existing `react-i18next` warning if `BuddyShellHost.test.tsx` emits it.

- [ ] **Step 3: Run Bandit on touched backend scope**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit \
  -r tldw_Server_API/app/core/Persona/visuals.py \
  tldw_Server_API/tests/Persona/test_persona_visuals_core.py \
  -s B101 \
  -f json \
  -o /tmp/bandit_persona_buddy_sprite_atlas_v1.json
```

Expected: no findings. B101 is excluded for test assertions.

- [ ] **Step 4: Run diff checks and inspect status**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files changed.

- [ ] **Step 5: Update Backlog tasks**

Update the implementation task with:

- Completed acceptance criteria.
- Verification command outputs.
- Bandit result path.
- Known skips or warnings.
- Final summary.

Update `TASK-301` only if the plan changed during implementation.

- [ ] **Step 6: Commit final task updates**

Run:

```bash
git add backlog/tasks
git commit -m "chore: close persona buddy sprite atlas tasks"
```

Expected: commit succeeds if task files changed. If task closure was included in prior commits, skip this commit and record why.

- [ ] **Step 7: Push and open PR**

Run:

```bash
git push origin codex/persona-buddy-sprite-atlas-v1
gh pr create \
  --repo rmusser01/tldw_server \
  --base dev \
  --head codex/persona-buddy-sprite-atlas-v1 \
  --title "Harden Persona Buddy sprite atlas support" \
  --body "Closes #1611"
```

Expected: PR opens against `dev`. Fill in the project-required human change summary before merge if the PR is materially AI-authored.

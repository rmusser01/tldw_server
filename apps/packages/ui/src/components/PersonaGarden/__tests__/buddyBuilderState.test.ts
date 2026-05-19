import { describe, expect, it } from "vitest"

import {
  asPersonaVisualCustomStateId,
  type PersonaVisualImportPreviewResponse,
  type PersonaVisualManifest
} from "@/types/persona-visuals"

import {
  BASIC_BUDDY_STARTER_IDS,
  groupBuddyStarterPacksByTier,
  resetBuddyBuilderForSource,
  summarizeBuddyDraftReadiness
} from "../buddyBuilderState"

const makeManifest = (
  states: PersonaVisualManifest["states"] = {}
): PersonaVisualManifest => ({
  manifest_version: 1,
  renderer_type: "sprite_frames",
  states,
  animations: {},
  state_catalog: {}
})

const makePreview = (
  overrides: Partial<PersonaVisualImportPreviewResponse>
): PersonaVisualImportPreviewResponse => ({
  preview_id: "preview-1",
  job_id: "job-1",
  portability_job_id: "portability-1",
  operation: "import_preview",
  target_persona_id: "persona-1",
  status: "completed",
  visual_status: "completed",
  stage: "completed",
  bundle_summary: {},
  validation_warnings: [],
  conflicts: [],
  proposed_plan: {},
  quota_estimate: {},
  required_choices: [],
  target_warnings: [],
  ...overrides
})

describe("buddyBuilderState", () => {
  it("keeps the six Basic defaults in the expected order", () => {
    expect(BASIC_BUDDY_STARTER_IDS).toEqual([
      "search-lens-basic",
      "index-card-basic",
      "archive-cube-basic",
      "paperclip-basic",
      "terminal-tile-basic",
      "migu-marker-basic"
    ])
  })

  it("resets downstream import and draft state when the source changes", () => {
    const next = resetBuddyBuilderForSource(
      {
        source: "bundled",
        selectedStarterId: "search-lens-basic",
        selectedImportFile: new File(["zip"], "pet.zip"),
        importPreview: { job_id: "job-1" },
        selectedDraftPackId: "pack-1",
        activationReady: true
      },
      "codex_import"
    )

    expect(next.source).toBe("codex_import")
    expect(next.selectedStarterId).toBeNull()
    expect(next.selectedImportFile).toBeNull()
    expect(next.importPreview).toBeNull()
    expect(next.selectedDraftPackId).toBeNull()
    expect(next.activationReady).toBe(false)
  })

  it("groups starter packs by tier and treats only art-ready Basic packs as recommended", () => {
    const groups = groupBuddyStarterPacksByTier([
      {
        id: "search-lens-basic",
        title: "Search Lens",
        description: "",
        renderer_type: "sprite_frames",
        manifest_version: 1,
        states_offered: [],
        asset_count: 1,
        total_bytes: 1,
        tags: [],
        license_label: "bundled",
        complexity_tier: "basic",
        production_status: "art_ready",
        neutral_anchor_required: true,
        expected_asset_groups: [],
        animation_coverage_notes: []
      },
      {
        id: "paperclip-basic",
        title: "Paperclip",
        description: "",
        renderer_type: "sprite_frames",
        manifest_version: 1,
        states_offered: [],
        asset_count: 1,
        total_bytes: 1,
        tags: [],
        license_label: "bundled",
        complexity_tier: "basic",
        production_status: "scaffold",
        neutral_anchor_required: true,
        expected_asset_groups: [],
        animation_coverage_notes: []
      },
      {
        id: "lofi-study-intricate",
        title: "Lofi Study",
        description: "",
        renderer_type: "sprite_frames",
        manifest_version: 1,
        states_offered: [],
        asset_count: 1,
        total_bytes: 1,
        tags: [],
        license_label: "bundled",
        complexity_tier: "intricate",
        production_status: "scaffold",
        neutral_anchor_required: true,
        expected_asset_groups: [],
        animation_coverage_notes: []
      }
    ])

    expect(groups.basic[0].recommended).toBe(true)
    expect(groups.basic[1].recommended).toBe(false)
    expect(groups.intricate[0].recommended).toBe(false)
  })

  it("summarizes Codex/Petdex import semantics and atlas dimensions from backend preview metadata", () => {
    const summary = summarizeBuddyDraftReadiness({
      manifest: makeManifest({
        idle: { animation_id: "idle" },
        listening: { animation_id: "listening" },
        thinking: { animation_id: "thinking" },
        speaking: { animation_id: "speaking" },
        error: { animation_id: "error" }
      }),
      importPreview: makePreview({
        schema_version: "codex.pet.v1",
        bundle_summary: {
          assets: [
            {
              source_asset_id: "atlas",
              asset_role: "sprite_sheet",
              asset_group: "animation_atlas",
              width: 1536,
              height: 1872
            }
          ]
        }
      })
    })

    expect(summary.sourceLabel).toBe("Codex/Petdex pet")
    expect(summary.atlasSummary).toEqual([
      { assetId: "atlas", width: 1536, height: 1872 }
    ])
    expect(summary.canActivate).toBe(true)
  })

  it("distinguishes native Persona Visual archives from Codex previews without relying on MIME", () => {
    const native = summarizeBuddyDraftReadiness({
      manifest: makeManifest(),
      importPreview: makePreview({ schema_version: "persona_visual_pack.v1" })
    })
    const codex = summarizeBuddyDraftReadiness({
      manifest: makeManifest(),
      importPreview: makePreview({ schema_version: "codex.pet.v1" })
    })

    expect(native.sourceLabel).toBe("Persona Visual pack")
    expect(codex.sourceLabel).toBe("Codex/Petdex pet")
  })

  it("reports missing required state blockers and movement state coverage", () => {
    const manifest = makeManifest({
      idle: { animation_id: "idle" },
      listening: { animation_id: "listening" },
      [asPersonaVisualCustomStateId("moving_right")]: { animation_id: "move-r" }
    })
    manifest.state_catalog = {
      [asPersonaVisualCustomStateId("moving_right")]: {
        label: "Moving right",
        kind: "live_variant",
        description: "Drag movement to the right"
      },
      [asPersonaVisualCustomStateId("moving_left")]: {
        label: "Moving left",
        kind: "live_variant",
        description: "Drag movement to the left"
      }
    }

    const summary = summarizeBuddyDraftReadiness({ manifest })

    expect(summary.blockers).toEqual(
      expect.arrayContaining([
        "Missing required state: thinking",
        "Missing required state: speaking",
        "Missing required state: error"
      ])
    )
    expect(summary.canActivate).toBe(false)
    expect(summary.movementStates).toEqual([
      { id: "moving_left", resolved: false },
      { id: "moving_right", resolved: true }
    ])
  })
})

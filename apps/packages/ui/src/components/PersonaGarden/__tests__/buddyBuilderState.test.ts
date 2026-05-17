import { describe, expect, it } from "vitest"

import {
  BASIC_BUDDY_STARTER_IDS,
  groupBuddyStarterPacksByTier,
  resetBuddyBuilderForSource
} from "../buddyBuilderState"

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
})

import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  fetchWithAuth: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    fetchWithAuth: (...args: unknown[]) => mocks.fetchWithAuth(...args)
  }
}))

import {
  copyPersonaVisualStarterPack,
  getPersonaVisualRendererCapabilities,
  getPersonaVisualStarterPack,
  listPersonaVisualStarterPacks
} from "../persona-visuals"

describe("persona visuals service", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("loads renderer capabilities through the persona visual-renderers endpoint", async () => {
    const renderers = [
      {
        renderer_type: "sprite_frames",
        display_name: "Sprite Frames",
        manifest_versions: [1],
        can_validate: true,
        can_activate: true,
        buddy_runtime_supported: true,
        import_supported: true,
        export_supported: true,
        disabled_reason: null,
        renderer_contract_versions: [1],
        supported_asset_roles: ["frame", "sprite_sheet"],
        required_role_categories: [],
        role_category_map: { sprite_sheet: ["sprite_sheet"] },
        allowed_mime_types: ["image/png", "image/jpeg", "image/webp", "image/gif"],
        allowed_extensions: [".png", ".jpg", ".jpeg", ".webp", ".gif"],
        max_file_count: 256,
        max_total_bytes: 104857600,
        max_texture_width: 4096,
        max_texture_height: 4096,
        feature_flag: null,
        setup_status: "supported",
        setup_blockers: [],
        requires_static_fallback: false,
        requires_license_ack: false
      }
    ]
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ renderers })
    })

    const response = await getPersonaVisualRendererCapabilities()

    expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
      "/api/v1/persona/visual-renderers",
      expect.objectContaining({ method: "GET" })
    )
    expect(response).toEqual({ renderers })
  })

  it("normalizes malformed renderer capability responses to an empty list", async () => {
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({})
    })

    await expect(getPersonaVisualRendererCapabilities()).resolves.toEqual({
      renderers: []
    })

    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ renderers: {} })
    })

    await expect(getPersonaVisualRendererCapabilities()).resolves.toEqual({
      renderers: []
    })
  })

  it("loads bundled starter packs through the starter catalog endpoint", async () => {
    const starterPack = {
      id: "search-lens-basic",
      title: "Search Lens Buddy",
      description: "A deterministic sprite-frame starter.",
      renderer_type: "sprite_frames",
      manifest_version: 1,
      states_offered: ["idle", "listening", "thinking", "speaking", "error"],
      asset_count: 1,
      total_bytes: 92,
      tags: ["starter", "sprite_frames"],
      license_label: "bundled",
      complexity_tier: "basic",
      production_status: "art_ready",
      neutral_anchor_required: true,
      expected_asset_groups: [
        "identity_brief",
        "neutral_anchor",
        "preview_image",
        "required_state_loops"
      ],
      animation_coverage_notes: [
        "Reviewed bundled basic default with neutral-anchor-derived required-state loops."
      ]
    }
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ starter_packs: [starterPack] })
    })

    await expect(listPersonaVisualStarterPacks()).resolves.toEqual({
      starter_packs: [starterPack]
    })
    expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
      "/api/v1/persona/visual-starter-packs",
      expect.objectContaining({ method: "GET" })
    )
  })

  it("normalizes direct-list starter pack responses for defensive tests", async () => {
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => [{ id: "starter-1", title: "Starter 1" }]
    })

    await expect(listPersonaVisualStarterPacks()).resolves.toEqual({
      starter_packs: [
        expect.objectContaining({
          id: "starter-1",
          complexity_tier: "basic",
          production_status: "scaffold",
          neutral_anchor_required: true,
          expected_asset_groups: [],
          animation_coverage_notes: []
        })
      ]
    })
  })

  it("copies a bundled starter pack into a target persona draft", async () => {
    const copiedPack = {
      id: "starter-copy-1",
      persona_id: "persona-1",
      title: "Search Lens Buddy",
      renderer_type: "sprite_frames",
      status: "draft",
      manifest: {
        manifest_version: 1,
        renderer_type: "sprite_frames",
        states: {},
        animations: {}
      },
      assets: []
    }
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 201,
      json: async () => copiedPack
    })

    await expect(
      copyPersonaVisualStarterPack("search-lens-basic", {
        target_persona_id: "persona-1",
        title: "Starter copy"
      })
    ).resolves.toEqual(copiedPack)

    expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
      "/api/v1/persona/visual-starter-packs/search-lens-basic/copy",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "Content-Type": "application/json"
        }),
        body: JSON.stringify({
          target_persona_id: "persona-1",
          title: "Starter copy"
        })
      })
    )
  })

  it("normalizes malformed starter pack wrapper responses to an empty list", async () => {
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({})
    })

    await expect(listPersonaVisualStarterPacks()).resolves.toEqual({
      starter_packs: []
    })

    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ starter_packs: {} })
    })

    await expect(listPersonaVisualStarterPacks()).resolves.toEqual({
      starter_packs: []
    })
  })

  it("loads starter pack details with an encoded id", async () => {
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        id: "starter/with space",
        title: "Starter Detail",
        description: "Detail",
        renderer_type: "sprite_frames",
        manifest_version: 1,
        states_offered: ["idle"],
        asset_count: 1,
        total_bytes: 128,
        tags: [],
        license_label: "bundled",
        manifest: {
          manifest_version: 1,
          renderer_type: "sprite_frames",
          states: {},
          animations: {}
        },
        assets: []
      })
    })

    await getPersonaVisualStarterPack("starter/with space")

    expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
      "/api/v1/persona/visual-starter-packs/starter%2Fwith%20space",
      expect.objectContaining({ method: "GET" })
    )
  })

  it("does not route empty starter pack detail ids to the collection endpoint", async () => {
    await expect(getPersonaVisualStarterPack("")).rejects.toThrow(
      "Starter pack id is required"
    )
    expect(mocks.fetchWithAuth).not.toHaveBeenCalled()
  })

  it("copies a starter pack to a target persona without activation fields", async () => {
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 201,
      json: async () => ({
        id: "copied-pack",
        persona_id: "persona-1",
        title: "Search Lens Buddy",
        renderer_type: "sprite_frames",
        status: "draft",
        manifest: {
          manifest_version: 1,
          renderer_type: "sprite_frames",
          states: {},
          animations: {}
        }
      })
    })

    await copyPersonaVisualStarterPack("starter-1", {
      target_persona_id: "persona-1"
    })

    expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
      "/api/v1/persona/visual-starter-packs/starter-1/copy",
      expect.objectContaining({ method: "POST" })
    )
    const [, init] = mocks.fetchWithAuth.mock.calls.at(-1) as [string, any]
    expect(init.headers).toEqual(
      expect.objectContaining({ "Content-Type": "application/json" })
    )
    expect(JSON.parse(String(init.body))).toEqual({
      target_persona_id: "persona-1"
    })
  })

  it("does not route empty starter pack copy ids to the collection endpoint", async () => {
    await expect(
      copyPersonaVisualStarterPack("", {
        target_persona_id: "persona-1"
      })
    ).rejects.toThrow("Starter pack id is required")
    expect(mocks.fetchWithAuth).not.toHaveBeenCalled()
  })

  it("rejects whitespace-only starter pack ids before fetch", async () => {
    await expect(getPersonaVisualStarterPack("   ")).rejects.toThrow(
      "Starter pack id is required"
    )
    expect(mocks.fetchWithAuth).not.toHaveBeenCalled()
  })
})

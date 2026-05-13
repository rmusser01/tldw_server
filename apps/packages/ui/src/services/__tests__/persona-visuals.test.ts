import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  fetchWithAuth: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    fetchWithAuth: (...args: unknown[]) => mocks.fetchWithAuth(...args)
  }
}))

import { getPersonaVisualRendererCapabilities } from "../persona-visuals"

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
        allowed_mime_types: ["image/png", "image/webp"],
        allowed_extensions: [".png", ".webp"],
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
})

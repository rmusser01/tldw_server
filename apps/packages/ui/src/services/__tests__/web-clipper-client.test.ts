import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  resolveApiPath: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: vi.fn(),
  bgStream: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"
import { buildWebClipSaveRuntime, mapWebClipperOutcomeToBanner } from "@/services/web-clipper/save-runtime"

describe("web clipper client and runtime helpers", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.spyOn(TldwApiClient.prototype, "resolveApiPath").mockImplementation(
      async (_key: string, candidates: string[]) => candidates[0] as never
    )
  })

  it("posts web clip saves to the canonical save endpoint", async () => {
    mocks.bgRequest.mockResolvedValue({
      clip_id: "clip-123",
      note_id: "clip-123",
      status: "saved",
      workspace_placement_saved: false,
      workspace_placement_count: 0,
      warnings: []
    })

    const client = new TldwApiClient()
    await client.saveWebClip({
      clip_id: "clip-123",
      clip_type: "article",
      source_url: "https://example.com/story",
      source_title: "Example Story",
      destination_mode: "note",
      note: { title: "Example Story", comment: null, folder_id: null, keywords: [] },
      content: { visible_body: "Alpha", full_extract: "Alpha", selected_text: "Alpha" },
      attachments: [],
      enhancements: { run_ocr: false, run_vlm: false },
      capture_metadata: { captured_via: "browser" }
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/web-clipper/save",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: expect.objectContaining({
          clip_id: "clip-123",
          source_url: "https://example.com/story"
        })
      })
    )
  })

  it("polls web clip status by clip id", async () => {
    mocks.bgRequest.mockResolvedValue({
      clip_id: "clip-123",
      status: "saved",
      note: { id: "clip-123", title: "Example Story", version: 1 },
      workspace_placements: [],
      attachments: [],
      analysis: {},
      content_budget: {}
    })

    const client = new TldwApiClient()
    await client.getWebClipStatus("clip-123")

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/web-clipper/clip-123",
        method: "GET"
      })
    )
  })

  it("posts enrichment payloads with source note version intact", async () => {
    mocks.bgRequest.mockResolvedValue({
      clip_id: "clip-123",
      enrichment_type: "ocr",
      status: "complete",
      source_note_version: 7,
      inline_applied: false,
      inline_summary: "Summary",
      conflict_reason: "source_note_version_mismatch",
      warnings: ["User edited note before enrichment completed."]
    })

    const client = new TldwApiClient()
    await client.persistWebClipEnrichment("clip-123", {
      clip_id: "clip-123",
      enrichment_type: "ocr",
      status: "complete",
      inline_summary: "Summary",
      structured_payload: { raw_text: "Summary" },
      source_note_version: 7
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/web-clipper/clip-123/enrichments",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: expect.objectContaining({
          clip_id: "clip-123",
          source_note_version: 7
        })
      })
    )
  })

  it.each([
    [
      "saved",
      {
        severity: "success",
        title: "Clip saved",
        message: "The clip was saved successfully."
      }
    ],
    [
      "saved_with_warnings",
      {
        severity: "warning",
        title: "Clip saved with warnings",
        message: "The clip was saved, but follow-up work reported warnings."
      }
    ],
    [
      "partially_saved",
      {
        severity: "warning",
        title: "Clip partially saved",
        message: "Some clip stages completed, but at least one destination failed."
      }
    ],
    [
      "failed",
      {
        severity: "error",
        title: "Clip save failed",
        message: "The clip could not be saved."
      }
    ]
  ] as const)(
    "maps backend state %s to a warning-preserving banner",
    (status, expectedBanner) => {
      const banner = mapWebClipperOutcomeToBanner({
        status,
        warnings: ["attachment slot page-screenshot failed"]
      })

      expect(banner).toMatchObject(expectedBanner)
      expect(banner.warnings).toEqual(["attachment slot page-screenshot failed"])
    }
  )

  it("keeps warning details when building the save runtime", () => {
    const runtime = buildWebClipSaveRuntime({
      clip_id: "clip-123",
      note_id: "clip-123",
      status: "saved_with_warnings",
      workspace_placement_saved: true,
      workspace_placement_count: 1,
      warnings: ["Attachment upload failed for slot page-screenshot"]
    })

    expect(runtime.banner.severity).toBe("warning")
    expect(runtime.banner.warnings).toEqual([
      "Attachment upload failed for slot page-screenshot"
    ])
    expect(runtime.warnings).toEqual([
      "Attachment upload failed for slot page-screenshot"
    ])
  })
})

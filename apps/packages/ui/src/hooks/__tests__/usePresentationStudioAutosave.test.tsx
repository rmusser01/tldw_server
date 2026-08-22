import React from "react"
import { act, render } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { usePresentationStudioAutosave } from "../usePresentationStudioAutosave"
import { usePresentationStudioStore } from "@/store/presentation-studio"

const clientMocks = vi.hoisted(() => ({
  patchPresentation: vi.fn(),
  getPresentation: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  clonePresentationVisualStyleSnapshot: (style: Record<string, unknown> | null | undefined) =>
    style ? { ...style } : null,
  tldwClient: {
    patchPresentation: (...args: unknown[]) => clientMocks.patchPresentation(...args),
    getPresentation: (...args: unknown[]) => clientMocks.getPresentation(...args)
  }
}))

const AutosaveHarness: React.FC = () => {
  usePresentationStudioAutosave()
  return null
}

describe("usePresentationStudioAutosave", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    clientMocks.patchPresentation.mockReset()
    clientMocks.getPresentation.mockReset()
    usePresentationStudioStore.getState().reset()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("merges remote updates and retries after a 412 without dropping local edits", async () => {
    usePresentationStudioStore.getState().loadProject(
      {
        id: "presentation-1",
        title: "Deck",
        description: null,
        theme: "black",
        visual_style_id: "minimal-academic",
        visual_style_scope: "builtin",
        visual_style_name: "Minimal Academic",
        visual_style_version: 1,
        visual_style_snapshot: {
          id: "minimal-academic",
          scope: "builtin",
          name: "Minimal Academic",
          appearance_defaults: { theme: "white" }
        },
        slides: [
          {
            order: 0,
            layout: "content",
            title: "Opening",
            content: "Server copy",
            speaker_notes: "Original narration",
            metadata: {
              studio: {
                slideId: "slide-1",
                audio: { status: "ready", asset_ref: "output:1" },
                image: { status: "missing" }
              }
            }
          }
        ],
        studio_data: { origin: "blank" },
        created_at: "2026-03-13T00:00:00Z",
        last_modified: "2026-03-13T00:00:00Z",
        deleted: false,
        client_id: "1",
        version: 1
      },
      { etag: 'W/"v1"' }
    )
    usePresentationStudioStore.getState().updateSlide("slide-1", {
      speaker_notes: "Local narration edit"
    })

    clientMocks.patchPresentation
      .mockRejectedValueOnce(new Error("412 precondition_failed"))
      .mockResolvedValueOnce({
        id: "presentation-1",
        title: "Deck",
        description: null,
        theme: "black",
        visual_style_id: "minimal-academic",
        visual_style_scope: "builtin",
        visual_style_name: "Minimal Academic",
        visual_style_version: 1,
        visual_style_snapshot: {
          id: "minimal-academic",
          scope: "builtin",
          name: "Minimal Academic",
          appearance_defaults: { theme: "white" }
        },
        slides: [
          {
            order: 0,
            layout: "content",
            title: "Opening",
            content: "Server copy",
            speaker_notes: "Local narration edit",
            metadata: {
              studio: {
                slideId: "slide-1",
                audio: { status: "stale", asset_ref: "output:1" },
                image: { status: "missing" }
              }
            }
          },
          {
            order: 1,
            layout: "content",
            title: "Remote slide",
            content: "Added elsewhere",
            speaker_notes: "",
            metadata: {
              studio: {
                slideId: "slide-2",
                audio: { status: "missing" },
                image: { status: "missing" }
              }
            }
          }
        ],
        studio_data: { origin: "blank" },
        created_at: "2026-03-13T00:00:00Z",
        last_modified: "2026-03-13T00:01:00Z",
        deleted: false,
        client_id: "1",
        version: 3
      })
    clientMocks.getPresentation.mockResolvedValue({
      record: {
        id: "presentation-1",
        content_kind: "structured_slides",
        title: "Deck",
        description: null,
        theme: "black",
        visual_style_id: "minimal-academic",
        visual_style_scope: "builtin",
        visual_style_name: "Minimal Academic",
        visual_style_version: 1,
        visual_style_snapshot: {
          id: "minimal-academic",
          scope: "builtin",
          name: "Minimal Academic",
          appearance_defaults: { theme: "white" }
        },
        slides: [
          {
            order: 0,
            layout: "content",
            title: "Opening",
            content: "Server copy",
            speaker_notes: "Original narration",
            metadata: {
              studio: {
                slideId: "slide-1",
                audio: { status: "ready", asset_ref: "output:1" },
                image: { status: "missing" }
              }
            }
          },
          {
            order: 1,
            layout: "content",
            title: "Remote slide",
            content: "Added elsewhere",
            speaker_notes: "",
            metadata: {
              studio: {
                slideId: "slide-2",
                audio: { status: "missing" },
                image: { status: "missing" }
              }
            }
          }
        ],
        studio_data: { origin: "blank" },
        created_at: "2026-03-13T00:00:00Z",
        last_modified: "2026-03-13T00:01:00Z",
        deleted: false,
        client_id: "1",
        version: 2
      },
      etag: 'W/"server-tag"'
    })

    render(<AutosaveHarness />)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(900)
    })

    await vi.waitFor(() => {
      expect(clientMocks.patchPresentation).toHaveBeenCalledTimes(2)
    })
    expect(clientMocks.getPresentation).toHaveBeenCalledWith("presentation-1")
    expect(clientMocks.patchPresentation.mock.calls[1][2]).toEqual({
      ifMatch: 'W/"server-tag"'
    })
    expect(clientMocks.patchPresentation.mock.calls[1][1]).toEqual(
      expect.objectContaining({
        visual_style_id: "minimal-academic",
        visual_style_scope: "builtin",
        visual_style_name: "Minimal Academic",
        visual_style_version: 1,
        visual_style_snapshot: expect.objectContaining({
          id: "minimal-academic",
          scope: "builtin",
          name: "Minimal Academic"
        }),
        slides: expect.arrayContaining([
          expect.objectContaining({
            speaker_notes: "Local narration edit"
          }),
          expect.objectContaining({
            title: "Remote slide"
          })
        ])
      })
    )

    const state = usePresentationStudioStore.getState()
    expect(state.slides[0]?.speaker_notes).toBe("Local narration edit")
    expect(state.slides[1]?.title).toBe("Remote slide")
    expect(state.isDirty).toBe(false)
  })

  it.each([
    {
      content_kind: "standalone_html",
      html_document: "<!doctype html><title>Private draft</title>",
      html_sha256: "a".repeat(64),
      html_bytes: 50,
      html_slide_count: 1
    },
    {
      content_kind: "unsupported",
      unsupported_content_kind: "future_canvas",
      read_only: true
    }
  ])("does not retain a $content_kind record in the structured store", (record) => {
    const store = usePresentationStudioStore.getState()
    const before = JSON.stringify(store)

    expect(() =>
      store.loadProject({
        id: "unsafe-1",
        title: "Unsafe",
        description: null,
        theme: "black",
        created_at: "2026-07-15T00:00:00Z",
        last_modified: "2026-07-15T00:00:01Z",
        deleted: false,
        client_id: "1",
        version: 1,
        ...record
      } as never)
    ).toThrow("Structured presentation required")
    expect(JSON.stringify(usePresentationStudioStore.getState())).toBe(before)
    expect(JSON.stringify(usePresentationStudioStore.getState())).not.toContain("Private draft")
  })

  it("stops structured conflict recovery when detail resolves to standalone HTML", async () => {
    usePresentationStudioStore.getState().loadProject(
      {
        id: "presentation-1",
        content_kind: "structured_slides",
        title: "Deck",
        description: null,
        theme: "black",
        slides: [],
        created_at: "2026-03-13T00:00:00Z",
        last_modified: "2026-03-13T00:00:00Z",
        deleted: false,
        client_id: "1",
        version: 1
      },
      { etag: 'W/"v1"' }
    )
    usePresentationStudioStore.getState().updateProjectMeta({ title: "Local edit" })
    clientMocks.patchPresentation.mockRejectedValueOnce(new Error("412 precondition_failed"))
    clientMocks.getPresentation.mockResolvedValue({
      record: {
        ...standaloneDetailForAutosave,
        html_document: "<!doctype html><title>Must stay out of store</title>"
      },
      etag: '"v2"'
    })

    render(<AutosaveHarness />)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(900)
    })

    expect(clientMocks.patchPresentation).toHaveBeenCalledTimes(1)
    expect(usePresentationStudioStore.getState().autosaveState).toBe("error")
    expect(JSON.stringify(usePresentationStudioStore.getState())).not.toContain(
      "Must stay out of store"
    )
  })
})

const standaloneDetailForAutosave = {
  id: "presentation-1",
  content_kind: "standalone_html",
  title: "HTML deck",
  description: null,
  theme: "black",
  source_type: "prompt",
  source_ref: null,
  source_query: null,
  created_at: "2026-03-13T00:00:00Z",
  last_modified: "2026-03-13T00:01:00Z",
  deleted: false,
  client_id: "1",
  version: 2,
  html_sha256: "a".repeat(64),
  html_bytes: 50,
  html_slide_count: 1,
  generation_provenance: {}
}

import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PresentationStudioPage } from "../PresentationStudioPage"
import { usePresentationStudioStore } from "@/store/presentation-studio"

const onlineMocks = vi.hoisted(() => ({
  useServerOnline: vi.fn()
}))

const capabilityMocks = vi.hoisted(() => ({
  useServerCapabilities: vi.fn()
}))

const routerMocks = vi.hoisted(() => ({
  navigate: vi.fn()
}))

const clientMocks = vi.hoisted(() => ({
  createPresentation: vi.fn(),
  getPresentation: vi.fn(),
  listVisualStyles: vi.fn()
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => onlineMocks.useServerOnline()
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => capabilityMocks.useServerCapabilities()
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return {
    ...actual,
    useNavigate: () => routerMocks.navigate
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  buildPresentationVisualStyleSnapshot: (style: Record<string, unknown>) => ({
    ...style
  }),
  clonePresentationVisualStyleSnapshot: (style: Record<string, unknown> | null | undefined) =>
    style ? { ...style } : null,
  tldwClient: {
    createPresentation: (...args: unknown[]) => clientMocks.createPresentation(...args),
    getPresentation: (...args: unknown[]) => clientMocks.getPresentation(...args),
    listVisualStyles: (...args: unknown[]) => clientMocks.listVisualStyles(...args)
  }
}))

describe("PresentationStudioPage bootstrap", () => {
  beforeEach(() => {
    usePresentationStudioStore.getState().reset()
    routerMocks.navigate.mockReset()
    clientMocks.createPresentation.mockReset()
    clientMocks.getPresentation.mockReset()
    clientMocks.listVisualStyles.mockReset()
    onlineMocks.useServerOnline.mockReturnValue(true)
    capabilityMocks.useServerCapabilities.mockReturnValue({
      loading: false,
      capabilities: {
        hasSlides: true,
        hasPresentationStudio: true,
        hasPresentationRender: true
      }
    })
    clientMocks.listVisualStyles.mockResolvedValue([
      {
        id: "minimal-academic",
        name: "Minimal Academic",
        scope: "builtin",
        description: "Structured, restrained, study-first slides.",
        generation_rules: {},
        artifact_preferences: [],
        appearance_defaults: { theme: "white" },
        fallback_policy: {},
        version: 1
      }
    ])
  })

  it("loads the precreate form in strict mode without creating until submit", async () => {
    let resolveCreate: ((value: any) => void) | null = null
    clientMocks.createPresentation.mockReturnValue(
      new Promise((resolve) => {
        resolveCreate = resolve
      })
    )

    render(
      <React.StrictMode>
        <PresentationStudioPage mode="new" />
      </React.StrictMode>
    )

    await waitFor(() => {
      expect(clientMocks.listVisualStyles).toHaveBeenCalled()
    })
    expect(clientMocks.createPresentation).not.toHaveBeenCalled()

    const createButton = await screen.findByTestId("presentation-studio-create-button")
    createButton.click()

    await waitFor(() => {
      expect(clientMocks.createPresentation).toHaveBeenCalledTimes(1)
    })

    resolveCreate?.({
      id: "presentation-strict",
      content_kind: "structured_slides",
      title: "Untitled Presentation",
      description: null,
      theme: "white",
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
          layout: "title",
          title: "Title slide",
          content: "",
          speaker_notes: "",
          metadata: {
            studio: {
              slideId: "slide-strict",
              transition: "fade",
              timing_mode: "auto",
              manual_duration_ms: null,
              audio: { status: "missing" },
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
    })

    await waitFor(() => {
      expect(routerMocks.navigate).toHaveBeenCalledWith(
        "/presentation-studio/presentation-strict",
        {
          replace: true
        }
      )
    })
  })

  it("leaves the loading state after a detail fetch resolves", async () => {
    clientMocks.getPresentation.mockResolvedValue({
      record: {
        id: "presentation-load",
        title: "Loaded Presentation",
        description: null,
        theme: "black",
        content_kind: "structured_slides",
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
            layout: "title",
            title: "Loaded slide",
            content: "",
            speaker_notes: "",
            metadata: {
              studio: {
                slideId: "slide-load",
                transition: "fade",
                timing_mode: "auto",
                manual_duration_ms: null,
                audio: { status: "missing" },
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
      etag: '"server-detail-etag"'
    })

    render(
      <React.StrictMode>
        <PresentationStudioPage mode="detail" projectId="presentation-load" />
      </React.StrictMode>
    )

    expect(screen.getByText("Loading presentation…")).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByTestId("presentation-studio-slide-rail")).toBeInTheDocument()
    })

    expect(screen.queryByText("Loading presentation…")).not.toBeInTheDocument()
  })
})

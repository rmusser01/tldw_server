import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
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
  listVisualStyles: vi.fn(),
  createVisualStyle: vi.fn(),
  patchVisualStyle: vi.fn(),
  deleteVisualStyle: vi.fn()
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
    listVisualStyles: (...args: unknown[]) => clientMocks.listVisualStyles(...args),
    createVisualStyle: (...args: unknown[]) => clientMocks.createVisualStyle(...args),
    patchVisualStyle: (...args: unknown[]) => clientMocks.patchVisualStyle(...args),
    deleteVisualStyle: (...args: unknown[]) => clientMocks.deleteVisualStyle(...args)
  }
}))

const visualStyles = [
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
  },
  {
    id: "timeline",
    name: "Timeline",
    scope: "builtin",
    description: "Chronology-forward deck structure.",
    generation_rules: {},
    artifact_preferences: ["timeline"],
    appearance_defaults: { theme: "beige" },
    fallback_policy: {},
    version: 1
  }
]

describe("PresentationStudioPage", () => {
  beforeEach(() => {
    usePresentationStudioStore.getState().reset()
    routerMocks.navigate.mockReset()
    clientMocks.createPresentation.mockReset()
    clientMocks.getPresentation.mockReset()
    clientMocks.listVisualStyles.mockReset()
    clientMocks.createVisualStyle.mockReset()
    clientMocks.patchVisualStyle.mockReset()
    clientMocks.deleteVisualStyle.mockReset()
    onlineMocks.useServerOnline.mockReturnValue(true)
    capabilityMocks.useServerCapabilities.mockReturnValue({
      loading: false,
      capabilities: {
        hasSlides: true,
        hasPresentationStudio: true,
        hasPresentationRender: true
      }
    })
    clientMocks.listVisualStyles.mockResolvedValue(visualStyles)
  })

  it("creates a blank project from the precreate form and redirects to its detail route", async () => {
    clientMocks.createPresentation.mockResolvedValue({
      id: "presentation-1",
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
              slideId: "slide-1",
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

    render(<PresentationStudioPage mode="new" />)

    await waitFor(() => {
      expect(clientMocks.listVisualStyles).toHaveBeenCalledTimes(1)
    })
    expect(
      screen.getByText(
        /Updates deck appearance defaults and future generated slides\. Existing slide content stays unchanged\./
      )
    ).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("presentation-studio-create-button"))

    await waitFor(() => {
      expect(clientMocks.createPresentation).toHaveBeenCalledTimes(1)
    })
    expect(clientMocks.createPresentation).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Untitled Presentation",
        visual_style_id: "minimal-academic",
        visual_style_scope: "builtin",
        visual_style_name: "Minimal Academic",
        visual_style_version: 1,
        visual_style_snapshot: expect.objectContaining({
          id: "minimal-academic",
          scope: "builtin",
          name: "Minimal Academic"
        }),
        studio_data: { origin: "blank", entry_surface: "webui_new" },
        slides: [
          expect.objectContaining({
            metadata: {
              studio: expect.objectContaining({
                transition: "fade",
                timing_mode: "auto",
                manual_duration_ms: null
              })
            }
          })
        ]
      })
    )
    expect(routerMocks.navigate).toHaveBeenCalledWith("/presentation-studio/presentation-1", {
      replace: true
    })
    expect(usePresentationStudioStore.getState().projectId).toBe("presentation-1")
  })

  it("completes blank project creation in strict mode", async () => {
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

    fireEvent.click(await screen.findByTestId("presentation-studio-create-button"))

    await waitFor(() => {
      expect(clientMocks.createPresentation).toHaveBeenCalledTimes(1)
    })

    resolveCreate?.({
      id: "presentation-strict",
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

  it("loads a detail project and leaves the loading state", async () => {
    clientMocks.getPresentation.mockResolvedValue({
      record: {
        id: "presentation-load",
        content_kind: "structured_slides",
        title: "Loaded Presentation",
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
            layout: "title",
            title: "Loaded slide",
            content: "",
            speaker_notes: "",
            metadata: {
              studio: {
                slideId: "slide-load",
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
      etag: 'W/"opaque-server-tag"'
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
    expect(usePresentationStudioStore.getState().etag).toBe('W/"opaque-server-tag"')
    fireEvent.click(screen.getByRole("button", { name: /Timeline/ }))

    await waitFor(() => {
      expect(usePresentationStudioStore.getState().theme).toBe("beige")
    })
  })

  it("updates the deck visual style preference on the detail page without mutating slides", async () => {
    clientMocks.getPresentation.mockResolvedValue({
      record: {
        id: "presentation-style",
        content_kind: "structured_slides",
        title: "Styled Deck",
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
            layout: "title",
            title: "Loaded slide",
            content: "",
            speaker_notes: "",
            metadata: {
              studio: {
                slideId: "slide-load",
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
      etag: 'W/"v1"'
    })

    render(<PresentationStudioPage mode="detail" projectId="presentation-style" />)

    await screen.findByText("Timeline")
    fireEvent.click(screen.getByRole("button", { name: /Timeline/ }))

    const state = usePresentationStudioStore.getState()
    expect(state.visualStyleId).toBe("timeline")
    expect(state.visualStyleScope).toBe("builtin")
    expect(state.visualStyleName).toBe("Timeline")
    expect(state.slides).toHaveLength(1)
    expect(state.theme).toBe("beige")
  })

  it("creates a custom visual style and selects it in new mode", async () => {
    clientMocks.listVisualStyles
      .mockResolvedValueOnce(visualStyles)
      .mockResolvedValueOnce([
        ...visualStyles,
        {
          id: "style-user-1",
          name: "CGPSC Sprint",
          scope: "user",
          description: "Recall-heavy revision deck",
          generation_rules: { exam_focus: "high" },
          artifact_preferences: ["timeline", "comparison_matrix"],
          appearance_defaults: { theme: "beige" },
          fallback_policy: {},
          version: 1
        }
      ])
    clientMocks.createVisualStyle.mockResolvedValue({
      id: "style-user-1",
      name: "CGPSC Sprint",
      scope: "user",
      description: "Recall-heavy revision deck",
      generation_rules: { exam_focus: "high" },
      artifact_preferences: ["timeline", "comparison_matrix"],
      appearance_defaults: { theme: "beige" },
      fallback_policy: {},
      version: 1
    })

    render(<PresentationStudioPage mode="new" />)

    fireEvent.click(await screen.findByTestId("presentation-studio-new-custom-style"))
    fireEvent.change(screen.getByLabelText("Custom style name"), {
      target: { value: "CGPSC Sprint" }
    })
    fireEvent.change(screen.getByLabelText("Custom style description"), {
      target: { value: "Recall-heavy revision deck" }
    })
    fireEvent.change(screen.getByLabelText("Default theme"), {
      target: { value: "beige" }
    })
    fireEvent.change(screen.getByLabelText("Content density"), {
      target: { value: "high" }
    })
    fireEvent.change(screen.getByLabelText("Bullet emphasis"), {
      target: { value: "high" }
    })
    fireEvent.click(screen.getByLabelText("Timeline artifact preference"))
    fireEvent.click(screen.getByLabelText("Comparison Matrix artifact preference"))
    fireEvent.click(screen.getByLabelText("Exam focus emphasis signal"))
    fireEvent.click(screen.getByText("Advanced JSON overrides"))
    fireEvent.change(screen.getByLabelText("Additional generation rules JSON"), {
      target: { value: '{"visual_callouts":"high"}' }
    })

    fireEvent.click(screen.getByTestId("presentation-studio-save-custom-style"))

    await waitFor(() => {
      expect(clientMocks.createVisualStyle).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "CGPSC Sprint",
          description: "Recall-heavy revision deck",
          artifact_preferences: ["timeline", "comparison_matrix"],
          appearance_defaults: { theme: "beige" },
          generation_rules: {
            density: "high",
            bullet_bias: "high",
            exam_focus: true,
            visual_callouts: "high"
          },
          fallback_policy: {
            mode: "outline",
            preserve_key_stats: true
          }
        })
      )
    })
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /CGPSC Sprint/ })).toHaveAttribute(
        "aria-pressed",
        "true"
      )
    })
  })

  it("edits and deletes a selected custom visual style in detail mode", async () => {
    clientMocks.getPresentation.mockResolvedValue({
      record: {
        id: "presentation-custom-style",
        content_kind: "structured_slides",
        title: "Styled Deck",
        description: null,
        theme: "black",
        visual_style_id: "style-user-1",
        visual_style_scope: "user",
        visual_style_name: "CGPSC Sprint",
        visual_style_version: 1,
        visual_style_snapshot: {
          id: "style-user-1",
          scope: "user",
          name: "CGPSC Sprint",
          description: "Recall-heavy revision deck",
          appearance_defaults: { theme: "beige" }
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
      etag: 'W/"v1"'
    })
    clientMocks.listVisualStyles
      .mockResolvedValueOnce([
        ...visualStyles,
        {
          id: "style-user-1",
          name: "CGPSC Sprint",
          scope: "user",
          description: "Recall-heavy revision deck",
          generation_rules: { exam_focus: "high" },
          artifact_preferences: ["timeline"],
          appearance_defaults: { theme: "beige" },
          fallback_policy: {},
          version: 1
        }
      ])
      .mockResolvedValueOnce([
        ...visualStyles,
        {
          id: "style-user-1",
          name: "CGPSC Rapid Review",
          scope: "user",
          description: "Updated revision deck",
          generation_rules: { exam_focus: "high" },
          artifact_preferences: ["timeline"],
          appearance_defaults: { theme: "moon" },
          fallback_policy: {},
          version: 2
        }
      ])
      .mockResolvedValueOnce(visualStyles)
    clientMocks.patchVisualStyle.mockResolvedValue({
      id: "style-user-1",
      name: "CGPSC Rapid Review",
      scope: "user",
      description: "Updated revision deck",
      generation_rules: { exam_focus: "high" },
      artifact_preferences: ["timeline"],
      appearance_defaults: { theme: "moon" },
      fallback_policy: {},
      version: 2
    })
    clientMocks.deleteVisualStyle.mockResolvedValue(undefined)

    render(<PresentationStudioPage mode="detail" projectId="presentation-custom-style" />)

    fireEvent.click(await screen.findByTestId("presentation-studio-edit-custom-style"))
    fireEvent.change(screen.getByLabelText("Custom style name"), {
      target: { value: "CGPSC Rapid Review" }
    })
    fireEvent.change(screen.getByLabelText("Custom style description"), {
      target: { value: "Updated revision deck" }
    })
    fireEvent.change(screen.getByLabelText("Default theme"), {
      target: { value: "moon" }
    })
    fireEvent.click(screen.getByTestId("presentation-studio-save-custom-style"))

    await waitFor(() => {
      expect(clientMocks.patchVisualStyle).toHaveBeenCalledWith(
        "style-user-1",
        expect.objectContaining({
          name: "CGPSC Rapid Review",
          description: "Updated revision deck",
          appearance_defaults: { theme: "moon" }
        })
      )
    })
    await waitFor(() => {
      expect(usePresentationStudioStore.getState().visualStyleName).toBe(
        "CGPSC Rapid Review"
      )
    })

    fireEvent.click(screen.getByTestId("presentation-studio-delete-custom-style"))

    await waitFor(() => {
      expect(clientMocks.deleteVisualStyle).toHaveBeenCalledWith("style-user-1")
    })
    await waitFor(() => {
      expect(usePresentationStudioStore.getState().visualStyleId).toBeNull()
    })
  })

  it("shows stale audio and ready image badges for seeded slide media state", () => {
    usePresentationStudioStore.getState().loadProject(
      {
        id: "presentation-1",
        title: "Seeded deck",
        description: null,
        theme: "black",
        slides: [
          {
            order: 0,
            layout: "content",
            title: "Opening",
            content: "",
            speaker_notes: "Narration seed",
            metadata: {
              studio: {
                slideId: "slide-1",
                audio: { status: "stale" },
                image: { status: "ready", asset_ref: "output:123" }
              }
            }
          }
        ],
        studio_data: { origin: "extension_capture" },
        created_at: "2026-03-13T00:00:00Z",
        last_modified: "2026-03-13T00:00:00Z",
        deleted: false,
        client_id: "1",
        version: 2
      },
      { etag: 'W/"v2"' }
    )

    render(<PresentationStudioPage mode="detail" projectId="presentation-1" />)

    const mediaRail = screen.getByTestId("presentation-studio-media-rail")
    expect(within(mediaRail).getByText("Audio status")).toBeInTheDocument()
    expect(within(mediaRail).getByText("Image status")).toBeInTheDocument()
    expect(within(mediaRail).getAllByText("stale").length).toBeGreaterThan(0)
    expect(within(mediaRail).getAllByText("ready").length).toBeGreaterThan(0)
  })

  it("explains readiness issues and narration timing for the selected slide", () => {
    usePresentationStudioStore.getState().loadProject(
      {
        id: "presentation-readiness",
        title: "Readiness deck",
        description: null,
        theme: "black",
        slides: [
          {
            order: 0,
            layout: "content",
            title: "Problem framing",
            content: "Summarize the core problem.",
            speaker_notes: "Open with the core problem.",
            metadata: {
              studio: {
                slideId: "slide-readiness",
                audio: { status: "stale", duration_ms: 92_000 },
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
        version: 3
      },
      { etag: 'W/"v3"' }
    )

    render(<PresentationStudioPage mode="detail" projectId="presentation-readiness" />)

    const mediaRail = screen.getByTestId("presentation-studio-media-rail")
    expect(within(mediaRail).getByText("Narration timing")).toBeInTheDocument()
    expect(within(mediaRail).getAllByText("1m 32s").length).toBeGreaterThan(0)
    expect(
      within(mediaRail).getAllByText("Refresh narration to match the latest script changes.")
        .length
    ).toBeGreaterThan(0)
    expect(
      within(mediaRail).getAllByText("Add or generate a slide image before publishing.").length
    ).toBeGreaterThan(0)
    expect(within(mediaRail).getByText("Slides with stale narration")).toBeInTheDocument()
    expect(within(mediaRail).getByText("Estimated narration length")).toBeInTheDocument()
  })

  it("surfaces task-oriented editing guidance and slide controls for the active slide", () => {
    usePresentationStudioStore.getState().loadProject(
      {
        id: "presentation-ux",
        title: "Narrated deck",
        description: null,
        theme: "black",
        slides: [
          {
            order: 0,
            layout: "content",
            title: "Intro",
            content: "Frame the talk in a single sentence.",
            speaker_notes: "Set up the presentation.",
            metadata: {
              studio: {
                slideId: "slide-ux-1",
                audio: { status: "ready", asset_ref: "output:1" },
                image: { status: "ready", asset_ref: "output:2" }
              }
            }
          },
          {
            order: 1,
            layout: "content",
            title: "Problem framing",
            content: "Summarize the core problem, constraints, and target audience.",
            speaker_notes:
              "Open with the core problem, then explain who the presentation is for.",
            metadata: {
              studio: {
                slideId: "slide-ux-2",
                audio: { status: "stale" },
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
        version: 2
      },
      { etag: 'W/"v2"' }
    )
    usePresentationStudioStore.getState().selectSlide("slide-ux-2")

    render(<PresentationStudioPage mode="detail" projectId="presentation-ux" />)

    expect(screen.getByRole("button", { name: "Duplicate slide" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Move earlier" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Move later" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Delete slide" })).toBeEnabled()
    expect(screen.getByRole("heading", { name: "On-slide copy" })).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Narration" })).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Preview" })).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Deck readiness" })).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Draft sync" })).toBeInTheDocument()
    expect(screen.getAllByText("Ready to render").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Needs attention").length).toBeGreaterThan(0)
    expect(
      screen.getByRole("button", { name: "Drag to reorder slide 1" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Drag to reorder slide 2" })
    ).toBeInTheDocument()
    expect(
      screen.getByText("This text is visible on the slide itself.")
    ).toBeInTheDocument()
    expect(
      screen.getByText("This script is spoken in the generated narration audio.")
    ).toBeInTheDocument()
  })

  it("lets the editor configure transition and manual slide timing", () => {
    usePresentationStudioStore.getState().loadProject(
      {
        id: "presentation-timing",
        title: "Timed deck",
        description: null,
        theme: "black",
        slides: [
          {
            order: 0,
            layout: "content",
            title: "Intro",
            content: "Frame the opening.",
            speaker_notes: "Open the deck.",
            metadata: {
              studio: {
                slideId: "slide-timing",
                audio: { status: "ready", duration_ms: 18_000, asset_ref: "output:1" },
                image: { status: "ready", asset_ref: "output:2" }
              }
            }
          }
        ],
        studio_data: { origin: "blank" },
        created_at: "2026-03-13T00:00:00Z",
        last_modified: "2026-03-13T00:00:00Z",
        deleted: false,
        client_id: "1",
        version: 2
      },
      { etag: 'W/"v2"' }
    )

    render(<PresentationStudioPage mode="detail" projectId="presentation-timing" />)

    expect(screen.getByRole("heading", { name: "Transitions & timing" })).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("Transition"), {
      target: { value: "wipe" }
    })
    fireEvent.change(screen.getByLabelText("Duration mode"), {
      target: { value: "manual" }
    })
    fireEvent.change(screen.getByLabelText("Manual duration (seconds)"), {
      target: { value: "45" }
    })

    const mediaRail = screen.getByTestId("presentation-studio-media-rail")
    expect(within(mediaRail).getByText("Transition")).toBeInTheDocument()
    expect(within(mediaRail).getByText("Wipe")).toBeInTheDocument()
    expect(within(mediaRail).getByText("Effective duration")).toBeInTheDocument()
    expect(within(mediaRail).getAllByText("45s").length).toBeGreaterThan(0)
  })
})

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import OptionPresentationStudioStart from "../option-presentation-studio-start"

const onlineMocks = vi.hoisted(() => ({
  useServerOnline: vi.fn()
}))

const capabilityMocks = vi.hoisted(() => ({
  useServerCapabilities: vi.fn()
}))

const connectionMocks = vi.hoisted(() => ({
  useConnectionState: vi.fn()
}))

const clientMocks = vi.hoisted(() => ({
  createPresentation: vi.fn(),
  getConfig: vi.fn()
}))

vi.mock("@/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({
    routeId,
    children
  }: {
    routeId: string
    children: React.ReactNode
  }) => <div data-testid={`route-boundary-${routeId}`}>{children}</div>
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => onlineMocks.useServerOnline()
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => capabilityMocks.useServerCapabilities()
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => connectionMocks.useConnectionState()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    createPresentation: (...args: unknown[]) => clientMocks.createPresentation(...args),
    getConfig: (...args: unknown[]) => clientMocks.getConfig(...args)
  }
}))

describe("presentation studio start route", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    onlineMocks.useServerOnline.mockReturnValue(true)
    capabilityMocks.useServerCapabilities.mockReturnValue({
      loading: false,
      capabilities: {
        hasSlides: true,
        hasPresentationStudio: true,
        hasPresentationRender: true
      }
    })
    connectionMocks.useConnectionState.mockReturnValue({
      serverUrl: "http://127.0.0.1:8000"
    })
    clientMocks.createPresentation.mockResolvedValue({
      id: "presentation-123",
      title: "Deck",
      description: null,
      theme: "black",
      studio_data: { origin: "blank" },
      slides: [],
      created_at: "2026-03-13T00:00:00Z",
      last_modified: "2026-03-13T00:00:00Z",
      deleted: false,
      client_id: "1",
      version: 1
    })
    clientMocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user"
    })
    vi.spyOn(window, "open").mockImplementation(() => null)
  })

  it("creates a blank project and opens the WebUI editor", async () => {
    render(<OptionPresentationStudioStart />)

    fireEvent.change(screen.getByLabelText("Project title"), {
      target: { value: "Launch Plan" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Create blank project" }))

    await waitFor(() =>
      expect(clientMocks.createPresentation).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Launch Plan",
          studio_data: expect.objectContaining({
            origin: "blank",
            entry_surface: "extension_start"
          }),
          slides: [
            expect.objectContaining({
              order: 0,
              layout: "title",
              title: "Launch Plan",
              speaker_notes: ""
            })
          ]
        })
      )
    )

    await waitFor(() => {
      expect(window.open).toHaveBeenCalledWith(
        "http://127.0.0.1:8080/presentation-studio/presentation-123",
        "_blank",
        "noopener,noreferrer"
      )
    })
  })

  it("creates a seeded project when narration is provided", async () => {
    render(<OptionPresentationStudioStart />)

    fireEvent.change(screen.getByLabelText("Project title"), {
      target: { value: "Quarterly Review" }
    })
    fireEvent.change(screen.getByLabelText("Narration seed"), {
      target: { value: "Open with the key revenue changes across the quarter." }
    })
    fireEvent.click(screen.getByRole("button", { name: "Create seeded project" }))

    await waitFor(() =>
      expect(clientMocks.createPresentation).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Quarterly Review",
          studio_data: expect.objectContaining({
            origin: "extension_capture",
            entry_surface: "extension_start"
          }),
          slides: [
            expect.objectContaining({
              order: 0,
              layout: "content",
              title: "Quarterly Review",
              speaker_notes: "Open with the key revenue changes across the quarter."
            })
          ]
        })
      )
    )
  })

  it("blocks the route when presentation studio is unsupported", () => {
    capabilityMocks.useServerCapabilities.mockReturnValue({
      loading: false,
      capabilities: {
        hasSlides: true,
        hasPresentationStudio: false,
        hasPresentationRender: false
      }
    })

    render(<OptionPresentationStudioStart />)

    expect(
      screen.getByText("Presentation Studio is not available on this server.")
    ).toBeInTheDocument()
  })
})

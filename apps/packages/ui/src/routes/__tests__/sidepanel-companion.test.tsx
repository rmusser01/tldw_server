import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

const mocks = vi.hoisted(() => ({
  capabilitiesState: { loading: false, capabilities: { hasPersonalization: true, hasCompanion: true } },
  isCompanionConsentRequiredError: vi.fn((error: { status?: number; message?: string }) => {
    return (
      error?.status === 409 &&
      String(error?.message || "").includes("Enable personalization before using companion.")
    )
  }),
  recordExplicitCompanionCapture: vi.fn(async (..._args: unknown[]) => null)
}))

vi.mock("@/hooks/useServerCapabilities", () => ({ useServerCapabilities: () => mocks.capabilitiesState }))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/components/Option/CompanionHome", () => ({
  CompanionHomeShell: ({
    surface,
    onPersonalizationEnabled
  }: {
    surface: "options" | "sidepanel"
    onPersonalizationEnabled?: () => void
  }) => (
    <div data-testid="companion-home-shell">
      {surface}
      <button
        data-testid="companion-home-personalization-enabled"
        onClick={() => onPersonalizationEnabled?.()}
        type="button"
      >
        personalization enabled
      </button>
    </div>
  )
}))

vi.mock("@/services/companion", () => ({
  isCompanionConsentRequiredError: (error?: unknown) =>
    mocks.isCompanionConsentRequiredError(error as { status?: number; message?: string }),
  recordExplicitCompanionCapture: (...args: unknown[]) =>
    mocks.recordExplicitCompanionCapture(...args)
}))

vi.mock("~/components/Sidepanel/Chat/SidepanelHeaderSimple", () => ({
  SidepanelHeaderSimple: ({ activeTitle }: { activeTitle?: string }) => (
    <div data-testid="sidepanel-header">{activeTitle || "header"}</div>
  )
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

import SidepanelCompanion from "../sidepanel-companion"

const renderRoute = () =>
  render(
    <MemoryRouter>
      <SidepanelCompanion />
    </MemoryRouter>
  )

describe("SidepanelCompanion", () => {
  it.each([
    { loading: true, hasPersonalization: true, hasCompanion: true },
    { loading: false, hasPersonalization: true, hasCompanion: false },
    { loading: false, hasPersonalization: false, hasCompanion: true }
  ])("retains pending capture until Companion router availability resolves (%j)", async (state) => {
    mocks.capabilitiesState = { loading: state.loading, capabilities: {
      hasPersonalization: state.hasPersonalization, hasCompanion: state.hasCompanion
    } }
    const capture = { id: "capture-1", selectionText: "Remember this paragraph.",
      pageUrl: "https://example.com/article", pageTitle: "Example article", action: "save_selection" }
    const pending = JSON.stringify(capture)
    window.sessionStorage.setItem("tldw:companion:pendingCapture", pending)
    const view = renderRoute()
    await screen.findByTestId("companion-home-shell")
    expect(mocks.recordExplicitCompanionCapture).not.toHaveBeenCalled()
    window.dispatchEvent(new CustomEvent("tldw:companion-pending-capture", { detail: capture }))
    expect(mocks.recordExplicitCompanionCapture).not.toHaveBeenCalled()
    expect(window.sessionStorage.getItem("tldw:companion:pendingCapture")).toBe(pending)

    mocks.capabilitiesState = { loading: false, capabilities: { hasPersonalization: true, hasCompanion: true } }
    view.rerender(<MemoryRouter><SidepanelCompanion /></MemoryRouter>)
    await waitFor(() => expect(mocks.recordExplicitCompanionCapture).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(window.sessionStorage.getItem("tldw:companion:pendingCapture")).toBeNull())
    window.dispatchEvent(new CustomEvent("tldw:companion-pending-capture", { detail: capture }))
    expect(mocks.recordExplicitCompanionCapture).toHaveBeenCalledTimes(1)
  })

  beforeEach(() => {
    vi.clearAllMocks()
    mocks.capabilitiesState = { loading: false, capabilities: { hasPersonalization: true, hasCompanion: true } }
    window.sessionStorage.clear()
    mocks.recordExplicitCompanionCapture.mockResolvedValue({
      id: "activity-1",
      event_type: "extension.selection_saved",
      source_type: "browser_selection",
      source_id: "capture-1",
      surface: "extension.sidepanel",
      tags: ["extension", "selection"],
      provenance: {
        capture_mode: "explicit",
        route: "extension.context_menu",
        action: "save_selection"
      },
      metadata: {
        selection: "Remember this paragraph.",
        page_url: "https://example.com/article",
        page_title: "Example article"
      },
      created_at: "2026-03-10T12:00:00Z"
    })
  })

  it("renders the Companion Home shell in the sidepanel wrapper", async () => {
    renderRoute()

    expect(await screen.findByTestId("companion-home-shell")).toBeInTheDocument()
    expect(screen.getByText("sidepanel")).toBeInTheDocument()
    expect(screen.getByTestId("sidepanel-header")).toHaveTextContent("Companion")
  })

  it("XS-17: pins a way back to chat above the Companion dashboard", async () => {
    renderRoute()

    const shell = await screen.findByTestId("companion-home-shell")
    const chatLink = screen.getByRole("link", { name: "Open chat" })
    expect(chatLink).toHaveAttribute("href", "/?view=chat")
    expect(
      chatLink.compareDocumentPosition(shell) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()
  })

  it("records the pending selection capture when the companion route opens", async () => {
    window.sessionStorage.setItem(
      "tldw:companion:pendingCapture",
      JSON.stringify({
        id: "capture-1",
        selectionText: "Remember this paragraph.",
        pageUrl: "https://example.com/article",
        pageTitle: "Example article",
        action: "save_selection"
      })
    )

    renderRoute()

    await waitFor(() => {
      expect(mocks.recordExplicitCompanionCapture).toHaveBeenCalledWith(
        expect.objectContaining({
          event_type: "extension.selection_saved",
          source_type: "browser_selection",
          source_id: "capture-1",
          surface: "extension.sidepanel",
          dedupe_key: "extension.selection_saved:capture-1",
          metadata: expect.objectContaining({
            selection: "Remember this paragraph.",
            page_url: "https://example.com/article",
            page_title: "Example article"
          }),
          provenance: expect.objectContaining({
            capture_mode: "explicit",
            action: "save_selection"
          })
        })
      )
    })
  })

  it("keeps a pending capture and shows a consent-required banner when opt-in is missing", async () => {
    window.sessionStorage.setItem(
      "tldw:companion:pendingCapture",
      JSON.stringify({
        id: "capture-1",
        selectionText: "Remember this paragraph.",
        pageUrl: "https://example.com/article",
        pageTitle: "Example article",
        action: "save_selection"
      })
    )
    mocks.recordExplicitCompanionCapture.mockRejectedValue(
      Object.assign(new Error("Enable personalization before using companion."), {
        status: 409
      })
    )

    renderRoute()

    expect(
      await screen.findByText("Enable personalization before saving to companion.")
    ).toBeInTheDocument()
    expect(window.sessionStorage.getItem("tldw:companion:pendingCapture")).toContain(
      "\"capture-1\""
    )
  })

  it("retries the pending capture after personalization is enabled from companion home", async () => {
    window.sessionStorage.setItem(
      "tldw:companion:pendingCapture",
      JSON.stringify({
        id: "capture-1",
        selectionText: "Remember this paragraph.",
        pageUrl: "https://example.com/article",
        pageTitle: "Example article",
        action: "save_selection"
      })
    )
    mocks.recordExplicitCompanionCapture
      .mockRejectedValueOnce(
        Object.assign(new Error("Enable personalization before using companion."), {
          status: 409
        })
      )
      .mockResolvedValueOnce({
        id: "activity-1",
        event_type: "extension.selection_saved",
        source_type: "browser_selection",
        source_id: "capture-1",
        surface: "extension.sidepanel",
        tags: ["extension", "selection"],
        provenance: {
          capture_mode: "explicit",
          route: "extension.context_menu",
          action: "save_selection"
        },
        metadata: {
          selection: "Remember this paragraph.",
          page_url: "https://example.com/article",
          page_title: "Example article"
        },
        created_at: "2026-03-10T12:00:00Z"
      })

    renderRoute()

    expect(
      await screen.findByText("Enable personalization before saving to companion.")
    ).toBeInTheDocument()
    expect(window.sessionStorage.getItem("tldw:companion:pendingCapture")).toContain(
      "\"capture-1\""
    )
    expect(mocks.recordExplicitCompanionCapture).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByTestId("companion-home-personalization-enabled"))

    await waitFor(() => {
      expect(mocks.recordExplicitCompanionCapture).toHaveBeenCalledTimes(2)
    })
    expect(await screen.findByText("Saved selection to companion.")).toBeInTheDocument()
    expect(window.sessionStorage.getItem("tldw:companion:pendingCapture")).toBeNull()
  })
})

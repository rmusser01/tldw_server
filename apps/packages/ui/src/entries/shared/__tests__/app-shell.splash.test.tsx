import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { afterAll, beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/hooks/useTheme", () => ({
  useTheme: () => ({ antdTheme: {} })
}))

vi.mock("@/components/Common/PageAssistProvider", () => ({
  PageAssistProvider: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  )
}))

vi.mock("@/context/FontSizeProvider", () => ({
  FontSizeProvider: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  )
}))

vi.mock("@/components/Common/LocaleJsonDiagnostics", () => ({
  LocaleJsonDiagnostics: () => null
}))

const visualLoads = vi.hoisted(() => ({ barrel: 0, leaf: 0 }))
vi.mock("@/components/Common/SplashScreen", () => {
  visualLoads.barrel++
  return {
    SplashOverlay: ({ message }: { message: string }) => (
      <div data-testid="splash-overlay">{message}</div>
    )
  }
})
vi.mock("@/components/Common/SplashScreen/SplashOverlay", () => {
  visualLoads.leaf++
  return {
    default: ({ message }: { message: string }) => (
      <div data-testid="splash-overlay">{message}</div>
    )
  }
})

import { SPLASH_TRIGGER_EVENT } from "@/services/splash-events"
import { AppShell } from "../AppShell"

const Router: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <>{children}</>
)

const originalVisibilityStateDescriptor = Object.getOwnPropertyDescriptor(
  Document.prototype,
  "visibilityState"
)

const setDocumentVisibility = (state: DocumentVisibilityState) => {
  Object.defineProperty(Document.prototype, "visibilityState", {
    configurable: true,
    get: () => state
  })
  document.dispatchEvent(new Event("visibilitychange"))
}

describe("AppShell splash integration", () => {
  beforeEach(() => {
    localStorage.clear()
    setDocumentVisibility("visible")
  })

  afterAll(() => {
    if (originalVisibilityStateDescriptor) {
      Object.defineProperty(
        Document.prototype,
        "visibilityState",
        originalVisibilityStateDescriptor
      )
    }
  })

  it("does not import optional visuals for hidden application content", () => {
    render(
      <AppShell router={Router} direction="ltr" emptyDescription="No data">
        <button>Usable chat</button>
      </AppShell>
    )
    expect(screen.getByRole("button", { name: "Usable chat" })).toBeEnabled()
    expect(visualLoads.barrel + visualLoads.leaf).toBe(0)
  })

  it("shows splash overlay when splash trigger event fires", async () => {
    render(
      <AppShell router={Router} direction="ltr" emptyDescription="No data">
        <div>App content</div>
      </AppShell>
    )

    expect(screen.queryByTestId("splash-overlay")).not.toBeInTheDocument()

    window.dispatchEvent(new CustomEvent(SPLASH_TRIGGER_EVENT))

    await waitFor(() => {
      expect(screen.getByTestId("splash-overlay")).toBeInTheDocument()
    })
  })

  it("shows splash on forced login trigger even when splash preference is disabled", async () => {
    localStorage.setItem("tldw_splash_disabled", "true")
    render(
      <AppShell router={Router} direction="ltr" emptyDescription="No data">
        <div>App content</div>
      </AppShell>
    )

    expect(screen.queryByTestId("splash-overlay")).not.toBeInTheDocument()

    window.dispatchEvent(
      new CustomEvent(SPLASH_TRIGGER_EVENT, { detail: { force: true } })
    )

    await waitFor(() => {
      expect(screen.getByTestId("splash-overlay")).toBeInTheDocument()
    })
  })

  it("suspends child rendering when hidden and quick ingest is not open", async () => {
    render(
      <AppShell
        router={Router}
        direction="ltr"
        emptyDescription="No data"
        suspendWhenHidden>
        <div data-testid="app-content">App content</div>
      </AppShell>
    )

    expect(screen.getByTestId("app-content")).toBeInTheDocument()

    setDocumentVisibility("hidden")
    await waitFor(() => {
      expect(screen.queryByTestId("app-content")).not.toBeInTheDocument()
    })

    setDocumentVisibility("visible")
    await waitFor(() => {
      expect(screen.getByTestId("app-content")).toBeInTheDocument()
    })
  })

  it("keeps child rendering when hidden if quick ingest modal is open", async () => {
    render(
      <AppShell
        router={Router}
        direction="ltr"
        emptyDescription="No data"
        suspendWhenHidden>
        <div data-testid="app-content">App content</div>
        <div className="quick-ingest-modal">
          <div className="ant-modal-content">Quick ingest modal</div>
        </div>
      </AppShell>
    )

    expect(screen.getByTestId("app-content")).toBeInTheDocument()

    setDocumentVisibility("hidden")
    await waitFor(() => {
      expect(screen.getByTestId("app-content")).toBeInTheDocument()
    })
  })
})

import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter, Route, Routes } from "react-router-dom"

const runtimeListeners = new Set<
  (message: {
    from?: string
    type?: string
    text?: string
    payload?: unknown
  }) => void
>()

const mocks = vi.hoisted(() => ({
  ensureSidepanelOpen: vi.fn(),
  sendMessage: vi.fn(),
  notify: vi.fn()
}))

vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineBackground", {
    configurable: true,
    value: (options: unknown) => options
  })
  return {}
})

vi.mock("@/services/background-helpers", () => ({
  ensureSidepanelOpen: (...args: unknown[]) =>
    mocks.ensureSidepanelOpen(...args),
  notify: (...args: unknown[]) => mocks.notify(...args)
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      sendMessage: (...args: unknown[]) => mocks.sendMessage(...args),
      onMessage: {
        addListener: (listener: (message: unknown) => void) => {
          runtimeListeners.add(listener as never)
        },
        removeListener: (listener: (message: unknown) => void) => {
          runtimeListeners.delete(listener as never)
        }
      }
    },
    i18n: {
      getMessage: (key: string) =>
        ({
          contextSaveToClipper: "Save to Clipper",
          contextSaveToClipperRestrictedPage:
            "This page is restricted, so the clipper cannot capture it."
        } as Record<string, string>)[key] || key
    }
  }
}))

vi.mock("~/hooks/useDarkmode", () => ({
  useDarkMode: () => ({ mode: "light" })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    i18n: {
      language: "en",
      resolvedLanguage: "en"
    }
  })
}))

vi.mock("@/components/Common/PageAssistLoader", () => ({
  PageAssistLoader: () =>
    React.createElement("div", { "data-testid": "route-loader" }, "Loading")
}))

vi.mock("@/hooks/useAutoButtonTitles", () => ({
  useAutoButtonTitles: () => {}
}))

vi.mock("@/i18n", () => ({
  ensureI18nNamespaces: vi.fn().mockResolvedValue(undefined)
}))

vi.mock("@/utils/ui-diagnostics", () => ({
  registerUiDiagnostics: vi.fn()
}))

vi.mock("@/store/layout-ui", () => ({
  useLayoutUiStore: (
    selector: (state: { setChatSidebarCollapsed: () => void }) => unknown
  ) => selector({ setChatSidebarCollapsed: () => {} })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: null,
    loading: false
  })
}))

vi.mock("@/config/platform", () => ({
  platformConfig: { target: "browser" }
}))

vi.mock("@/routes/route-capabilities", () => ({
  isRouteEnabledForCapabilities: () => true
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    setSetting: vi.fn().mockResolvedValue(undefined)
  }
})

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    React.createElement("div", { "data-testid": "option-layout" }, children)
  )
}))

import { RouteShell } from "@/routes/app-route"
import { buildClipDraft } from "@/services/web-clipper/draft-builder"
import { CLIPPER_CAPTURE_MESSAGE_TYPE } from "@/services/web-clipper/pending-draft"
import { launchWebClipperFromContextMenu } from "@/entries/background"
import type { RouteDefinition } from "@/routes/route-registry"

const ROUTES: Record<"options" | "sidepanel", RouteDefinition[]> = {
  options: [
    {
      kind: "options",
      path: "/",
      element: React.createElement("div", { "data-testid": "home-route" }, "Home")
    }
  ],
  sidepanel: [
    {
      kind: "sidepanel",
      path: "/chat",
      element: React.createElement(
        "div",
        { "data-testid": "sidepanel-chat" },
        "Chat"
      )
    },
    {
      kind: "sidepanel",
      path: "/clipper",
      element: React.createElement(
        "div",
        { "data-testid": "sidepanel-clipper" },
        "Clipper"
      )
    }
  ]
}

const renderRouteShell = (kind: "options" | "sidepanel", path: string) =>
  render(
    React.createElement(
      MemoryRouter,
      { initialEntries: [path] },
      React.createElement(
        Routes,
        null,
        React.createElement(Route, {
          path: "*",
          element: React.createElement(RouteShell, {
            kind,
            routes: ROUTES[kind]
          })
        })
      )
    )
  )

describe("web clipper background launcher", () => {
  beforeEach(() => {
    runtimeListeners.clear()
    window.sessionStorage.clear()
    vi.clearAllMocks()
    vi.useRealTimers()
    mocks.sendMessage.mockResolvedValue({ handled: true })

    Object.defineProperty(globalThis, "browser", {
      configurable: true,
      value: {
        runtime: {
          onMessage: {
            addListener: vi.fn((listener: (message: unknown) => void) => {
              runtimeListeners.add(listener)
            }),
            removeListener: vi.fn((listener: (message: unknown) => void) => {
              runtimeListeners.delete(listener)
            })
          }
        }
      }
    })
  })

  afterEach(() => {
    Reflect.deleteProperty(globalThis, "browser")
    Reflect.deleteProperty(globalThis, "defineBackground")
  })

  it("routes the clipper handoff into the dedicated clipper sidepanel route", async () => {
    renderRouteShell("sidepanel", "/chat")

    await waitFor(() => {
      expect(screen.getByTestId("sidepanel-chat")).toBeVisible()
    })

    const draft = buildClipDraft({
      requestedType: "article",
      pageUrl: "https://example.com/story",
      pageTitle: "Story",
      extracted: {
        articleText: "",
        fullPageText: "Fallback body"
      }
    })

    act(() => {
      for (const listener of runtimeListeners) {
        listener({
          from: "background",
          type: CLIPPER_CAPTURE_MESSAGE_TYPE,
          payload: draft
        })
      }
    })

    await waitFor(() => {
      expect(screen.getByTestId("sidepanel-clipper")).toBeVisible()
    })

    const stored = window.sessionStorage.getItem(
      "tldw:web-clipper:pendingDraft"
    )
    expect(stored).not.toBeNull()
    expect(JSON.parse(String(stored))).toMatchObject({
      clipType: "article",
      pageUrl: "https://example.com/story",
      pageTitle: "Story",
      captureMetadata: {
        fallbackPath: ["article", "full_page"]
      }
    })
  })

  it("opens the sidepanel and sends a clipper message instead of the notes flow", async () => {
    await launchWebClipperFromContextMenu(
      {
        pageUrl: "https://example.com/story",
        pageTitle: "Story",
        selectionText: "Selected excerpt"
      },
      { id: 8, url: "https://example.com/story", title: "Story" }
    )

    expect(mocks.ensureSidepanelOpen).toHaveBeenCalledWith(8)
    expect(mocks.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        from: "background",
        type: CLIPPER_CAPTURE_MESSAGE_TYPE,
        payload: expect.objectContaining({
          clipType: "selection",
          captureMetadata: expect.objectContaining({
            fallbackPath: ["selection"]
          })
        })
      })
    )
    expect(mocks.sendMessage).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "save-to-notes" })
    )
  })

  it("retries clipper delivery until the sidepanel listener is ready", async () => {
    vi.useFakeTimers()
    mocks.sendMessage
      .mockRejectedValueOnce(
        new Error("Could not establish connection. Receiving end does not exist.")
      )
      .mockRejectedValueOnce(
        new Error("Could not establish connection. Receiving end does not exist.")
      )
      .mockResolvedValueOnce({ handled: true })

    const handoffPromise = launchWebClipperFromContextMenu(
      {
        pageUrl: "https://example.com/story",
        selectionText: "Selected excerpt"
      },
      { id: 9, url: "https://example.com/story", title: "Story" }
    )

    await Promise.resolve()

    expect(mocks.sendMessage).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(500)
    expect(mocks.sendMessage).toHaveBeenCalledTimes(2)

    await vi.advanceTimersByTimeAsync(500)
    await handoffPromise

    expect(mocks.sendMessage).toHaveBeenCalledTimes(3)
    expect(mocks.notify).not.toHaveBeenCalled()
  })

  it("fails restricted pages with a user-visible explanation instead of sending a silent message", async () => {
    await launchWebClipperFromContextMenu(
      {
        pageUrl: "chrome://extensions",
        pageTitle: "Extensions"
      },
      { id: 14, url: "chrome://extensions", title: "Extensions" }
    )

    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(mocks.notify).toHaveBeenCalledWith(
      expect.stringContaining("Clipper"),
      expect.stringContaining("restricted")
    )
  })
})

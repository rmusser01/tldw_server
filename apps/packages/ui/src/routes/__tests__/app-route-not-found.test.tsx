import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { RouteShell } from "../app-route"
import type { RouteDefinition } from "../route-registry"

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
  PageAssistLoader: () => <div data-testid="route-loader">Loading</div>
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
  useLayoutUiStore: (selector: (state: { setChatSidebarCollapsed: () => void }) => unknown) =>
    selector({ setChatSidebarCollapsed: () => {} })
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
  default: ({
    children,
    hideHeader
  }: {
    children: React.ReactNode
    hideHeader?: boolean
  }) => (
    <div data-testid="option-layout" data-hide-header={String(Boolean(hideHeader))}>
      {children}
    </div>
  )
}))

const ROUTES: Record<"options" | "sidepanel", RouteDefinition[]> = {
  options: [
    { kind: "options", path: "/", element: <div data-testid="home-route">Home</div> },
    {
      kind: "options",
      path: "/research",
      element: <div data-testid="research-route">Research</div>
    },
    {
      kind: "options",
      path: "/knowledge",
      element: <div data-testid="knowledge-route">Knowledge</div>
    },
    { kind: "options", path: "/media", element: <div data-testid="media-route">Media</div> },
    {
      kind: "options",
      path: "/settings",
      element: <div data-testid="settings-route">Settings</div>
    }
  ],
  sidepanel: [
    {
      kind: "sidepanel",
      path: "/chat",
      element: <div data-testid="sidepanel-chat">Chat</div>
    }
  ]
}

const renderRouteShell = (
  kind: "options" | "sidepanel",
  path: string,
  renderUnmatchedRoute?: Parameters<typeof RouteShell>[0]["renderUnmatchedRoute"]
) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route
          path="*"
          element={
            <RouteShell
              kind={kind}
              routes={ROUTES[kind]}
              renderUnmatchedRoute={renderUnmatchedRoute}
            />
          }
        />
      </Routes>
    </MemoryRouter>
  )

describe("RouteShell unknown-route recovery", () => {
  it("starts settings routes with the root header hidden", async () => {
    renderRouteShell("options", "/settings")

    expect(await screen.findByTestId("settings-route")).toBeVisible()
    expect(screen.getByTestId("option-layout")).toHaveAttribute(
      "data-hide-header",
      "true"
    )
  })

  it("renders a not-found recovery panel instead of blank content", async () => {
    renderRouteShell("options", "/missing-route?foo=bar")

    expect(
      await screen.findByRole("heading", { name: "We could not find that route" })
    ).toBeVisible()
    expect(screen.getByTestId("not-found-recovery-panel")).toBeVisible()
    expect(screen.getByText("/missing-route?foo=bar")).toBeVisible()
    expect(screen.getByTestId("not-found-open-research")).toBeVisible()
  })

  it("supports primary recovery navigation from the fallback panel", async () => {
    const user = userEvent.setup()
    renderRouteShell("options", "/missing-route")

    expect(
      await screen.findByRole("link", { name: "Go to Home" })
    ).toHaveAttribute("href", "/")

    await user.click(screen.getByTestId("not-found-go-home"))
    expect(screen.getByTestId("home-route")).toBeVisible()
  })

  it("lets callers override the unmatched route renderer for deferred route loading", async () => {
    renderRouteShell(
      "options",
      "/missing-route",
      ({ attemptedRoute }) => (
        <div data-testid="deferred-unmatched-route">{attemptedRoute}</div>
      )
    )

    expect(await screen.findByTestId("deferred-unmatched-route")).toHaveTextContent(
      "/missing-route"
    )
    expect(screen.queryByTestId("not-found-recovery-panel")).not.toBeInTheDocument()
  })
})

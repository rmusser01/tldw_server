import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

vi.mock("@/components/Common/PageAssistLoader", () => ({
  PageAssistLoader: ({ label }: { label: string }) => (
    <div data-testid="route-loader">{label}</div>
  )
}))

vi.mock("@/config/platform", () => ({
  platformConfig: { target: "browser" }
}))

vi.mock("@/routes/route-capabilities", () => ({
  isRouteEnabledForCapabilities: () => true
}))

vi.mock("../app-route", () => ({
  RouteNotFoundState: ({ routeLabel }: { routeLabel: string }) => (
    <div data-testid="route-not-found">{routeLabel}</div>
  )
}))

vi.mock("../route-registry", () => ({
  optionRoutes: [
    {
      kind: "options",
      path: "/knowledge",
      element: <div data-testid="deferred-knowledge-route">Knowledge</div>
    }
  ]
}))

vi.mock("../option-settings-route-registry", () => ({
  optionSettingsRoutes: [
    {
      kind: "options",
      path: "/settings/chat",
      element: <div data-testid="deferred-settings-route">Settings Chat</div>
    },
    {
      kind: "options",
      path: "/settings/provider-keys",
      element: <div data-testid="deferred-provider-keys-route">Provider Keys</div>
    }
  ]
}))

vi.mock("../option-chat-route-registry", () => ({
  optionChatRoutes: [
    {
      kind: "options",
      path: "/chat",
      element: <div data-testid="deferred-chat-route">Chat</div>
    }
  ]
}))

vi.mock("../option-media-view-route-registry", () => ({
  optionMediaViewRoutes: [
    {
      kind: "options",
      path: "/media",
      element: <div data-testid="deferred-media-view-route">Media</div>
    }
  ]
}))

vi.mock("../option-media-review-route-registry", () => ({
  optionMediaReviewRoutes: [
    {
      kind: "options",
      path: "/media-multi",
      element: <div data-testid="deferred-media-review-route">Media Multi</div>
    }
  ]
}))

import { DeferredOptionsRoute } from "../deferred-options-route"

describe("DeferredOptionsRoute", () => {
  it("resolves chat deep links through the dedicated chat registry", async () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <DeferredOptionsRoute
          attemptedRoute="/chat"
          capabilities={null}
          capabilitiesLoading={false}
          label="Loading options..."
          description="Preparing routes"
        />
      </MemoryRouter>
    )

    expect(screen.getByTestId("route-loader")).toHaveTextContent("Loading options...")
    expect(await screen.findByTestId("deferred-chat-route")).toBeInTheDocument()
  })

  it("resolves media browsing deep links through the media-view registry", async () => {
    render(
      <MemoryRouter initialEntries={["/media"]}>
        <DeferredOptionsRoute
          attemptedRoute="/media"
          capabilities={null}
          capabilitiesLoading={false}
          label="Loading options..."
          description="Preparing routes"
        />
      </MemoryRouter>
    )

    expect(
      await screen.findByTestId("deferred-media-view-route")
    ).toBeInTheDocument()
  })

  it("resolves media review deep links through the media-review registry", async () => {
    render(
      <MemoryRouter initialEntries={["/media-multi"]}>
        <DeferredOptionsRoute
          attemptedRoute="/media-multi"
          capabilities={null}
          capabilitiesLoading={false}
          label="Loading options..."
          description="Preparing routes"
        />
      </MemoryRouter>
    )

    expect(
      await screen.findByTestId("deferred-media-review-route")
    ).toBeInTheDocument()
  })

  it("resolves settings deep links through the smaller settings registry", async () => {
    render(
      <MemoryRouter initialEntries={["/settings/chat"]}>
        <DeferredOptionsRoute
          attemptedRoute="/settings/chat"
          capabilities={null}
          capabilitiesLoading={false}
          label="Loading options..."
          description="Preparing routes"
        />
      </MemoryRouter>
    )

    expect(
      await screen.findByTestId("deferred-settings-route")
    ).toBeInTheDocument()
  })

  it("resolves provider key settings deep links through the smaller settings registry", async () => {
    render(
      <MemoryRouter initialEntries={["/settings/provider-keys"]}>
        <DeferredOptionsRoute
          attemptedRoute="/settings/provider-keys"
          capabilities={null}
          capabilitiesLoading={false}
          label="Loading options..."
          description="Preparing routes"
        />
      </MemoryRouter>
    )

    expect(
      await screen.findByTestId("deferred-provider-keys-route")
    ).toBeInTheDocument()
  })
})

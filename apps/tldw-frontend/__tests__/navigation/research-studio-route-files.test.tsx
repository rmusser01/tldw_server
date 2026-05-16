import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mockRouter = {
  asPath: "/workspace-playground?tab=studio#workspace-studio-panel",
  pathname: "/workspace-playground",
  replace: vi.fn(),
  prefetch: vi.fn(() => Promise.resolve(true))
}

vi.mock("next/router", () => ({
  useRouter: () => mockRouter
}))

vi.mock("next/dynamic", () => ({
  default: (_loader: unknown, options?: { ssr?: boolean }) =>
    function DynamicResearchStudioPage() {
      return (
        <div
          data-testid="research-studio-page"
          data-ssr={String(Boolean(options?.ssr))}
        />
      )
    }
}))

import ResearchStudioPage from "../../pages/research-studio"
import WorkspacePlaygroundRedirect from "../../pages/workspace-playground"
import WorkspaceStudioRedirect from "../../pages/workspace-studio"

describe("Research Studio Next route files", () => {
  beforeEach(() => {
    mockRouter.asPath = "/workspace-playground?tab=studio#workspace-studio-panel"
    mockRouter.pathname = "/workspace-playground"
    mockRouter.replace.mockReset()
    mockRouter.prefetch.mockClear()
    mockRouter.prefetch.mockResolvedValue(true)
  })

  it("provides /research-studio as the canonical page for the workspace surface", () => {
    render(<ResearchStudioPage />)

    expect(screen.getByTestId("research-studio-page")).toHaveAttribute(
      "data-ssr",
      "false"
    )
  })

  it("redirects /workspace-playground to /research-studio and preserves route state", async () => {
    render(<WorkspacePlaygroundRedirect />)

    expect(screen.getByTestId("route-redirect-panel")).toHaveTextContent(
      "Research Studio has moved"
    )
    await waitFor(() => {
      expect(mockRouter.replace).toHaveBeenCalledWith(
        "/research-studio?tab=studio#workspace-studio-panel"
      )
    })
  })

  it("redirects /workspace-studio to /research-studio and preserves route state", async () => {
    mockRouter.asPath = "/workspace-studio?tab=studio#workspace-studio-panel"
    mockRouter.pathname = "/workspace-studio"

    render(<WorkspaceStudioRedirect />)

    expect(screen.getByTestId("route-redirect-panel")).toHaveTextContent(
      "Research Studio has moved"
    )
    await waitFor(() => {
      expect(mockRouter.replace).toHaveBeenCalledWith(
        "/research-studio?tab=studio#workspace-studio-panel"
      )
    })
  })
})

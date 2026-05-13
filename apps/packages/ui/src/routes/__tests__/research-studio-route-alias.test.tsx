import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import {
  MemoryRouter,
  Route,
  Routes,
  useLocation
} from "react-router-dom"

import { RouteAliasNavigate } from "../RouteAliasNavigate"
import {
  RESEARCH_STUDIO_PATH,
  WORKSPACE_STUDIO_PATH
} from "../route-paths"
import { isHostedVisibleOptionPath } from "../option-route-visibility"

const LocationProbe = () => {
  const location = useLocation()
  return (
    <div data-testid="location">
      {`${location.pathname}${location.search}${location.hash}`}
    </div>
  )
}

describe("Research Studio shared route aliases", () => {
  it("defines /research-studio as the canonical route path", () => {
    expect(RESEARCH_STUDIO_PATH).toBe("/research-studio")
  })

  it("keeps the canonical Research Studio route visible in hosted mode", () => {
    expect(isHostedVisibleOptionPath(RESEARCH_STUDIO_PATH)).toBe(true)
  })

  it("uses a shared alias redirect that preserves search and hash state", async () => {
    render(
      <MemoryRouter
        initialEntries={[
          `${WORKSPACE_STUDIO_PATH}?tab=studio&shared=abc#workspace-studio-panel`
        ]}
      >
        <Routes>
          <Route
            path={WORKSPACE_STUDIO_PATH}
            element={<RouteAliasNavigate to={RESEARCH_STUDIO_PATH} />}
          />
          <Route path={RESEARCH_STUDIO_PATH} element={<LocationProbe />} />
        </Routes>
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.getByTestId("location")).toHaveTextContent(
        "/research-studio?tab=studio&shared=abc#workspace-studio-panel"
      )
    })
  })
})

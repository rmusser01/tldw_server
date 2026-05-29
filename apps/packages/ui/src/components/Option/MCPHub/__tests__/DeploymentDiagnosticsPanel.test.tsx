// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest"
import { render, screen } from "@testing-library/react"

import { useMcpToolsStore } from "@/store/mcp-tools"
import { DeploymentDiagnosticsPanel } from "../DeploymentDiagnosticsPanel"

describe("DeploymentDiagnosticsPanel", () => {
  afterEach(() => {
    useMcpToolsStore.setState({ healthState: "unknown" })
    delete (window as unknown as { __tldwServerReadinessState?: unknown })
      .__tldwServerReadinessState
  })

  it("summarizes quickstart same-origin diagnostics", () => {
    useMcpToolsStore.setState({ healthState: "healthy" })

    render(
      <DeploymentDiagnosticsPanel
        env={{ NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "quickstart" }}
        pageOrigin="http://localhost:3000"
      />
    )

    expect(screen.getByText("Deployment Diagnostics")).toBeTruthy()
    expect(screen.getByText("quickstart")).toBeTruthy()
    expect(screen.getByText("same-origin proxy")).toBeTruthy()
    expect(screen.getByText("http://localhost:3000")).toBeTruthy()
    expect(screen.getByText("relative (same origin)")).toBeTruthy()
    expect(screen.getByText("/api/v1/health")).toBeTruthy()
    expect(screen.getByText("healthy")).toBeTruthy()
  })

  it("summarizes advanced direct API diagnostics", () => {
    useMcpToolsStore.setState({ healthState: "unhealthy" })

    render(
      <DeploymentDiagnosticsPanel
        env={{
          NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "advanced",
          NEXT_PUBLIC_API_URL: "http://127.0.0.1:8000"
        }}
        pageOrigin="http://localhost:3000"
      />
    )

    expect(screen.getByText("advanced")).toBeTruthy()
    expect(screen.getByText("direct API")).toBeTruthy()
    expect(screen.getByText("http://localhost:3000")).toBeTruthy()
    expect(screen.getByText("http://127.0.0.1:8000")).toBeTruthy()
    expect(screen.getByText("http://127.0.0.1:8000/api/v1/health")).toBeTruthy()
    expect(screen.getByText("unhealthy")).toBeTruthy()
  })

  it("shows the last WebUI readiness health result when available", () => {
    ;(window as unknown as { __tldwServerReadinessState?: unknown })
      .__tldwServerReadinessState = {
      state: "degraded",
      degradedChecks: ["mcp"],
      healthUrl: "http://127.0.0.1:8000/api/v1/health",
      httpStatus: 206,
      healthStatus: "degraded",
      checkedAt: "2026-05-28T20:00:00.000Z"
    }
    useMcpToolsStore.setState({ healthState: "healthy" })

    render(
      <DeploymentDiagnosticsPanel
        env={{ NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "quickstart" }}
        pageOrigin="http://localhost:3000"
      />
    )

    expect(screen.getByText("Last health status")).toBeTruthy()
    expect(screen.getByText("degraded")).toBeTruthy()
    expect(screen.getByText("Last status code")).toBeTruthy()
    expect(screen.getByText("206")).toBeTruthy()
    expect(screen.getByText("Last checked")).toBeTruthy()
    expect(screen.getByText("2026-05-28T20:00:00.000Z")).toBeTruthy()
    expect(screen.getByText("http://127.0.0.1:8000/api/v1/health")).toBeTruthy()
  })
})

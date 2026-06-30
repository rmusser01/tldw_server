// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SkillsWorkspace } from "../SkillsWorkspace"

const mocks = vi.hoisted(() => ({
  capabilitiesState: {
    capabilities: { hasSkills: true },
    loading: false,
    refresh: vi.fn()
  },
  connectionState: {
    uxState: "connected_ok" as
      | "connected_ok"
      | "connected_degraded"
      | "testing"
      | "configuring_url"
      | "configuring_auth"
      | "error_auth"
      | "error_unreachable"
      | "unconfigured",
    hasCompletedFirstRun: true
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      return fallbackOrOptions?.defaultValue ?? _key
    }
  })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => mocks.capabilitiesState
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => mocks.connectionState
}))

vi.mock("../Manager", () => ({
  SkillsManager: () => <div data-testid="skills-manager">Skills manager</div>
}))

const renderWorkspace = () =>
  render(
    <MemoryRouter initialEntries={["/skills"]}>
      <SkillsWorkspace />
    </MemoryRouter>
  )

describe("SkillsWorkspace capability states", () => {
  beforeEach(() => {
    mocks.capabilitiesState = {
      capabilities: { hasSkills: true },
      loading: false,
      refresh: vi.fn()
    }
    mocks.connectionState = {
      uxState: "connected_ok",
      hasCompletedFirstRun: true
    }
  })

  it("names the capability loading state and withholds the manager", () => {
    mocks.capabilitiesState = {
      capabilities: null,
      loading: true,
      refresh: vi.fn()
    }

    renderWorkspace()

    expect(screen.getByRole("status")).toHaveTextContent(
      "Checking Skills API support"
    )
    expect(screen.queryByTestId("skills-manager")).not.toBeInTheDocument()
  })

  it("uses a shared recovery state with a refresh action when Skills are unsupported", () => {
    const refresh = vi.fn()
    mocks.capabilitiesState = {
      capabilities: { hasSkills: false },
      loading: false,
      refresh
    }

    renderWorkspace()

    expect(
      screen.getByRole("heading", {
        name: "Skills are not available on this server"
      })
    ).toBeInTheDocument()
    expect(screen.getByTestId("skills-capability-state")).toHaveAttribute(
      "data-ds-component",
      "RecoveryCallout"
    )
    expect(screen.getByLabelText("Diagnostics")).toHaveTextContent("/api/v1/skills")
    expect(screen.queryByTestId("skills-manager")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Refresh capabilities" }))
    expect(refresh).toHaveBeenCalledTimes(1)
  })

  it("keeps route-specific setup guidance when the server is not configured", () => {
    mocks.connectionState = {
      uxState: "unconfigured",
      hasCompletedFirstRun: false
    }

    renderWorkspace()

    expect(
      screen.getByRole("heading", {
        name: "Finish setup before using Skills."
      })
    ).toBeInTheDocument()
    expect(screen.queryByTestId("skills-manager")).not.toBeInTheDocument()
  })

  it("renders SkillsManager only after Skills support is known", () => {
    renderWorkspace()

    expect(screen.getByTestId("skills-manager")).toBeInTheDocument()
  })

  it("keeps Skills usable when the connected server is degraded", () => {
    mocks.connectionState = {
      uxState: "connected_degraded",
      hasCompletedFirstRun: true
    }

    renderWorkspace()

    expect(screen.getByTestId("skills-manager")).toBeInTheDocument()
  })

  it("uses Skills-specific recovery guidance when the server is unreachable", () => {
    mocks.connectionState = {
      uxState: "error_unreachable",
      hasCompletedFirstRun: true
    }

    renderWorkspace()

    expect(
      screen.getByRole("heading", {
        name: "Can't reach your tldw server right now."
      })
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "To use Skills, reconnect to your tldw server so skill definitions can be stored and executed."
      )
    ).toBeInTheDocument()
    expect(screen.queryByTestId("skills-manager")).not.toBeInTheDocument()
  })
})

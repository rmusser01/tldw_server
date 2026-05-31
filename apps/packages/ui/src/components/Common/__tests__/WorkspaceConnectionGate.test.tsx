import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { WorkspaceConnectionGate } from "../WorkspaceConnectionGate"

const navigateMock = vi.fn()
let currentLocation = {
  pathname: "/collections",
  search: "",
  hash: ""
}

let uxState:
  | "connected_ok"
  | "demo_mode"
  | "testing"
  | "configuring_auth"
  | "error_auth"
  | "error_unreachable"
  | "unconfigured" = "connected_ok"
let hasCompletedFirstRun = true

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock,
  useLocation: () => currentLocation
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState,
    hasCompletedFirstRun
  })
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({
    children,
    maxWidthClassName
  }: {
    children: React.ReactNode
    maxWidthClassName?: string
  }) => (
    <div data-testid="page-shell" data-max-width={maxWidthClassName}>
      {children}
    </div>
  )
}))

vi.mock("@/components/Common/FeatureEmptyState", () => ({
  __esModule: true,
  default: ({
    title,
    description,
    primaryActionLabel,
    onPrimaryAction,
    secondaryActionLabel,
    onSecondaryAction
  }: {
    title?: React.ReactNode
    description?: React.ReactNode
    primaryActionLabel?: React.ReactNode
    onPrimaryAction?: () => void
    secondaryActionLabel?: React.ReactNode
    onSecondaryAction?: () => void
  }) => (
    <div>
      <h2>{title}</h2>
      <p>{description}</p>
      {primaryActionLabel ? (
        <button type="button" onClick={onPrimaryAction}>
          {primaryActionLabel}
        </button>
      ) : null}
      {secondaryActionLabel ? (
        <button type="button" onClick={onSecondaryAction}>
          {secondaryActionLabel}
        </button>
      ) : null}
    </div>
  )
}))

describe("WorkspaceConnectionGate", () => {
  beforeEach(() => {
    uxState = "connected_ok"
    hasCompletedFirstRun = true
    currentLocation = {
      pathname: "/collections",
      search: "",
      hash: ""
    }
    navigateMock.mockReset()
  })

  it("shows auth recovery guidance instead of rendering children when credentials are missing", () => {
    uxState = "error_auth"

    render(
      <WorkspaceConnectionGate featureName="Collections">
        <div>Collections content</div>
      </WorkspaceConnectionGate>
    )

    expect(
      screen.getByText("Add your credentials before Collections can load data.")
    ).toBeInTheDocument()
    expect(screen.queryByText("Collections content")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Open Settings" }))
    expect(navigateMock).toHaveBeenCalledWith("/settings/tldw")
  })

  it("routes first-run users back to setup when configuration has not been completed", () => {
    uxState = "unconfigured"
    hasCompletedFirstRun = false

    render(
      <WorkspaceConnectionGate featureName="Collections">
        <div>Collections content</div>
      </WorkspaceConnectionGate>
    )

    expect(
      screen.getByText("Finish setup before using Collections.")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Finish Setup" }))
    expect(navigateMock).toHaveBeenCalledWith("/")
  })

  it("preserves character-chat route intent when first-run users start from Characters", () => {
    uxState = "unconfigured"
    hasCompletedFirstRun = false
    currentLocation = {
      pathname: "/characters",
      search: "?from=header-select&create=true",
      hash: ""
    }

    render(
      <WorkspaceConnectionGate featureName="Characters">
        <div>Characters content</div>
      </WorkspaceConnectionGate>
    )

    fireEvent.click(screen.getByRole("button", { name: "Finish Setup" }))
    expect(navigateMock).toHaveBeenCalledWith(
      "/?intent=character-chat&returnTo=%2Fcharacters%3Ffrom%3Dheader-select%26create%3Dtrue"
    )
  })

  it("surfaces unreachable server guidance with a diagnostics path", () => {
    uxState = "error_unreachable"

    render(
      <WorkspaceConnectionGate featureName="Collections">
        <div>Collections content</div>
      </WorkspaceConnectionGate>
    )

    expect(
      screen.getByText("Can't reach your tldw server right now.")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Health & diagnostics" }))
    expect(navigateMock).toHaveBeenCalledWith("/settings/health")
  })

  it("shows a startup loading state while the connection check is still running", () => {
    uxState = "testing"

    render(
      <WorkspaceConnectionGate featureName="Collections">
        <div>Collections content</div>
      </WorkspaceConnectionGate>
    )

    expect(screen.getByRole("status")).toHaveTextContent(
      "Checking server connection..."
    )
    expect(screen.queryByText("Collections content")).not.toBeInTheDocument()
  })

  it("preserves demo fallback content when demo mode is active", () => {
    uxState = "demo_mode"

    render(
      <WorkspaceConnectionGate
        featureName="Collections"
        renderDemo={() => <div>Demo collections</div>}
      >
        <div>Collections content</div>
      </WorkspaceConnectionGate>
    )

    expect(screen.getByText("Demo collections")).toBeInTheDocument()
    expect(screen.queryByText("Collections content")).not.toBeInTheDocument()
  })

  it("renders children when the connection is ready", () => {
    render(
      <WorkspaceConnectionGate featureName="Collections">
        <div>Collections content</div>
      </WorkspaceConnectionGate>
    )

    expect(screen.getByText("Collections content")).toBeInTheDocument()
  })
})

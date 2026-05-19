import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

import {
  BackendRecoveryUiProvider
} from "../BackendRecoveryUiContext"
import { RouteErrorBoundary } from "../RouteErrorBoundary"

const mockClassifyBackendUnreachableError = vi.fn()

vi.mock("@/services/backend-unreachable", () => ({
  classifyBackendUnreachableError: (...args: unknown[]) =>
    mockClassifyBackendUnreachableError(...args)
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn().mockResolvedValue(null)
  })
}))

vi.mock("../BackendUnavailableRecovery", () => ({
  __esModule: true,
  default: ({
    details
  }: {
    details?: { title?: React.ReactNode; message?: React.ReactNode }
  }) => (
    <div data-testid="backend-unavailable-recovery">
      <h1>{details?.title ?? "backend fallback"}</h1>
      <p>{details?.message ?? "backend message"}</p>
    </div>
  )
}))

const BACKEND_UNREACHABLE_RESULT = {
  kind: "backend_unreachable" as const,
  subtype: "connection_refused" as const,
  title: "Cannot connect to the API server",
  message: "Cannot reach the API server at the configured server URL.",
  fixHint: "Make sure the tldw server is running and reachable from this browser.",
  rawMessage: "Failed to fetch",
  diagnostics: {
    matchedPattern: "failed_to_fetch"
  }
}

const ThrowingChild = () => {
  throw new Error("route failure")
}

describe("RouteErrorBoundary backend recovery", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    mockClassifyBackendUnreachableError.mockReset()
    mockClassifyBackendUnreachableError.mockReturnValue({
      kind: "other",
      rawMessage: "route failure",
      diagnostics: {
        reason: "not_transport"
      }
    })
  })

  it("renders backend recovery when the WebUI provider enables it", async () => {
    mockClassifyBackendUnreachableError.mockReturnValue(
      BACKEND_UNREACHABLE_RESULT
    )

    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <BackendRecoveryUiProvider routeRecoveryEnabled>
          <RouteErrorBoundary routeId="chat" routeLabel="Chat">
            <ThrowingChild />
          </RouteErrorBoundary>
        </BackendRecoveryUiProvider>
      </MemoryRouter>
    )

    expect(
      await screen.findByTestId("backend-unavailable-recovery")
    ).toBeInTheDocument()
    expect(
      screen.queryByTestId("route-error-boundary-chat")
    ).not.toBeInTheDocument()
  })

  it("keeps the default generic fallback when backend recovery is not enabled", async () => {
    mockClassifyBackendUnreachableError.mockReturnValue(
      BACKEND_UNREACHABLE_RESULT
    )

    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <RouteErrorBoundary routeId="chat" routeLabel="Chat">
          <ThrowingChild />
        </RouteErrorBoundary>
      </MemoryRouter>
    )

    expect(
      await screen.findByTestId("route-error-boundary-chat")
    ).toBeInTheDocument()
    expect(
      screen.queryByTestId("backend-unavailable-recovery")
    ).not.toBeInTheDocument()
  })

  it("keeps the generic fallback for non-backend errors even when enabled", async () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <BackendRecoveryUiProvider routeRecoveryEnabled>
          <RouteErrorBoundary routeId="chat" routeLabel="Chat">
            <ThrowingChild />
          </RouteErrorBoundary>
        </BackendRecoveryUiProvider>
      </MemoryRouter>
    )

    expect(
      await screen.findByTestId("route-error-boundary-chat")
    ).toBeInTheDocument()
    expect(
      screen.queryByTestId("backend-unavailable-recovery")
    ).not.toBeInTheDocument()
  })
})

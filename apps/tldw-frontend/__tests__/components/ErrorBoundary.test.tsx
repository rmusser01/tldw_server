import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import ErrorBoundary from "@web/components/ErrorBoundary"

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

vi.mock("@/components/Common/BackendUnavailableRecovery", () => ({
  __esModule: true,
  default: ({
    details,
    onRetry,
    onReload,
    onOpenDiagnostics,
    onOpenSettings
  }: {
    details?: { title?: React.ReactNode; message?: React.ReactNode }
    onRetry: () => void
    onReload: () => void
    onOpenDiagnostics: () => void
    onOpenSettings: () => void
  }) => (
    <div data-testid="backend-unavailable-recovery">
      <h1>{details?.title ?? "fallback title"}</h1>
      <p>{details?.message ?? "fallback body"}</p>
      <button type="button" onClick={onRetry}>
        Try again
      </button>
      <button type="button" onClick={onReload}>
        Reload page
      </button>
      <button type="button" onClick={onOpenDiagnostics}>
        Open Health & diagnostics
      </button>
      <button type="button" onClick={onOpenSettings}>
        Open Settings
      </button>
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

const StableChild = () => <div data-testid="stable-child">Stable child</div>

const ThrowingChild = () => {
  throw new Error("render failure")
}

describe("ErrorBoundary", () => {
  beforeEach(() => {
    mockClassifyBackendUnreachableError.mockReset()
    mockClassifyBackendUnreachableError.mockReturnValue({
      kind: "other",
      rawMessage: "render failure",
      diagnostics: {
        reason: "not_transport"
      }
    })
    vi.spyOn(console, "error").mockImplementation(() => {})
  })

  it("renders the backend recovery screen for caught backend-unreachable errors", async () => {
    mockClassifyBackendUnreachableError.mockReturnValue(
      BACKEND_UNREACHABLE_RESULT
    )

    render(
      <ErrorBoundary>
        <ThrowingChild />
      </ErrorBoundary>
    )

    expect(
      await screen.findByTestId("backend-unavailable-recovery")
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", {
        name: "Cannot connect to the API server"
      })
    ).toBeInTheDocument()
  })

  it("renders the backend recovery screen for classified unhandled rejections", async () => {
    mockClassifyBackendUnreachableError.mockReturnValue(
      BACKEND_UNREACHABLE_RESULT
    )

    render(
      <ErrorBoundary>
        <StableChild />
      </ErrorBoundary>
    )

    const event = new Event("unhandledrejection", {
      cancelable: true
    }) as Event & { reason?: unknown }
    Object.defineProperty(event, "reason", {
      configurable: true,
      value: new Error("Failed to fetch")
    })

    window.dispatchEvent(event)

    expect(
      await screen.findByTestId("backend-unavailable-recovery")
    ).toBeInTheDocument()
    expect(event.defaultPrevented).toBe(true)
  })

  it("keeps the generic fallback for normal runtime errors", async () => {
    render(
      <ErrorBoundary>
        <ThrowingChild />
      </ErrorBoundary>
    )

    expect(await screen.findByTestId("error-boundary")).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: "Something went wrong" })
    ).toBeInTheDocument()
    expect(
      screen.queryByTestId("backend-unavailable-recovery")
    ).not.toBeInTheDocument()
  })

  it("clears backend recovery state when retry is pressed", async () => {
    mockClassifyBackendUnreachableError.mockReturnValue(
      BACKEND_UNREACHABLE_RESULT
    )
    const user = userEvent.setup()

    render(
      <ErrorBoundary>
        <StableChild />
      </ErrorBoundary>
    )

    const event = new Event("unhandledrejection", {
      cancelable: true
    }) as Event & { reason?: unknown }
    Object.defineProperty(event, "reason", {
      configurable: true,
      value: new Error("Failed to fetch")
    })
    window.dispatchEvent(event)

    expect(
      await screen.findByTestId("backend-unavailable-recovery")
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Try again" }))

    await waitFor(() => {
      expect(screen.getByTestId("stable-child")).toBeInTheDocument()
    })
    expect(
      screen.queryByTestId("backend-unavailable-recovery")
    ).not.toBeInTheDocument()
  })
})

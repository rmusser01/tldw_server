import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { getDesignSystemState } from "@/design-system"
import { ConnectionPhase, type ConnectionState } from "@/types/connection"
import { WorkspaceStatusBar } from "../WorkspaceStatusBar"

const registryLabels = vi.hoisted(() => ({
  degraded: "Registry Degraded"
}))

const connectionStoreState = {
  state: {
    phase: ConnectionPhase.CONNECTED,
    serverUrl: "http://127.0.0.1:8000",
    lastCheckedAt: null,
    lastError: null,
    lastStatusCode: null,
    isConnected: true,
    isChecking: false,
    consecutiveFailures: 0,
    knowledgeStatus: "ready" as const,
    knowledgeLastCheckedAt: null,
    knowledgeError: null,
    mode: "normal" as const,
    configStep: "none" as const,
    errorKind: "none" as const,
    hasCompletedFirstRun: true,
    userPersona: null,
    lastConfigUpdatedAt: null,
    checksSinceConfigChange: 0,
  } as ConnectionState,
  checkOnce: vi.fn(),
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    },
  }),
}))

vi.mock("@/store/connection", () => ({
  useConnectionStore: (
    selector: (state: typeof connectionStoreState) => unknown
  ) => selector(connectionStoreState),
}))

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)

        return {
          ...state,
          label: key === "degraded" ? registryLabels.degraded : state.label
        }
      }
    )
  }
})

describe("WorkspaceStatusBar", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    connectionStoreState.state = {
      phase: ConnectionPhase.CONNECTED,
      serverUrl: "http://127.0.0.1:8000",
      lastCheckedAt: null,
      lastError: null,
      lastStatusCode: null,
      isConnected: true,
      isChecking: false,
      consecutiveFailures: 0,
      knowledgeStatus: "ready",
      knowledgeLastCheckedAt: null,
      knowledgeError: null,
      mode: "normal",
      configStep: "none",
      errorKind: "none",
      hasCompletedFirstRun: true,
      userPersona: null,
      lastConfigUpdatedAt: null,
      checksSinceConfigChange: 0,
    }
  })

  it("shows retry for retriable connection errors", () => {
    connectionStoreState.state.phase = ConnectionPhase.ERROR
    connectionStoreState.state.isConnected = false
    connectionStoreState.state.errorKind = "unreachable"

    render(<WorkspaceStatusBar />)

    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
  })

  it("shows retry for authentication errors", () => {
    connectionStoreState.state.phase = ConnectionPhase.ERROR
    connectionStoreState.state.isConnected = false
    connectionStoreState.state.errorKind = "auth"

    render(<WorkspaceStatusBar />)

    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
  })

  it("does not show retry while the workspace is still being configured", () => {
    connectionStoreState.state.phase = ConnectionPhase.UNCONFIGURED
    connectionStoreState.state.isConnected = false
    connectionStoreState.state.configStep = "url"

    render(<WorkspaceStatusBar />)

    expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument()
  })

  it("uses the design-system registry label for degraded connection status", () => {
    connectionStoreState.state.errorKind = "partial"

    render(<WorkspaceStatusBar />)

    expect(screen.getByTestId("workspace-statusbar-connection")).toHaveTextContent(
      registryLabels.degraded
    )
    expect(vi.mocked(getDesignSystemState)).toHaveBeenCalledWith("degraded")
  })

  it("renders an actionable status details control when provided", () => {
    const onClick = vi.fn()

    render(
      <WorkspaceStatusBar
        statusMessages={["Legacy workspace data retained"]}
        statusAction={{
          label: "Details",
          ariaLabel: "Review migration recovery details",
          onClick
        }}
      />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Review migration recovery details" })
    )

    expect(onClick).toHaveBeenCalledTimes(1)
  })
})

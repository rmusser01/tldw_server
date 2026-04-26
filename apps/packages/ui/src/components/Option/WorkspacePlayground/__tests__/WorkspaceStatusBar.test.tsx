import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ConnectionPhase } from "@/types/connection"
import { WorkspaceStatusBar } from "../WorkspaceStatusBar"

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
  },
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
    connectionStoreState.state.errorKind = "error_auth"

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
})

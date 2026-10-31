import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { KnowledgeQASetupDiagnostics } from "@/components/Option/KnowledgeQA/SetupDiagnostics"
import { ConnectionPhase, deriveConnectionUxState } from "@/types/connection"

const fixture = vi.hoisted(() => ({
  config: { serverUrl: "https://offline.test", authMode: "single-user", apiKey: "synthetic-restored-key" },
  response: { ok: false, status: 0, error: "Failed to fetch" } as { ok: boolean; status: number; error: string | null },
  send: vi.fn()
}))
vi.mock("@/services/tldw-server-url", () => ({ getStoredTldwServerURL: async () => fixture.config.serverUrl }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: async () => undefined, getConfig: async () => fixture.config,
  ragHealth: async () => ({ status: "ok" }), updateConfig: async () => undefined
} }))
vi.mock("@/services/api-send", () => ({ apiSend: (...args: unknown[]) => fixture.send(...args) }))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false
}))
vi.mock("wxt/browser", () => ({ browser: {
  runtime: {}, storage: { local: { get: async () => ({}) }, onChanged: { addListener: () => undefined } }
} }))
import { useConnectionStore } from "@/store/connection"
const baseline = useConnectionStore.getState().state
const retry = vi.fn()
const renderCurrentDiagnostics = () => {
  const connection = useConnectionStore.getState().state
  return render(<KnowledgeQASetupDiagnostics connection={connection} uxState={deriveConnectionUxState(connection)}
    retryCountdownSeconds={8} onOpenSetup={vi.fn()} onOpenSettings={vi.fn()} onOpenDiagnostics={vi.fn()} onRetryConnection={retry} />)
}
beforeEach(() => {
  vi.clearAllMocks()
  fixture.config = { serverUrl: "https://offline.test", authMode: "single-user", apiKey: "synthetic-restored-key" }
  fixture.response = { ok: false, status: 0, error: "Failed to fetch" }
  fixture.send.mockImplementation(async () => fixture.response)
  useConnectionStore.setState({ state: { ...baseline, serverUrl: fixture.config.serverUrl,
    phase: ConnectionPhase.UNCONFIGURED, configStep: "auth", errorKind: "none", lastCheckedAt: null, isChecking: false, isConnected: false } })
})
afterEach(() => { vi.restoreAllMocks() })

describe("Knowledge diagnostics after actual connection state transitions", () => {
  it.each(["Failed to fetch", "NetworkError when attempting to fetch resource.", "The operation was aborted.", "Request timed out."])(
    "keeps restored credentials configured and a network failure uncertain: %s", async error => {
      fixture.response.error = error
      await useConnectionStore.getState().checkOnce({ force: true })
      const state = useConnectionStore.getState().state
      expect(state).toMatchObject({ configStep: "auth", errorKind: "unreachable" })
      expect.soft(state.lastError).toBe(error)
      renderCurrentDiagnostics()
      expect.soft(screen.getByTestId("knowledge-setup-check-credentials")).toHaveTextContent("Credentials are configured")
      expect.soft(screen.getByTestId("knowledge-setup-check-browser-access")).not.toHaveTextContent("Blocked")
      expect.soft(screen.getByTestId("knowledge-setup-check-backend")).not.toHaveTextContent("Waiting for credentials")
      const button = screen.queryByRole("button", { name: "Retry connection" })
      expect.soft(button).not.toBeNull()
      if (button) fireEvent.click(button)
      expect.soft(retry).toHaveBeenCalledTimes(1)
    }
  )
  it.each([401, 403])("keeps credential recovery for an actual auth rejection %s", async status => {
    fixture.response = { ok: false, status, error: "Invalid credentials" }
    await useConnectionStore.getState().checkOnce({ force: true })
    renderCurrentDiagnostics()
    expect(screen.getByTestId("knowledge-setup-check-credentials")).toHaveTextContent("Missing")
    expect(screen.getByRole("button", { name: "Update credentials" })).toBeVisible()
  })
  it("does not dispatch when credentials are missing", async () => {
    fixture.config.apiKey = ""
    await useConnectionStore.getState().checkOnce({ force: true })
    renderCurrentDiagnostics()
    expect(screen.getByTestId("knowledge-setup-check-credentials")).toHaveTextContent("Missing")
    expect(fixture.send).not.toHaveBeenCalled()
  })
  it("retains an explicit browser allowlist denial", async () => {
    fixture.response = { ok: false, status: 400, error: "Absolute URL requests are blocked unless the request origin is explicitly allowlisted." }
    await useConnectionStore.getState().checkOnce({ force: true })
    renderCurrentDiagnostics()
    expect(screen.getByTestId("knowledge-setup-check-browser-access")).toHaveTextContent("Blocked")
  })
  it("clears outage guidance when the configured server recovers", async () => {
    await useConnectionStore.getState().checkOnce({ force: true })
    fixture.response = { ok: true, status: 200, error: null }
    await useConnectionStore.getState().checkOnce({ force: true })
    renderCurrentDiagnostics()
    expect(useConnectionStore.getState().state).toMatchObject({ isConnected: true, lastError: null })
    expect(within(screen.getByTestId("knowledge-setup-check-backend")).getByText("Ready")).toBeVisible()
    expect(screen.getByTestId("knowledge-setup-check-credentials")).toHaveTextContent("Credentials are configured")
  })
})

// @vitest-environment jsdom
import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ConnectionPhase } from "@/types/connection"

// XS-11: "Retry now" runs the real connection check. Every check passes
// through the "testing" state, so the offline card (and the keyboard focus on
// its Retry button) has to survive that state instead of flipping to the
// setup copy and unmounting the button.

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      getURL: vi.fn((path: string) => `chrome-extension://test${path}`)
    },
    tabs: { create: vi.fn() },
    storage: {
      local: { set: vi.fn(async () => undefined) },
      onChanged: { addListener: vi.fn(), removeListener: vi.fn() }
    }
  }
}))

vi.mock("@/services/tldw-server-url", () => ({
  getStoredTldwServerURL: vi.fn(async () => null)
}))

vi.mock("@/services/api-send", () => ({
  apiSend: vi.fn()
}))

const client = vi.hoisted(() => ({
  getConfig: vi.fn(),
  initialize: vi.fn(),
  ragHealth: vi.fn(),
  updateConfig: vi.fn(),
  clearManualSingleUserCredentials: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: client }))

vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: vi.fn(() => null),
  isCookieSessionConfigInvalidated: vi.fn(() => false)
}))

import { useConnectionStore } from "@/store/connection"
import { EmptySidePanel } from "../empty"

const deferred = () => {
  let reject!: (error: Error) => void
  const promise = new Promise<void>((_resolve, rejectPromise) => {
    reject = rejectPromise
  })
  return { promise, reject }
}

const setUnreachable = () => {
  useConnectionStore.setState({
    state: {
      ...useConnectionStore.getState().state,
      phase: ConnectionPhase.ERROR,
      serverUrl: "http://127.0.0.1:8000",
      isConnected: false,
      isChecking: false,
      errorKind: "unreachable",
      lastError: "timeout-or-offline",
      lastCheckedAt: Date.now() - 60_000,
      consecutiveFailures: 3,
      hasCompletedFirstRun: true,
      mode: "normal",
      configStep: "none",
      offlineBypass: false
    }
  })
}

describe("EmptySidePanel retry (XS-11)", { timeout: 30_000 }, () => {
  beforeEach(() => {
    localStorage.removeItem("__tldw_allow_offline")
    localStorage.removeItem("__tldw_force_unconfigured")
    localStorage.setItem("__tldw_first_run_complete", "true")
    client.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    })
    setUnreachable()
  })

  afterEach(() => {
    localStorage.removeItem("__tldw_first_run_complete")
  })

  it("keeps the offline card and a busy Retry button while the retry runs, then returns focus", async () => {
    const check = deferred()
    client.initialize.mockReturnValueOnce(check.promise)
    const user = userEvent.setup()
    render(<EmptySidePanel />)

    const retry = screen.getByRole("button", { name: "Retry now" })
    await user.click(retry)
    expect(retry).toHaveFocus()

    // The real check is now in flight ("testing").
    await waitFor(() =>
      expect(useConnectionStore.getState().state.isChecking).toBe(true)
    )
    expect(screen.getByText("Can’t reach your tldw server")).toBeInTheDocument()
    expect(
      screen.queryByText("Connect to your server to start chatting")
    ).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry now" })).toBe(retry)
    expect(retry).toHaveAttribute("aria-busy", "true")
    expect(
      screen.getByRole("button", { name: "Open diagnostics" })
    ).toBeInTheDocument()
    // Chromium drops focus to <body> once a focused button is disabled; jsdom
    // keeps it (and ignores blur() on a disabled element), so move focus to a
    // throwaway element and remove it.
    act(() => {
      const sink = document.createElement("button")
      document.body.appendChild(sink)
      sink.focus()
      sink.remove()
    })
    expect(document.body).toHaveFocus()

    await act(async () => {
      check.reject(new Error("still down"))
      await check.promise.catch(() => undefined)
    })

    await waitFor(() =>
      expect(useConnectionStore.getState().state.phase).toBe(
        ConnectionPhase.ERROR
      )
    )
    await waitFor(() => expect(retry).not.toHaveAttribute("aria-busy", "true"))
    expect(screen.getByRole("button", { name: "Retry now" })).toBe(retry)
    expect(retry).toHaveFocus()
  })

  it("does not mark Retry busy or move focus during a background check", async () => {
    render(<EmptySidePanel />)
    const diagnostics = screen.getByRole("button", { name: "Open diagnostics" })
    diagnostics.focus()

    // The 5s background poll puts the store into "testing" on its own.
    act(() => {
      useConnectionStore.setState({
        state: {
          ...useConnectionStore.getState().state,
          phase: ConnectionPhase.SEARCHING,
          errorKind: "none",
          lastError: null,
          isChecking: true
        }
      })
    })

    expect(screen.getByText("Can’t reach your tldw server")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Open diagnostics" })).toBe(
      diagnostics
    )
    expect(diagnostics).toHaveFocus()
    expect(screen.getByRole("button", { name: "Retry now" })).not.toHaveAttribute(
      "aria-busy",
      "true"
    )

    act(() => setUnreachable())
    expect(diagnostics).toHaveFocus()
  })

  it("leaves the error card once the check reaches the server", async () => {
    render(<EmptySidePanel />)
    act(() => {
      useConnectionStore.setState({
        state: {
          ...useConnectionStore.getState().state,
          phase: ConnectionPhase.SEARCHING,
          errorKind: "none",
          isChecking: true
        }
      })
    })
    act(() => {
      useConnectionStore.setState({
        state: {
          ...useConnectionStore.getState().state,
          phase: ConnectionPhase.UNCONFIGURED,
          configStep: "url",
          isChecking: false
        }
      })
    })
    expect(
      screen.queryByText("Can’t reach your tldw server")
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Retry now" })
    ).not.toBeInTheDocument()
  })
})

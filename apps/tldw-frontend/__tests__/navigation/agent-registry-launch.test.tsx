import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useRouter } from "next/router"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AgentRegistryPage } from "@/components/Option/AgentRegistry"

const connection = vi.hoisted(() => ({
  config: { serverUrl: "http://127.0.0.1:8000", authMode: "single-user", apiKey: "fixture-key" },
  loading: false,
  authorityLoading: false
}))
const router = vi.hoisted(() => ({
  asPath: "/agents", pathname: "/agents", query: {} as Record<string, string>,
  listeners: new Set<() => void>(), push: vi.fn(), replace: vi.fn(), back: vi.fn()
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => connection
}))
vi.mock("react-router-dom", async () => await import("@web/extension/shims/react-router-dom"))
vi.mock("next/router", async () => {
  const react = await import("react")
  return {
    useRouter: () => {
      react.useSyncExternalStore((listener) => {
        router.listeners.add(listener)
        return () => { router.listeners.delete(listener) }
      }, () => router.asPath)
      return router
    }
  }
})

const Host = () => {
  const current = useRouter()
  return current.pathname === "/acp-playground"
    ? <section><h1>ACP destination</h1><output aria-label="Selected agent">{current.query.agent}</output></section>
    : <AgentRegistryPage />
}
const originalUrl = window.location.href
let agentType = "opencode"
let configured = true

describe("Agent Registry Launch through the production Next navigation shim", () => {
  beforeEach(() => {
    agentType = "opencode"
    configured = true
    router.asPath = "/agents"
    router.pathname = "/agents"
    router.query = {}
    window.history.replaceState(null, "", "/agents")
    router.push.mockImplementation(async (href: string) => {
      const target = new URL(href, window.location.origin)
      window.history.pushState(null, "", href)
      router.asPath = href
      router.pathname = target.pathname
      router.query = Object.fromEntries(target.searchParams)
      router.listeners.forEach((listener) => listener())
      return true
    })
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const path = new URL(String(input)).pathname
      if (path === "/api/v1/acp/agents") {
        return new Response(JSON.stringify({ agents: [{
          type: agentType, name: "Fixture Agent", description: "Owned read-only fixture",
          is_configured: configured, support_state: "supported", verification_level: "live_e2e"
        }] }), { status: 200 })
      }
      if (path === "/api/v1/acp/health") {
        return new Response(JSON.stringify({ runner: "ok", agent: "ok", api_keys: "ok" }), { status: 200 })
      }
      if (path === "/api/v1/admin/acp/execution-health/summary") {
        return new Response(JSON.stringify({ sessions: { total: 0, by_status: {} } }), { status: 200 })
      }
      throw new Error(`Unexpected fixture request: ${path}`)
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    window.history.replaceState(null, "", originalUrl)
  })

  it.each(["opencode", "codex"])("Launch leaves /agents and renders selected %s at the canonical Web destination", async (type) => {
    agentType = type
    render(<Host />)
    const launch = await screen.findByRole("button", { name: "Launch" })
    await waitFor(() => expect(launch).toBeEnabled())
    await userEvent.click(launch)
    expect(await screen.findByRole("heading", { name: "ACP destination" })).toBeVisible()
    expect(screen.getByLabelText("Selected agent")).toHaveTextContent(type)
    expect(window.location.pathname).toBe("/acp-playground")
    expect(window.location.search).toBe(`?agent=${type}`)
    expect(window.location.hash).toBe("")
    expect(screen.queryByRole("heading", { name: "Fixture Agent" })).not.toBeInTheDocument()
  })

  it("an unavailable agent cannot route away from the Web Registry", async () => {
    configured = false
    render(<Host />)
    const launch = await screen.findByRole("button", { name: "Launch" })
    expect(launch).toBeDisabled()
    await userEvent.click(launch)
    expect(window.location.pathname).toBe("/agents")
    expect(window.location.search).toBe("")
    expect(window.location.hash).toBe("")
    expect(screen.getByRole("heading", { name: "Fixture Agent" })).toBeVisible()
    expect(screen.queryByRole("heading", { name: "ACP destination" })).not.toBeInTheDocument()
  })
})

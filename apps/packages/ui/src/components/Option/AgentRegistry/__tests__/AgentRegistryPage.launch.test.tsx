import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { HashRouter, Route, Routes, useSearchParams } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AgentRegistryPage } from "../index"

const connection = vi.hoisted(() => ({
  config: { serverUrl: "http://127.0.0.1:8000", authMode: "single-user", apiKey: "fixture-key" },
  loading: false,
  authorityLoading: false
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => connection
}))

const originalUrl = window.location.href
let agentType = "opencode"
let configured = true

const Destination = () => {
  const [params] = useSearchParams()
  return <section><h1>ACP destination</h1><output aria-label="Selected agent">{params.get("agent")}</output></section>
}

describe("Agent Registry Launch in the extension hash host", () => {
  beforeEach(() => {
    agentType = "opencode"
    configured = true
    window.history.replaceState(null, "", "/options.html#/agents")
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

  it.each(["opencode", "codex"])("Launch renders the ACP hash destination with selected %s", async (type) => {
    agentType = type
    render(<HashRouter><Routes>
      <Route path="/agents" element={<AgentRegistryPage />} />
      <Route path="/acp-playground" element={<Destination />} />
    </Routes></HashRouter>)
    const launch = await screen.findByRole("button", { name: "Launch" })
    await waitFor(() => expect(launch).toBeEnabled())
    await userEvent.click(launch)
    expect(await screen.findByRole("heading", { name: "ACP destination" })).toBeVisible()
    expect(screen.getByLabelText("Selected agent")).toHaveTextContent(type)
    expect(window.location.pathname).toBe("/options.html")
    expect(window.location.hash).toBe(`#/acp-playground?agent=${type}`)
    expect(screen.queryByRole("heading", { name: "Fixture Agent" })).not.toBeInTheDocument()
  })

  it("an unavailable agent cannot leave Registry or select a destination", async () => {
    configured = false
    render(<HashRouter><Routes>
      <Route path="/agents" element={<AgentRegistryPage />} />
      <Route path="/acp-playground" element={<Destination />} />
    </Routes></HashRouter>)
    const launch = await screen.findByRole("button", { name: "Launch" })
    expect(launch).toBeDisabled()
    await userEvent.click(launch)
    expect(window.location.hash).toBe("#/agents")
    expect(screen.getByRole("heading", { name: "Fixture Agent" })).toBeVisible()
    expect(screen.queryByRole("heading", { name: "ACP destination" })).not.toBeInTheDocument()
  })
})

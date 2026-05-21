import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi, beforeEach } from "vitest"

const mocks = vi.hoisted(() => ({
  fetchWithAuth: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options?: { defaultValue?: string }) =>
      options?.defaultValue || _key
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    fetchWithAuth: mocks.fetchWithAuth
  }
}))

vi.mock("../McpToolPicker", () => ({
  McpToolPicker: ({
    value,
    onChange
  }: {
    value: string
    onChange: (value: string) => void
  }) => (
    <select
      data-testid="mock-mcp-tool-picker"
      value={value}
      onChange={(event) => onChange(event.target.value)}
    >
      <option value="">Select a tool</option>
      <option value="knowledge.search">knowledge.search</option>
      <option value="notes.search">notes.search</option>
    </select>
  )
}))

import { PoliciesPanel } from "../PoliciesPanel"
import { ScopesPanel } from "../ScopesPanel"

describe("Persona Garden scope and policy editors", () => {
  beforeEach(() => {
    mocks.fetchWithAuth.mockReset()
  })

  it("loads and saves selected persona scope rules through the authenticated endpoint", async () => {
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: unknown }) => {
      if (path === "/api/v1/persona/profiles/persona-1/scope-rules" && !init) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "persona-1",
            rules: [
              {
                rule_type: "media_tag",
                rule_value: "research",
                include: true
              }
            ]
          })
        })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/scope-rules" &&
        init?.method === "PUT"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "persona-1",
            replaced_count: 1,
            rules: (init.body as any).rules
          })
        })
      }
      return Promise.resolve({ ok: false, error: `unhandled ${path}`, json: async () => ({}) })
    })

    render(
      <ScopesPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Research Persona"
      />
    )

    const valueInput = await screen.findByTestId("persona-scope-rule-value-0")
    fireEvent.change(valueInput, { target: { value: "science" } })
    fireEvent.click(screen.getByTestId("persona-scope-rule-include-0"))
    fireEvent.click(screen.getByTestId("persona-scope-save-button"))

    await waitFor(() => {
      expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
        "/api/v1/persona/profiles/persona-1/scope-rules",
        {
          method: "PUT",
          body: {
            rules: [
              {
                rule_type: "media_tag",
                rule_value: "science",
                include: false
              }
            ]
          }
        }
      )
    })
    expect(await screen.findByText("Scope rules saved.")).toBeInTheDocument()
  })

  it("blocks saving a blank added scope rule and keeps the editor open", async () => {
    mocks.fetchWithAuth.mockResolvedValue({
      ok: true,
      json: async () => ({ persona_id: "persona-1", rules: [] })
    })

    render(
      <ScopesPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Research Persona"
      />
    )

    fireEvent.click(await screen.findByTestId("persona-scope-add-rule"))
    fireEvent.click(screen.getByTestId("persona-scope-save-button"))

    expect(await screen.findByText("Rule value is required.")).toBeInTheDocument()
    expect(mocks.fetchWithAuth).toHaveBeenCalledTimes(1)
  })

  it("disables scope saves while rules are loading", async () => {
    let resolveLoad:
      | ((value: { ok: boolean; json: () => Promise<{ rules: unknown[] }> }) => void)
      | undefined
    mocks.fetchWithAuth.mockReturnValue(
      new Promise((resolve) => {
        resolveLoad = resolve
      })
    )

    render(
      <ScopesPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Research Persona"
      />
    )

    expect(await screen.findByText("Loading...")).toBeInTheDocument()
    expect(screen.getByTestId("persona-scope-save-button")).toBeDisabled()
    fireEvent.click(screen.getByTestId("persona-scope-save-button"))
    expect(mocks.fetchWithAuth).toHaveBeenCalledTimes(1)

    resolveLoad?.({
      ok: true,
      json: async () => ({ rules: [] })
    })
    await screen.findByText("No scope rules yet.")
  })

  it("keeps existing scope rules visible while a new persona load is in flight", async () => {
    let resolveSecondLoad:
      | ((value: { ok: boolean; json: () => Promise<{ rules: unknown[] }> }) => void)
      | undefined
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path === "/api/v1/persona/profiles/persona-1/scope-rules") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            rules: [{ rule_type: "media_tag", rule_value: "research", include: true }]
          })
        })
      }
      if (path === "/api/v1/persona/profiles/persona-2/scope-rules") {
        return new Promise((resolve) => {
          resolveSecondLoad = resolve
        })
      }
      return Promise.resolve({ ok: false, error: `unhandled ${path}`, json: async () => ({}) })
    })

    const { rerender } = render(
      <ScopesPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Research Persona"
      />
    )

    expect(await screen.findByDisplayValue("research")).toBeInTheDocument()

    rerender(
      <ScopesPanel
        selectedPersonaId="persona-2"
        selectedPersonaName="Ops Persona"
      />
    )

    expect(await screen.findByText("Loading...")).toBeInTheDocument()
    expect(screen.getByDisplayValue("research")).toBeInTheDocument()
    expect(screen.getByTestId("persona-scope-save-button")).toBeDisabled()

    resolveSecondLoad?.({
      ok: true,
      json: async () => ({
        rules: [{ rule_type: "media_tag", rule_value: "operations", include: true }]
      })
    })
    expect(await screen.findByDisplayValue("operations")).toBeInTheDocument()
  })

  it("loads and saves selected persona policy rules through the authenticated endpoint", async () => {
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: unknown }) => {
      if (path === "/api/v1/persona/profiles/persona-1/policy-rules" && !init) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "persona-1",
            rules: [
              {
                rule_kind: "mcp_tool",
                rule_name: "knowledge.search",
                allowed: true,
                require_confirmation: false,
                max_calls_per_turn: 2
              }
            ]
          })
        })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/policy-rules" &&
        init?.method === "PUT"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "persona-1",
            replaced_count: 1,
            rules: (init.body as any).rules
          })
        })
      }
      return Promise.resolve({ ok: false, error: `unhandled ${path}`, json: async () => ({}) })
    })

    render(<PoliciesPanel selectedPersonaId="persona-1" hasPendingPlan={false} />)

    const picker = await screen.findByTestId("mock-mcp-tool-picker")
    fireEvent.change(picker, { target: { value: "notes.search" } })
    fireEvent.click(screen.getByTestId("persona-policy-rule-confirm-0"))
    fireEvent.change(screen.getByTestId("persona-policy-rule-max-calls-0"), {
      target: { value: "3" }
    })
    fireEvent.click(screen.getByTestId("persona-policy-save-button"))

    await waitFor(() => {
      expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
        "/api/v1/persona/profiles/persona-1/policy-rules",
        {
          method: "PUT",
          body: {
            rules: [
              {
                rule_kind: "mcp_tool",
                rule_name: "notes.search",
                allowed: true,
                require_confirmation: true,
                max_calls_per_turn: 3
              }
            ]
          }
        }
      )
    })
    expect(await screen.findByText("Policy rules saved.")).toBeInTheDocument()
  })

  it("reports policy save failures without dropping the current rules", async () => {
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string }) => {
      if (path === "/api/v1/persona/profiles/persona-1/policy-rules" && !init) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "persona-1",
            rules: [
              {
                rule_kind: "skill",
                rule_name: "summarize",
                allowed: true,
                require_confirmation: true,
                max_calls_per_turn: null
              }
            ]
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: "policy validation failed",
        json: async () => ({})
      })
    })

    render(<PoliciesPanel selectedPersonaId="persona-1" hasPendingPlan />)

    expect(await screen.findByDisplayValue("summarize")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("persona-policy-save-button"))

    expect(await screen.findByText("policy validation failed")).toBeInTheDocument()
    expect(screen.getByDisplayValue("summarize")).toBeInTheDocument()
    expect(screen.getByText("A pending tool plan is available on the Live Session tab.")).toBeInTheDocument()
  })

  it("keeps existing policy rules visible while a new persona load is in flight", async () => {
    let resolveSecondLoad:
      | ((value: { ok: boolean; json: () => Promise<{ rules: unknown[] }> }) => void)
      | undefined
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path === "/api/v1/persona/profiles/persona-1/policy-rules") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            rules: [
              {
                rule_kind: "skill",
                rule_name: "summarize",
                allowed: true,
                require_confirmation: false
              }
            ]
          })
        })
      }
      if (path === "/api/v1/persona/profiles/persona-2/policy-rules") {
        return new Promise((resolve) => {
          resolveSecondLoad = resolve
        })
      }
      return Promise.resolve({ ok: false, error: `unhandled ${path}`, json: async () => ({}) })
    })

    const { rerender } = render(
      <PoliciesPanel selectedPersonaId="persona-1" selectedPersonaName="Research Persona" />
    )

    expect(await screen.findByDisplayValue("summarize")).toBeInTheDocument()

    rerender(
      <PoliciesPanel selectedPersonaId="persona-2" selectedPersonaName="Ops Persona" />
    )

    expect(await screen.findByText("Loading...")).toBeInTheDocument()
    expect(screen.getByDisplayValue("summarize")).toBeInTheDocument()
    expect(screen.getByTestId("persona-policy-save-button")).toBeDisabled()

    resolveSecondLoad?.({
      ok: true,
      json: async () => ({
        rules: [
          {
            rule_kind: "skill",
            rule_name: "draft_report",
            allowed: true,
            require_confirmation: true
          }
        ]
      })
    })
    expect(await screen.findByDisplayValue("draft_report")).toBeInTheDocument()
  })

  it("uses MCP picker selections for mcp_tool policy rule names", async () => {
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: unknown }) => {
      if (path === "/api/v1/persona/profiles/persona-1/policy-rules" && !init) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "persona-1",
            rules: [
              {
                rule_kind: "mcp_tool",
                rule_name: "knowledge.search",
                allowed: true,
                require_confirmation: false,
                max_calls_per_turn: null
              }
            ]
          })
        })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/policy-rules" &&
        init?.method === "PUT"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "persona-1",
            replaced_count: 1,
            rules: (init.body as any).rules
          })
        })
      }
      return Promise.resolve({ ok: false, error: `unhandled ${path}`, json: async () => ({}) })
    })

    render(<PoliciesPanel selectedPersonaId="persona-1" hasPendingPlan={false} />)

    const picker = await screen.findByTestId("mock-mcp-tool-picker")
    fireEvent.change(picker, { target: { value: "notes.search" } })
    fireEvent.click(screen.getByTestId("persona-policy-save-button"))

    await waitFor(() => {
      expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
        "/api/v1/persona/profiles/persona-1/policy-rules",
        expect.objectContaining({
          method: "PUT",
          body: {
            rules: [
              expect.objectContaining({
                rule_kind: "mcp_tool",
                rule_name: "notes.search"
              })
            ]
          }
        })
      )
    })
  })

  it("shows persona catalog default tool and capability context when available", async () => {
    mocks.fetchWithAuth.mockResolvedValue({
      ok: true,
      json: async () => ({
        persona_id: "persona-1",
        rules: []
      })
    })

    render(
      <PoliciesPanel
        selectedPersonaId="persona-1"
        hasPendingPlan={false}
        personaCapabilities={["agentic", "mcp_tools_configured"]}
        personaDefaultTools={["knowledge.search", "notes.search"]}
      />
    )

    expect(await screen.findByText("knowledge.search")).toBeInTheDocument()
    expect(screen.getByText("notes.search")).toBeInTheDocument()
    expect(screen.getByText("mcp_tools_configured")).toBeInTheDocument()
  })
})

import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import type { ArchetypeTemplate } from "@/types/archetype"

vi.mock("../MCPModuleToggleGrid", () => ({
  MCPModuleToggleGrid: ({
    enabledModules
  }: {
    enabledModules: string[]
  }) => <div data-testid="enabled-modules">{enabledModules.join(",")}</div>
}))

vi.mock("../MCPExternalCatalog", () => ({
  MCPExternalCatalog: ({
    connectedServers,
    onConnect
  }: {
    connectedServers: string[]
    onConnect: (draft: {
      serverKey?: string | null
      name: string
      baseUrl: string
      authType: "none" | "bearer" | "api_key"
      secret: string
    }) => void
  }) => (
    <div data-testid="external-catalog">
      <div data-testid="connected-servers">{connectedServers.join(",")}</div>
      <button
        type="button"
        onClick={() =>
          onConnect({
            serverKey: "slack",
            name: "Slack",
            baseUrl: "https://slack.example.com/mcp",
            authType: "none",
            secret: ""
          })
        }
      >
        Mock connect server
      </button>
    </div>
  )
}))

vi.mock("../MCPAccessControlTier", () => ({
  MCPAccessControlTier: ({
    mode
  }: {
    mode: string
  }) => <div data-testid="confirmation-mode">{mode}</div>
}))

import { ToolsConnectionsStep } from "../ToolsConnectionsStep"

function buildArchetype(
  overrides: Partial<ArchetypeTemplate>
): ArchetypeTemplate {
  return {
    key: "research_assistant",
    label: "Research Assistant",
    tagline: "Research support",
    icon: "microscope",
    persona: {
      name: "Research Assistant",
      system_prompt: null,
      personality_traits: []
    },
    mcp_modules: {
      enabled: ["search", "notes"],
      disabled: []
    },
    suggested_external_servers: [],
    policy: {
      confirmation_mode: "always",
      tool_overrides: []
    },
    voice_defaults: {},
    scope_rules: [],
    buddy: {
      species: null,
      palette: null,
      silhouette: null
    },
    starter_commands: [],
    ...overrides
  }
}

describe("ToolsConnectionsStep", () => {
  it("re-seeds module and confirmation state when archetype defaults change", () => {
    const { rerender } = render(
      <ToolsConnectionsStep
        archetypeDefaults={buildArchetype({})}
        onContinue={vi.fn()}
        saving={false}
      />
    )

    expect(screen.getByTestId("enabled-modules")).toHaveTextContent("search,notes")
    expect(screen.getByTestId("confirmation-mode")).toHaveTextContent("always")

    rerender(
      <ToolsConnectionsStep
        archetypeDefaults={buildArchetype({
          key: "study_buddy",
          label: "Study Buddy",
          mcp_modules: {
            enabled: ["flashcards"],
            disabled: []
          },
          policy: {
            confirmation_mode: "never",
            tool_overrides: []
          }
        })}
        onContinue={vi.fn()}
        saving={false}
      />
    )

    expect(screen.getByTestId("enabled-modules")).toHaveTextContent("flashcards")
    expect(screen.getByTestId("confirmation-mode")).toHaveTextContent("never")
  })

  it("tracks connected catalog servers by stable server key", () => {
    render(
      <ToolsConnectionsStep
        archetypeDefaults={buildArchetype({})}
        onContinue={vi.fn()}
        saving={false}
      />
    )

    expect(screen.getByTestId("connected-servers")).toHaveTextContent("")

    fireEvent.click(screen.getByRole("button", { name: "Mock connect server" }))

    expect(screen.getByTestId("connected-servers")).toHaveTextContent("slack")
  })
})

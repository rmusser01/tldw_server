import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasPersona: true, hasPersonalization: true, hasCompanion: true },
    loading: false
  })
}))

vi.mock("@/services/companion", () => ({
  isCompanionConsentRequiredResponse: () => false,
  fetchCompanionConversationPrompts: vi.fn().mockResolvedValue({
    prompt_source_kind: "context",
    prompts: []
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: vi.fn(),
    fetchWithAuth: vi.fn()
  }
}))

vi.mock("@/services/persona-stream", () => ({
  buildPersonaWebSocketUrl: vi.fn(() => "ws://persona.test/api/v1/persona/stream")
}))

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
    }
  })
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    Button: ({
      children,
      onClick,
      disabled,
      ...rest
    }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
      <button type="button" onClick={onClick} disabled={disabled} {...rest}>
        {children}
      </button>
    ),
    Checkbox: ({
      children,
      checked,
      onChange
    }: {
      children?: React.ReactNode
      checked?: boolean
      onChange?: (event: { target: { checked: boolean } }) => void
    }) => (
      <label>
        <input
          type="checkbox"
          checked={checked}
          onChange={(event) =>
            onChange?.({ target: { checked: event.currentTarget.checked } })
          }
        />
        {children}
      </label>
    ),
    Input: {
      TextArea: ({
        value,
        onChange,
        placeholder
      }: {
        value?: string
        onChange?: (event: React.ChangeEvent<HTMLTextAreaElement>) => void
        placeholder?: string
      }) => (
        <textarea
          value={value ?? ""}
          onChange={onChange}
          placeholder={placeholder}
        />
      )
    },
    Select: ({
      value,
      onChange,
      options = [],
      placeholder
    }: {
      value?: string
      onChange?: (value: string) => void
      options?: Array<{ label: string; value: string }>
      placeholder?: string
    }) => (
      <select
        aria-label={placeholder || "select"}
        value={value}
        onChange={(event) => onChange?.(event.currentTarget.value)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    ),
    Tag: ({ children }: { children?: React.ReactNode }) => <span>{children}</span>,
    Typography: {
      Text: ({ children }: { children?: React.ReactNode }) => <span>{children}</span>
    }
  }
})

vi.mock("@/components/Common/FeatureEmptyState", () => ({
  default: ({
    title,
    description
  }: {
    title: string
    description?: string
  }) => (
    <div data-testid="feature-empty-state">
      <div>{title}</div>
      {description ? <div>{description}</div> : null}
    </div>
  )
}))

vi.mock("@/components/Option/MCPHub", () => ({
  PersonaPolicySummary: () => <div data-testid="persona-policy-summary" />
}))

vi.mock("@/components/PersonaGarden/LiveSessionPanel", () => ({
  LiveSessionPanel: () => <div data-testid="live-session-panel" />
}))

vi.mock("@/components/PersonaGarden/CommandsPanel", () => ({
  CommandsPanel: ({
    isActive,
    openCommandId,
    rerunAfterSaveCommandId,
    onRerunAfterSave,
    draftCommandPhrase
  }: {
    isActive?: boolean
    openCommandId?: string | null
    rerunAfterSaveCommandId?: string | null
    draftCommandPhrase?: string
    onRerunAfterSave?: (commandId: string) => void
  }) => (
    <div data-testid="commands-panel">
      {isActive ? "active" : "inactive"}:{openCommandId || "none"}:
      {rerunAfterSaveCommandId || "none"}:{draftCommandPhrase || "none"}
      <button
        type="button"
        data-testid="commands-panel-rerun"
        onClick={() =>
          onRerunAfterSave?.(
            String(rerunAfterSaveCommandId || openCommandId || "").trim()
          )
        }
      >
        rerun
      </button>
    </div>
  )
}))

vi.mock("@/components/PersonaGarden/ConnectionsPanel", () => ({
  ConnectionsPanel: () => <div data-testid="connections-panel" />
}))

vi.mock("@/components/PersonaGarden/PersonaGardenTabs", () => ({
  PersonaGardenTabs: ({
    activeKey,
    items,
    onChange
  }: {
    activeKey: string
    items: Array<{ key: string; label: string; content: React.ReactNode }>
    onChange: (key: string) => void
  }) => (
    <div data-testid="persona-garden-tabs">
      <div data-testid="persona-garden-active-tab">{activeKey}</div>
      <div>
        {items.map((item) => (
          <button key={item.key} type="button" onClick={() => onChange(item.key)}>
            {item.key}
          </button>
        ))}
      </div>
      <div>{items.find((item) => item.key === activeKey)?.content}</div>
    </div>
  )
}))

vi.mock("@/components/PersonaGarden/PoliciesPanel", () => ({
  PoliciesPanel: () => <div data-testid="policies-panel" />
}))

vi.mock("@/components/PersonaGarden/ProfilePanel", () => ({
  ProfilePanel: () => <div data-testid="profile-panel" />
}))

vi.mock("@/components/PersonaGarden/ScopesPanel", () => ({
  ScopesPanel: () => <div data-testid="scopes-panel" />
}))

vi.mock("@/components/PersonaGarden/StateDocsPanel", () => ({
  StateDocsPanel: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="state-docs-panel">{children}</div>
  )
}))

vi.mock("@/components/PersonaGarden/TestLabPanel", () => ({
  TestLabPanel: ({
    isActive,
    onOpenCommand,
    onCreateCommandDraft,
    initialHeardText,
    rerunRequestToken
  }: {
    isActive?: boolean
    onOpenCommand?: (commandId: string, heardText: string) => void
    onCreateCommandDraft?: (heardText: string) => void
    initialHeardText?: string
    rerunRequestToken?: number
  }) => (
    <div data-testid="test-lab-panel">
      <div data-testid="test-lab-state">
        {isActive ? "active" : "inactive"}:{initialHeardText || "none"}:
        {String(rerunRequestToken ?? 0)}
      </div>
      <button
        type="button"
        data-testid="test-lab-open-command"
        onClick={() =>
          onOpenCommand?.("cmd-alert", "send alert for model drift")
        }
      >
        open command
      </button>
      <button
        type="button"
        data-testid="test-lab-success-open-command"
        onClick={() =>
          onOpenCommand?.("cmd-alert", String(initialHeardText || "send alert repaired"))
        }
      >
        open repaired command
      </button>
      <button
        type="button"
        data-testid="test-lab-create-command"
        onClick={() =>
          onCreateCommandDraft?.("start a focused research sprint")
        }
      >
        create command
      </button>
    </div>
  )
}))

vi.mock("@/components/PersonaGarden/VoiceExamplesPanel", () => ({
  VoiceExamplesPanel: () => <div data-testid="voice-examples-panel" />
}))

vi.mock("~/components/Sidepanel/Chat/SidepanelHeaderSimple", () => ({
  SidepanelHeaderSimple: ({ activeTitle }: { activeTitle?: string }) => (
    <div data-testid="sidepanel-header">{activeTitle || "header"}</div>
  )
}))

import SidepanelPersona from "../sidepanel-persona"

describe("SidepanelPersona command handoff", () => {
  it("switches from test lab to commands and back to rerun the repaired phrase", async () => {
    render(
      <MemoryRouter initialEntries={["/persona?tab=test-lab"]}>
        <SidepanelPersona />
      </MemoryRouter>
    )

    expect(screen.getByTestId("persona-garden-active-tab")).toHaveTextContent(
      "test-lab"
    )
    fireEvent.click(await screen.findByTestId("test-lab-open-command"))

    expect(screen.getByTestId("persona-garden-active-tab")).toHaveTextContent(
      "commands"
    )
    expect(await screen.findByTestId("commands-panel")).toHaveTextContent(
      "active:cmd-alert:cmd-alert:none"
    )

    fireEvent.click(screen.getByTestId("commands-panel-rerun"))

    expect(screen.getByTestId("persona-garden-active-tab")).toHaveTextContent(
      "test-lab"
    )
    expect(await screen.findByTestId("test-lab-state")).toHaveTextContent(
      "active:send alert for model drift:1"
    )

    fireEvent.click(screen.getByTestId("test-lab-success-open-command"))

    expect(screen.getByTestId("persona-garden-active-tab")).toHaveTextContent(
      "commands"
    )
    expect(await screen.findByTestId("commands-panel")).toHaveTextContent(
      "active:cmd-alert:cmd-alert:none"
    )
  })

  it("opens a new command draft from an unmatched test lab phrase", async () => {
    render(
      <MemoryRouter initialEntries={["/persona?tab=test-lab"]}>
        <SidepanelPersona />
      </MemoryRouter>
    )

    expect(screen.getByTestId("persona-garden-active-tab")).toHaveTextContent(
      "test-lab"
    )
    fireEvent.click(await screen.findByTestId("test-lab-create-command"))

    expect(screen.getByTestId("persona-garden-active-tab")).toHaveTextContent(
      "commands"
    )
    expect(await screen.findByTestId("commands-panel")).toHaveTextContent(
      "active:none:none:start a focused research sprint"
    )
  })
})

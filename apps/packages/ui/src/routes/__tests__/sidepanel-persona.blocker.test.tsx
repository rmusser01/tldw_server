import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import {
  MemoryRouter,
  UNSAFE_DataRouterContext,
  useLocation,
  useNavigate
} from "react-router-dom"

import { MemoryRouterWithFuture } from "@/entries/shared/router-utils"
import { usePersonaStateDocs } from "../hooks/usePersonaStateDocs"

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => false
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState: "connected_ok",
    hasCompletedFirstRun: true
  })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasPersona: true, hasPersonalization: true },
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
  CommandsPanel: () => <div data-testid="commands-panel" />
}))

vi.mock("@/components/PersonaGarden/ConnectionsPanel", () => ({
  ConnectionsPanel: () => <div data-testid="connections-panel" />
}))

vi.mock("@/components/PersonaGarden/PersonaGardenTabs", () => ({
  PersonaGardenTabs: () => <div data-testid="persona-garden-tabs" />
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
  TestLabPanel: () => <div data-testid="test-lab-panel" />
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

const PersonaRouteBlockerHarness: React.FC<{
  onRouter: (router: { navigate: (...args: any[]) => Promise<void> } | null) => void
}> = ({ onRouter }) => {
  const location = useLocation()
  const navigate = useNavigate()
  const dataRouter = React.useContext(UNSAFE_DataRouterContext)
  React.useEffect(() => {
    onRouter(dataRouter?.router ?? null)
    return () => onRouter(null)
  }, [dataRouter?.router, onRouter])
  const [, setError] = React.useState<string | null>(null)
  const stateDocs = usePersonaStateDocs({
    getTargetPersonaId: () => null,
    appendLog: () => undefined,
    setError
  })
  return (
    <div>
      <label htmlFor="persona-route-draft">Persona draft</label>
      <textarea
        id="persona-route-draft"
        value={stateDocs.soulMd}
        onChange={(event) => stateDocs.setSoulMd(event.currentTarget.value)}
      />
      <output aria-label="Current persona route">{location.pathname}</output>
      <button type="button" onClick={() => navigate("/next")}>Open next route</button>
      <button type="button" onClick={() => navigate(-1)}>Persona browser Back</button>
    </div>
  )
}

describe("SidepanelPersona route blocker integration", () => {
  it("renders the companion route under a standard MemoryRouter", () => {
    render(
      <MemoryRouter initialEntries={["/companion/conversation"]}>
        <SidepanelPersona mode="companion" />
      </MemoryRouter>
    )

    expect(screen.getByTestId("feature-empty-state")).toBeInTheDocument()
    expect(screen.getByText("Connect to use Companion")).toBeInTheDocument()
  })

  it("cancels a confirmed dirty numeric Back when its StrictMode owner retires before the next task", async () => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, "confirm")
    let blockerProceed: ReturnType<typeof vi.spyOn> | null = null
    const errors: unknown[] = []
    const onError = (event: ErrorEvent) => {
      errors.push(event.error)
      event.preventDefault()
    }
    const onUnhandled = (event: PromiseRejectionEvent) => {
      errors.push(event.reason)
      event.preventDefault()
    }
    window.addEventListener("error", onError)
    window.addEventListener("unhandledrejection", onUnhandled)

    try {
      let router: { navigate: (...args: any[]) => Promise<void> } | null = null
      const view = render(
        <React.StrictMode>
          <MemoryRouterWithFuture>
            <PersonaRouteBlockerHarness onRouter={(nextRouter) => { router = nextRouter }} />
          </MemoryRouterWithFuture>
        </React.StrictMode>
      )
      await waitFor(() => expect(router).not.toBeNull())
      await user.click(await screen.findByRole("button", { name: "Open next route" }))
      await waitFor(() =>
        expect(screen.getByRole("status", { name: "Current persona route" })).toHaveTextContent("/next")
      )
      await user.type(screen.getByLabelText("Persona draft"), "dirty persona draft")
      confirm.mockImplementationOnce(() => {
        const blockers = Array.from((router as any).state.blockers.values()) as Array<{
          proceed: () => void
        }>
        blockerProceed = vi.spyOn(blockers[0], "proceed")
        return true
      })

      fireEvent.click(screen.getByRole("button", { name: "Persona browser Back" }))
      expect(confirm).toHaveBeenCalledTimes(1)
      expect(blockerProceed).not.toBeNull()
      expect(blockerProceed).not.toHaveBeenCalled()
      expect(screen.getByRole("status", { name: "Current persona route" })).toHaveTextContent("/next")
      view.unmount()
      await new Promise((resolve) => window.setTimeout(resolve, 10))

      expect(blockerProceed).not.toHaveBeenCalled()
      expect(errors).toEqual([])
    } finally {
      window.removeEventListener("error", onError)
      window.removeEventListener("unhandledrejection", onUnhandled)
    }
  })
})

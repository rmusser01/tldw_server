import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { usePromptStudioStore } from "../../../../store/prompt-studio"
import {
  StudioTabContainer,
  getStudioStatusRefetchInterval
} from "../Studio/StudioTabContainer"

const state = vi.hoisted(() => ({
  isOnline: true,
  isMobile: false,
  processing: 0
}))

const useQueryMock = vi.hoisted(() => vi.fn())
const useMutationMock = vi.hoisted(() => vi.fn())
const invalidateQueriesMock = vi.hoisted(() => vi.fn())
const getConfigMock = vi.hoisted(() => vi.fn())
const setPromptStudioDefaultsMock = vi.hoisted(() => vi.fn())
const promptStudioServiceMocks = vi.hoisted(() => ({
  hasPromptStudio: vi.fn(),
  getPromptStudioStatus: vi.fn(),
  listProjects: vi.fn()
}))

let latestSocket: MockWebSocket | null = null
let settingsProjectsQueryOptions: any = null

class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3

  readyState = MockWebSocket.OPEN
  onopen: ((event: unknown) => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: ((event: unknown) => void) | null = null
  onclose: ((event: unknown) => void) | null = null
  sentMessages: string[] = []

  constructor(public url: string) {
    latestSocket = this
    queueMicrotask(() => this.onopen?.({}))
  }

  send(data: string) {
    this.sentMessages.push(data)
  }

  close() {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.({})
  }

  emitMessage(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) })
  }
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => state.isOnline
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => state.isMobile
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: (...args: unknown[]) =>
    (useQueryMock as (...args: unknown[]) => unknown)(...args),
  useMutation: (...args: unknown[]) =>
    (useMutationMock as (...args: unknown[]) => unknown)(...args),
  useQueryClient: () => ({
    invalidateQueries: (...args: unknown[]) =>
      (invalidateQueriesMock as (...args: unknown[]) => unknown)(...args),
    setQueryData: vi.fn()
  })
}))

vi.mock("antd", () => ({
  Segmented: ({ value, options, onChange, ...rest }: any) => (
    <div data-testid={rest["data-testid"] || "mock-segmented"}>
      <div data-testid="seg-value">{String(value)}</div>
      {options.map((option: any) => (
        <button
          key={String(option.value)}
          type="button"
          data-testid={`seg-option-${option.value}`}
          disabled={Boolean(option.disabled)}
          onClick={() => onChange?.(option.value)}
        >
          {option.label ?? String(option.value)}
        </button>
      ))}
    </div>
  ),
  Select: ({ value, options, onChange, ...rest }: any) => (
    <div data-testid={rest["data-testid"] || "mock-select"}>
      <div data-testid="select-value">{String(value)}</div>
      {options?.map((option: any) => (
        <button
          key={String(option.value)}
          type="button"
          data-testid={`select-option-${option.value}`}
          onClick={() => onChange?.(option.value)}
        >
          {String(option.label)}
        </button>
      ))}
    </div>
  ),
  Popover: ({ children, content }: any) => (
    <div>
      {children}
      <div data-testid="mock-popover-content">{content}</div>
    </div>
  ),
  Switch: ({ checked, onChange, ...rest }: any) => (
    <button
      type="button"
      data-testid={rest["data-testid"] || "mock-switch"}
      onClick={() => onChange?.(!checked)}
    >
      {checked ? "on" : "off"}
    </button>
  ),
  Badge: ({ count }: any) => <span data-testid="mock-badge">{count}</span>,
  Tooltip: ({ children }: any) => <>{children}</>,
  notification: {
    success: vi.fn(),
    error: vi.fn()
  }
}))

vi.mock("../Studio/QueueHealthWidget", () => ({
  QueueHealthWidget: () => <div data-testid="mock-queue-health-widget" />
}))

vi.mock("../Studio/Projects/ProjectsTab", () => ({
  ProjectsTab: () => <div data-testid="projects-tab-content" />
}))

vi.mock("../Studio/Prompts/StudioPromptsTab", () => ({
  StudioPromptsTab: () => <div data-testid="prompts-tab-content" />
}))

vi.mock("../Studio/TestCases/TestCasesTab", () => ({
  TestCasesTab: () => <div data-testid="test-cases-tab-content" />
}))

vi.mock("../Studio/Evaluations/EvaluationsTab", () => ({
  EvaluationsTab: () => <div data-testid="evaluations-tab-content" />
}))

vi.mock("../Studio/Optimizations/OptimizationsTab", () => ({
  OptimizationsTab: () => <div data-testid="optimizations-tab-content" />
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) =>
      (getConfigMock as (...args: unknown[]) => unknown)(...args)
  }
}))

vi.mock("@/services/prompt-studio", () => ({
  hasPromptStudio: (...args: unknown[]) =>
    (promptStudioServiceMocks.hasPromptStudio as (...args: unknown[]) => unknown)(
      ...args
    ),
  getPromptStudioStatus: (...args: unknown[]) =>
    (promptStudioServiceMocks.getPromptStudioStatus as (
      ...args: unknown[]
    ) => unknown)(...args),
  listProjects: (...args: unknown[]) =>
    (promptStudioServiceMocks.listProjects as (...args: unknown[]) => unknown)(
      ...args
    )
}))

vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: vi.fn(async () => ({
    defaultProjectId: null,
    autoSyncWorkspacePrompts: true
  })),
  setPromptStudioDefaults: (...args: unknown[]) =>
    (setPromptStudioDefaultsMock as (...args: unknown[]) => unknown)(...args)
}))

describe("StudioTabContainer stage 6 navigation and polling", () => {
  let statusQueryOptions: any

  beforeEach(() => {
    statusQueryOptions = null
    settingsProjectsQueryOptions = null
    state.isOnline = true
    state.isMobile = false
    state.processing = 0
    latestSocket = null
    ;(globalThis as any).WebSocket = MockWebSocket
    invalidateQueriesMock.mockReset()
    getConfigMock.mockReset()
    setPromptStudioDefaultsMock.mockReset()
    setPromptStudioDefaultsMock.mockImplementation(async (updates: any) => ({
      defaultProjectId:
        updates?.defaultProjectId !== undefined ? updates.defaultProjectId : null,
      autoSyncWorkspacePrompts:
        updates?.autoSyncWorkspacePrompts !== undefined
          ? updates.autoSyncWorkspacePrompts
          : true
    }))
    getConfigMock.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-api-key",
      accessToken: ""
    })
    promptStudioServiceMocks.hasPromptStudio.mockReset()
    promptStudioServiceMocks.getPromptStudioStatus.mockReset()
    promptStudioServiceMocks.listProjects.mockReset()
    promptStudioServiceMocks.hasPromptStudio.mockResolvedValue(true)
    promptStudioServiceMocks.getPromptStudioStatus.mockResolvedValue({
      data: {
        data: {
          queue_depth: 0,
          processing: 0,
          leases: {},
          success_rate: 1
        }
      }
    })
    promptStudioServiceMocks.listProjects.mockResolvedValue({
      data: [{ id: 11, name: "Project Eleven" }]
    })
    try { window.localStorage.removeItem("tldw-studio-onboarding-dismissed") } catch {}
    usePromptStudioStore.setState({
      activeSubTab: "projects",
      selectedProjectId: null
    })
    useQueryMock.mockReset()
    useMutationMock.mockReset()
    useMutationMock.mockImplementation((options: any) => ({
      mutate: async (variables: any) => {
        try {
          const result = await options?.mutationFn?.(variables)
          options?.onSuccess?.(result, variables, undefined)
          return result
        } catch (error) {
          options?.onError?.(error, variables, undefined)
          throw error
        }
      },
      isPending: false
    }))
    useQueryMock.mockImplementation((options: any) => {
      const key = String(options?.queryKey?.[1] || "")
      if (key === "capability") {
        return { data: true, isLoading: false }
      }
      if (key === "status") {
        statusQueryOptions = options
        return {
          data: {
            data: {
              data: {
                queue_depth: 0,
                processing: state.processing,
                leases: {},
                success_rate: 1
              }
            }
          }
        }
      }
      if (key === "settings-defaults") {
        return {
          data: {
            defaultProjectId: null,
            autoSyncWorkspacePrompts: true
          },
          isLoading: false
        }
      }
      if (key === "settings-projects") {
        settingsProjectsQueryOptions = options
        return {
          data: {
            data: [
              { id: 11, name: "Project Eleven" },
              { id: 12, name: "Project Twelve" }
            ]
          },
          isLoading: false
        }
      }
      return { data: undefined, isLoading: false }
    })
  })

  it("shows onboarding card and keeps project-required tabs disabled", () => {
    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    expect(screen.getByTestId("studio-onboarding-card")).toBeInTheDocument()
    expect(screen.getByText("Getting started with Prompt Studio")).toBeInTheDocument()
    expect(screen.getByTestId("seg-option-prompts")).toBeDisabled()
    expect(screen.getByTestId("seg-option-testCases")).toBeDisabled()
    expect(screen.getByTestId("seg-option-evaluations")).toBeDisabled()
    expect(screen.getByTestId("seg-option-optimizations")).toBeDisabled()
    expect(
      screen.getByLabelText("Prompts (Select a project first)")
    ).toBeInTheDocument()
  })

  it("shows warning banner after onboarding is dismissed", () => {
    try { window.localStorage.setItem("tldw-studio-onboarding-dismissed", "true") } catch {}

    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    expect(screen.queryByTestId("studio-onboarding-card")).not.toBeInTheDocument()
    expect(
      screen.getByText(
        "Select a project in the Projects tab to unlock Prompts, Test Cases, Evaluations, and Optimizations."
      )
    ).toBeInTheDocument()

    try { window.localStorage.removeItem("tldw-studio-onboarding-dismissed") } catch {}
  })

  it("dismisses onboarding card on dismiss button click", () => {
    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    expect(screen.getByTestId("studio-onboarding-card")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("studio-onboarding-dismiss"))
    expect(screen.queryByTestId("studio-onboarding-card")).not.toBeInTheDocument()
    expect(
      screen.getByText(
        "Select a project in the Projects tab to unlock Prompts, Test Cases, Evaluations, and Optimizations."
      )
    ).toBeInTheDocument()
    expect(
      window.localStorage.getItem("tldw-studio-onboarding-dismissed")
    ).toBe("true")

    try { window.localStorage.removeItem("tldw-studio-onboarding-dismissed") } catch {}
  })

  it("navigates to projects tab when step 1 is clicked in onboarding", () => {
    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    const step1 = screen.getByTestId("studio-onboarding-step-1")
    expect(step1).not.toBeDisabled()
    fireEvent.click(step1)
    expect(usePromptStudioStore.getState().activeSubTab).toBe("projects")
  })

  it("lets onboarding steps deep-link into gated tabs when no project is selected", () => {
    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    const steps: Array<[number, string]> = [
      [2, "prompts"],
      [3, "testCases"],
      [4, "evaluations"],
      [5, "optimizations"]
    ]

    for (const [step, tab] of steps) {
      const stepButton = screen.getByTestId(`studio-onboarding-step-${step}`)
      expect(stepButton).not.toBeDisabled()
      fireEvent.click(stepButton)
      expect(usePromptStudioStore.getState().activeSubTab).toBe(tab)
    }
  })

  it("guards against non-array studio project settings payloads", () => {
    const baseImpl = useQueryMock.getMockImplementation()
    useQueryMock.mockImplementation((options: any) => {
      const key = String(options?.queryKey?.[1] || "")
      if (key === "settings-projects") {
        return {
          data: {
            data: { error: "invalid_shape" }
          },
          isLoading: false
        }
      }
      return baseImpl?.(options)
    })

    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    expect(screen.getByTestId("select-option-none")).toHaveTextContent(
      "No default project"
    )
    expect(screen.queryByTestId("select-option-11")).not.toBeInTheDocument()
  })

  it("renders full-text mobile selector labels with project prerequisite hint", () => {
    state.isMobile = true

    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    expect(screen.getByTestId("studio-subtab-select-mobile")).toBeInTheDocument()
    expect(screen.getByTestId("select-option-projects")).toHaveTextContent("Projects")
    expect(screen.getByTestId("select-option-prompts")).toHaveTextContent(
      "Prompts (Select a project first)"
    )
  })

  it("uses adaptive refetch intervals for idle and active job states", () => {
    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    const getInterval = statusQueryOptions?.refetchInterval
    expect(typeof getInterval).toBe("function")
    expect(
      getInterval({
        state: { data: { data: { data: { processing: 0 } } } }
      })
    ).toBe(30000)
    expect(
      getInterval({
        state: { data: { data: { data: { processing: 2 } } } }
      })
    ).toBe(5000)

    expect(getStudioStatusRefetchInterval({ processing: 0 })).toBe(30000)
    expect(getStudioStatusRefetchInterval({ processing: 1 })).toBe(5000)
  })

  it("invalidates status query when realtime websocket receives job updates", async () => {
    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(latestSocket).not.toBeNull()
    })

    latestSocket?.emitMessage({ type: "job_progress", data: { progress: 50 } })

    await waitFor(() => {
      expect(invalidateQueriesMock).toHaveBeenCalledWith({
        queryKey: ["prompt-studio", "status"]
      })
    })
  })

  it("persists default project selection from studio settings", async () => {
    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    expect(screen.getByTestId("studio-settings-button")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("select-option-12"))

    await waitFor(() => {
      expect(setPromptStudioDefaultsMock).toHaveBeenCalledWith({
        defaultProjectId: 12
      })
    })
  })

  it("does not auto-select an unavailable default project", async () => {
    const baseImpl = useQueryMock.getMockImplementation()
    useQueryMock.mockImplementation((options: any) => {
      const key = String(options?.queryKey?.[1] || "")
      if (key === "settings-defaults") {
        return {
          data: {
            defaultProjectId: 99,
            autoSyncWorkspacePrompts: true
          },
          isLoading: false
        }
      }
      if (key === "settings-projects") {
        settingsProjectsQueryOptions = options
        return {
          data: {
            data: [{ id: 11, name: "Project Eleven" }]
          },
          isLoading: false
        }
      }
      return baseImpl?.(options)
    })

    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(usePromptStudioStore.getState().selectedProjectId).toBeNull()
    })
  })

  it("marks realtime as disconnected when websocket setup fails", async () => {
    getConfigMock.mockRejectedValueOnce(new Error("config failed"))

    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(
        screen.getByTestId("studio-ws-disconnected-banner")
      ).toBeInTheDocument()
    })
  })

  it("scopes Cmd/Ctrl+number tab shortcuts to the studio container", () => {
    usePromptStudioStore.setState({
      activeSubTab: "projects",
      selectedProjectId: 11
    })

    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    fireEvent.keyDown(document, { key: "2", metaKey: true })
    expect(usePromptStudioStore.getState().activeSubTab).toBe("projects")

    const container = screen.getByTestId("studio-tab-container")
    fireEvent.keyDown(container, { key: "2", metaKey: true })
    expect(usePromptStudioStore.getState().activeSubTab).toBe("prompts")
  })

  it("persists auto-sync toggle from studio settings", async () => {
    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    const autoSyncToggle = screen.getByTestId("studio-settings-auto-sync")
    autoSyncToggle.click()

    await waitFor(() => {
      expect(setPromptStudioDefaultsMock).toHaveBeenCalledWith({
        autoSyncWorkspacePrompts: false
      })
    })
  })

  it("caps the settings projects query at the backend pagination maximum", async () => {
    render(
      <MemoryRouter>
        <StudioTabContainer />
      </MemoryRouter>
    )

    await settingsProjectsQueryOptions?.queryFn?.()

    expect(promptStudioServiceMocks.listProjects).toHaveBeenCalledWith({
      page: 1,
      per_page: 100
    })
  })
})
